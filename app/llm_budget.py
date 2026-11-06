"""Optional process-safe, pre-reserved CNY ledger for real model acceptance."""
import json,os,time,uuid,hashlib
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from openai import OpenAI as NativeOpenAI

_audit_stage = ContextVar('rag_llm_audit_stage', default=None)

@contextmanager
def audit_model_call(name):
    token=_audit_stage.set(name)
    try:yield
    finally:_audit_stage.reset(token)

@contextmanager
def locked(path):
    if os.name=="nt":
        import msvcrt
    else:
        import fcntl
    lock=Path(str(path)+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        deadline=time.monotonic()+10
        while True:
            try:
                stream.seek(0)
                if os.name=="nt":msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                else:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic()>=deadline:raise TimeoutError('Budget ledger lock timeout')
                time.sleep(.02)
        try:yield
        finally:
            stream.seek(0)
            if os.name=="nt":msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

class Ledger:
    def __init__(self,path):self.path=Path(path)
    def update(self,operation):
        with locked(self.path):
            state=json.loads(self.path.read_text(encoding='utf-8'))
            value=operation(state)
            temp=self.path.with_name(self.path.name+'.'+uuid.uuid4().hex+'.tmp')
            try:
                temp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,self.path)
            finally:
                if temp.exists():temp.unlink()
            return value
    def reserve(self,kwargs):
        if kwargs.get('model')!='deepseek-flash' or kwargs.get('stream'):raise ValueError('Unapproved model or unaccounted streaming')
        maximum=kwargs.get('max_tokens')
        if not isinstance(maximum,int) or not 1<=maximum<=4096:raise ValueError('Explicit bounded max_tokens required')
        # UTF-8 bytes plus schema/request overhead, intentionally above expected token count.
        size=len(json.dumps(kwargs,ensure_ascii=False).encode())+4096
        if size>100000:raise ValueError('Acceptance request input too large')
        amount=(size*2+maximum*8)/1e6;key=uuid.uuid4().hex
        def operation(state):
            if state['currency']!='CNY' or state['limit']>20:raise ValueError('Invalid cost ceiling')
            if sum(r['reserved_upper_cny'] for r in state['requests'])+amount>state['limit']:raise RuntimeError('CNY test budget exhausted')
            from request_budget import current
            budget=current.get()
            stage_name=_audit_stage.get() or (budget.events[-1]['stage'] if budget and budget.events else 'unspecified')
            state['requests'].append({'id':key,'model_requested':kwargs['model'],'status':'RESERVED','reserved_upper_cny':amount,'reserved_input_tokens':size,'max_output_tokens':maximum,'automatic_retries':0,'test_module':os.getenv('RAG_LLM_TEST_MODULE','unspecified'),'test_stage':stage_name,'created_at_utc':datetime.now(timezone.utc).isoformat(),'prompt_sha256':hashlib.sha256(json.dumps(kwargs.get('messages',[]),ensure_ascii=False,sort_keys=True).encode()).hexdigest()})
            return key
        return self.update(operation)
    def finish(self,key,**detail):
        def operation(state):next(r for r in state['requests'] if r.get('id')==key).update(detail)
        self.update(operation)

class BudgetedClient:
    def __init__(self,**kwargs):
        from types import SimpleNamespace
        from urllib.parse import urlsplit
        uri=urlsplit(str(kwargs.get('base_url','')))
        if uri.scheme!='https' or uri.hostname!='api.deepseek.com' or uri.username or uri.password or uri.query:
            raise ValueError('Unexpected real-model destination')
        kwargs['max_retries']=0
        self.native=NativeOpenAI(**kwargs);self.ledger=Ledger(os.environ['RAG_LLM_COST_LEDGER'])
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self,**kwargs):
        key=self.ledger.reserve(kwargs);start=time.perf_counter()
        extra=kwargs.setdefault('extra_body',{});extra.setdefault('thinking',{'type':'disabled'})
        try:
            response=self.native.chat.completions.create(**kwargs)
            if response.usage is None:raise ValueError('Missing real model usage')
            usage=response.usage.model_dump()
            self.ledger.finish(key,status='REAL_RESPONSE_RECEIVED',model_returned=response.model,provider_response_id=getattr(response,'id',None),usage=usage,latency_ms=(time.perf_counter()-start)*1000,usage_peak_price_upper_cny=(usage['prompt_tokens']*2+usage['completion_tokens']*8)/1e6)
            return response
        except Exception as exc:
            self.ledger.finish(key,status='FAILED_OR_UNCERTAIN_CHARGE',error_type=type(exc).__name__,latency_ms=(time.perf_counter()-start)*1000)
            raise
    def __enter__(self):return self
    def __exit__(self,*args):self.native.close()

def client_factory(**kwargs):
    return BudgetedClient(**kwargs) if os.getenv('RAG_LLM_BUDGETED_REAL')=='1' else NativeOpenAI(**kwargs)
class BudgetedChatModel:
    """Minimal invoke contract used by the existing RAG generation/HyDE paths."""
    def __init__(self,model,max_tokens=512):
        self.model_name=self.model=model;self.max_tokens=max_tokens
    def invoke(self,prompt,**kwargs):
        from langchain_core.messages import AIMessage
        if hasattr(prompt,'to_messages'):prompt=prompt.to_messages()
        if isinstance(prompt,str):messages=[{'role':'user','content':prompt}]
        else:
            roles={'human':'user','ai':'assistant','system':'system'}
            messages=[p if isinstance(p,dict) else {'role':roles.get(p.type,p.type),'content':p.content} for p in prompt]
        with audit_model_call('hyde' if self.max_tokens==256 else 'answer_generation'), BudgetedClient(base_url=os.environ['DEEPSEEK_API_BASE'],api_key=os.environ['DEEPSEEK_API_KEY'],timeout=15,max_retries=0) as client:
            response=client.chat.completions.create(model=self.model,messages=messages,temperature=0,max_tokens=self.max_tokens)
        usage=response.usage
        return AIMessage(content=response.choices[0].message.content,usage_metadata={'input_tokens':usage.prompt_tokens,'output_tokens':usage.completion_tokens,'total_tokens':usage.total_tokens},response_metadata={'model_name':response.model,'finish_reason':response.choices[0].finish_reason})
