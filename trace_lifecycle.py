"""Server-authenticated trace ownership, expiry and content-free retention audit."""
from dataclasses import dataclass
from hashlib import sha256
import hmac,json,os,re,time
from threading import RLock
from fastapi import Header,HTTPException,Request
from knowledge_v2 import json_bytes,write_file

_lock=RLock()
@dataclass(frozen=True)
class Principal:
    subject: str
    administrator: bool=False

def principal(x_rag_v2_token: str | None=Header(default=None),authorization: str | None=Header(default=None)):
    manager=os.getenv('RAG_V2_ADMIN_TOKEN','')
    token=(authorization or '').removeprefix('Bearer ')
    digest=sha256(token.encode()).hexdigest()
    subjects=json.loads(os.getenv('RAG_V2_TRACE_SUBJECTS','{}'))
    for subject,expected in subjects.items():
        if token and hmac.compare_digest(digest,expected):return Principal(subject)
    if authorization:raise HTTPException(403,'Invalid subject credential')
    if manager and hmac.compare_digest(manager,x_rag_v2_token or ''):return Principal('management-administrator',True)
    raise HTTPException(403,'Authenticated trace subject required')

def authenticated_store(request: Request):
    store=getattr(request.app.state,'knowledge',None)
    if store is None:raise HTTPException(404,'Knowledge management is disabled')
    return store

def folder(store):
    path=store.root/'reliability-traces';path.mkdir(exist_ok=True);return path

def _audit(store,summary):
    with (folder(store)/'retention-audit.jsonl').open('ab') as f:
        f.write(json_bytes(summary)+b'\n');f.flush();os.fsync(f.fileno())

def save(store,body,owner):
    if not isinstance(owner,Principal):raise RuntimeError('Trace requires an authenticated principal')
    ttl=int(os.getenv('RAG_V2_TRACE_TTL_SECONDS','3600'))
    if not 1<=ttl<=30*86400:raise RuntimeError('Invalid trace TTL')
    value=body|{'trace_metadata':{'subject':owner.subject,'created_at':time.time(),'expires_at':time.time()+ttl,'ttl_seconds':ttl}}
    with _lock:
        path=folder(store)/(body['trace_id']+'.json');temporary=path.with_suffix('.tmp')
        write_file(temporary,json_bytes(value));temporary.replace(path)

def read(store,trace_id,owner):
    if not re.fullmatch('[a-f0-9]{32}',trace_id):raise HTTPException(404,'Trace not found')
    with _lock:
        path=folder(store)/(trace_id+'.json')
        if not path.is_file():raise HTTPException(404,'Trace not found')
        body=json.loads(path.read_bytes());meta=body.get('trace_metadata',{})
        if not owner.administrator and meta.get('subject')!=owner.subject:raise HTTPException(404,'Trace not found')
        # Historic unowned traces stay visible to administrators for compatibility only.
        expiry=meta['expires_at'] if meta else path.stat().st_mtime+3600
        if expiry<=time.time():raise HTTPException(410,'Trace expired')
        if owner.administrator:_audit(store,{'event':'administrator_read','trace_id':trace_id,'subject':owner.subject,'time':time.time()})
        return body

def cleanup(store,now=None):
    cutoff=time.time() if now is None else now;removed=[]
    with _lock:
        for path in folder(store).glob('*.json'):
            if not re.fullmatch('[a-f0-9]{32}',path.stem):continue
            raw=path.read_bytes();body=json.loads(raw);meta=body.get('trace_metadata')
            # Legacy files receive a bounded lifetime based on filesystem creation time.
            expiry=meta['expires_at'] if meta else path.stat().st_mtime+3600
            if expiry>cutoff:continue
            _audit(store,{'event':'trace_expired','trace_id':path.stem,'subject':meta.get('subject') if meta else 'legacy-unowned',
                'content_sha256':sha256(raw).hexdigest(),'answer_action':body.get('answer_action'),
                'snapshot_id':body.get('index_snapshot_id'),'expired_at':expiry,'removed_at':cutoff})
            path.unlink();removed.append(path.stem)
    return {'removed_count':len(removed),'trace_ids':removed}
