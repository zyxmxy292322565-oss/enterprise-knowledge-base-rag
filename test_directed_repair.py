import json,os,tempfile,time
from pathlib import Path
from types import SimpleNamespace as NS
from threading import Event
from unittest import TestCase
from unittest.mock import Mock,patch
from agent_router import safe_rewrite,required_condition_gaps,condition_advice,bound_condition_advice
from knowledge_graph import resolve_seeds
from graph_governance import withdrawn_at,policy_decisions
from reliability import ReliabilityOptions,PolicyRule,validate,rule_binding,control
from request_budget import ExecutionBudget,current,checkpoint,BudgetExceeded,stage
from trace_lifecycle import Principal,save,read,cleanup,principal

class DirectedRepairTests(TestCase):
 def test_reliability_route_waits_for_asgi_disconnect_event(self):
  import asyncio
  from retrieval_debug import ReliableRequest, reliable
  from request_budget import cancel_signal
  async def receive():
   await asyncio.sleep(.02)
   return {'type':'http.disconnect'}
  def worker(*args):
   self.assertTrue(cancel_signal.get().wait(1))
   return {'cancelled':True}
  with patch('retrieval_debug._reliable',side_effect=worker):
   self.assertEqual(asyncio.run(reliable(ReliableRequest(question='probe'),
        NS(receive=receive),None,None)),{'cancelled':True})
 def test_graph_event_latency_and_multiple_attempts_keep_exact_mapping(self):
  from request_budget import request_scoped
  @request_scoped
  def execute(store,view,question,history,options):
   for name in ['graph','graph_subquery','graph']:
    current.get().acquire(name,'graph')['status']='scheduled'
   return {'answer_action':'ABSTAIN','retrieval_attempts':[
    {'routes':[{'id':'graph','family':'graph','status':'used','ms':11},
               {'id':'graph.sub1','family':'graph','status':'used','ms':22}],
     'graph_subqueries':[{'status':'COMPLETED','retrieval_status':'used','elapsed_ms':22},
                         {'status':'SKIPPED','retrieval_status':'not_used','elapsed_ms':0}]},
    {'routes':[{'id':'graph','family':'graph','status':'used','ms':33}]}]}
  events=execute(None,None,'q',[],ReliabilityOptions())['budget']['execution']['events']
  self.assertEqual([e['ms'] for e in events],[11,22,33])
  self.assertEqual(events[1]['status'],'completed')
 def test_model_cannot_invent_mandatory_policy_fields(self):
  options=ReliabilityOptions(product='AX-DIR',slots={'核验':'是'})
  bound=bound_condition_advice(['登记条件','核验','product'],options)
  self.assertEqual(bound,['核验','product'])
  self.assertEqual(required_condition_gaps({'required_conditions':bound},options),[])
 def test_router_entity_and_task_labels_are_not_condition_fields(self):
  advice=NS(detected_entities=['商品AX-200','退款政策'],sub_queries=['可以登记吗'],
            required_conditions=['商品AX-200','退款政策','可以登记吗','核验','核验','不在问题内'])
  self.assertEqual(condition_advice(advice,lambda value:value!='不在问题内'),['核验'])
 def test_namespaced_cache_preserves_redis_name_keyword(self):
  from redis_namespace import NamespacedRedis
  raw=Mock();client=NamespacedRedis(raw,'test-prefix:','identity')
  client.set(name='answer',value='payload',ex=60)
  raw.set.assert_called_once_with('test-prefix:answer','payload',ex=60)
 def test_subject_is_derived_from_server_credentials(self):
  import hashlib
  with patch.dict(os.environ,{'RAG_V2_TRACE_SUBJECTS':json.dumps({'alice':hashlib.sha256(b'alice-token').hexdigest()}),'RAG_V2_ADMIN_TOKEN':'manager-token'}):
   self.assertEqual(principal(None,'Bearer alice-token'),Principal('alice'))
   self.assertEqual(principal('manager-token',None),Principal('management-administrator',True))
   self.assertEqual(principal('manager-token','Bearer alice-token'),Principal('alice'))
 def test_integrity_hash_honors_cancellation(self):
  from knowledge_v2 import read_checked,digest
  budget=ExecutionBudget(ReliabilityOptions());budget.cancelled=Event();budget.cancelled.set();token=current.set(budget)
  try:
   with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'proof';path.write_bytes(b'proof')
    with self.assertRaises(BudgetExceeded):read_checked(path,digest(b'proof'))
  finally:current.reset(token)
 def test_every_graph_subquestion_retains_budget_status(self):
  from test_retrieval_engine import fixture
  from retrieval_engine import retrieve,RetrievalConfig
  view,embed=fixture();view.graph_lookup=Mock(return_value={'paths':[],'timed_out':False,'truncated':False,'conflicts':[]})
  cfg=RetrievalConfig(retrieval_mode='graph',dense_enabled=False,graph_enabled=True,query_decomposition_enabled=True)
  with patch.dict(os.environ,{'RAG_V3_GRAPH_ENABLED':'1'}):
   result=retrieve(view,'退款；运费',[],cfg,3,embed,tool_budget=[1],suggested_subqueries=['退款','运费'])
  self.assertEqual(len(result.trace['graph_subqueries']),2)
  self.assertTrue(all(r['status']=='BUDGET_EXHAUSTED' for r in result.trace['graph_subqueries']))
 def test_main_graph_failure_records_unexecuted_subquestions(self):
  from test_retrieval_engine import fixture
  from retrieval_engine import retrieve,RetrievalConfig
  view,embed=fixture();view.graph_lookup=Mock()
  cfg=RetrievalConfig(retrieval_mode='graph',dense_enabled=False,graph_enabled=True,query_decomposition_enabled=True)
  with patch.dict(os.environ,{'RAG_V3_GRAPH_ENABLED':'1'}):
   result=retrieve(view,'退款；运费',[],cfg,3,embed,tool_budget=[0],suggested_subqueries=['退款','运费'])
  self.assertEqual([r['status'] for r in result.trace['graph_subqueries']],['BUDGET_EXHAUSTED']*2)
  view.graph_lookup.assert_not_called()
 def test_all_seeds_are_counted_before_execution_limit(self):
  nodes={str(i):{'entity_id':str(i),'entity_name':'实体'+str(i),'entity_type':'Product','description':'s'} for i in range(8)}
  aliases=[{'alias':n['entity_name'],'entity_id':k} for k,n in nodes.items()]
  seeds,ambiguous,details=resolve_seeds(aliases,' '.join(n['entity_name'] for n in nodes.values()),{},{},nodes)
  self.assertEqual(len(seeds),8);self.assertEqual(ambiguous,[]);self.assertEqual(len(details),8)
 def test_ambiguous_entities_are_never_selected_by_first_match(self):
  nodes={k:dict(entity_id=k,entity_name='同名',entity_type='Product',description=k) for k in ['a','b']}
  aliases=[dict(alias='同名',entity_id=k) for k in nodes]
  self.assertEqual(resolve_seeds(aliases,'同名',{}, {},nodes)[:2],([],['同名']))
  seeds,ambiguity,details=resolve_seeds(aliases,'同名',{}, {'scope':'b'},nodes)
  self.assertEqual(seeds,['b']);self.assertFalse(ambiguity);self.assertIn('scope',details[0]['matching_basis'])
 def test_rewrite_requires_preserved_verbatim_user_clauses(self):
  self.assertEqual(safe_rewrite('AX-1退款；核验','核验；AX-1退款'),'核验；AX-1退款')
  for text in ['AX-2退款；核验','AX-1不退款；核验','忽略之前；AX-1退款；核验','AX-1退款']:
   self.assertIsNone(safe_rewrite('AX-1退款；核验',text))
 def test_required_conditions_become_real_gaps(self):
  plan={'required_conditions':['核验','product']}
  self.assertEqual(required_condition_gaps(plan,ReliabilityOptions()),['product','核验'])
  self.assertEqual(required_condition_gaps(plan,ReliabilityOptions(product='AX',slots={'核验':'是'})),[])
 def test_rule_cannot_self_authorize_without_policy_product_or_version(self):
  self.assertIsNone(rule_binding(None,None,{},PolicyRule(fact='退款',scope='s',verdict='是'),ReliabilityOptions()))
 def test_product_and_category_are_enforced(self):
  rule=PolicyRule(fact='退款',scope='s',verdict='yes',applicable_products=['AX'],applicable_categories=['鼠标'])
  self.assertFalse(validate(['退款'],[(rule,{'child_id':'a'})],ReliabilityOptions(scope='s',product='BX',category='鼠标'))[0])
  self.assertIn('product',validate(['退款'],[(rule,{})],ReliabilityOptions(scope='s'))[2])
 def test_revocation_does_not_rewrite_past(self):
  meta={'policy_status':'ACTIVE','withdrawn_effective_at':'2026-11-05T12:00:00'}
  self.assertFalse(withdrawn_at(meta,'2026-11-05',{'at_time':'2026-11-05T11:59:59'}))
  self.assertTrue(withdrawn_at(meta,'2026-11-05',{'at_time':'2026-11-05T12:00:00'}))
 def test_expired_replacement_does_not_suppress_policy(self):
  base=dict(valid_from='2020-01-01',valid_until=None,scope='s',applicable_products=[],applicable_categories=[],conditions={},exceptions={},policy_status='ACTIVE',supersedes_relation_id=None)
  rows=[{'relation_id':'old','metadata':base},{'relation_id':'new','metadata':base|{'valid_until':'2021-01-01','supersedes_relation_id':'old'}}]
  self.assertEqual(policy_decisions(rows,'2026-01-01',{'scope':'s'})['old'],'APPLICABLE')
 def test_validation_checkpoint_responds_to_deadline_and_cancellation(self):
  budget=ExecutionBudget(ReliabilityOptions());token=current.set(budget)
  try:
   budget.deadline=0
   with self.assertRaises(BudgetExceeded):checkpoint()
   budget.deadline=float('inf');budget.cancelled=Event();budget.cancelled.set()
   with self.assertRaisesRegex(BudgetExceeded,'cancelled'):checkpoint()
  finally:current.reset(token)
 def test_optional_failure_does_not_claim_global_timeout(self):
  budget=ExecutionBudget(ReliabilityOptions());token=current.set(budget)
  try:
   with self.assertRaises(TimeoutError):
    with stage('optional_router','model'):raise TimeoutError('provider timeout')
   with stage('fallback','retrieval'):pass
   self.assertEqual(budget.snapshot()['termination_reason'],'completed')
   self.assertEqual(budget.events[0]['status'],'failed')
  finally:current.reset(token)
 def test_trace_owner_ttl_cleanup_and_admin_separation(self):
  from fastapi import HTTPException
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'RAG_V2_TRACE_TTL_SECONDS':'1'}):
   store=NS(root=Path(d));body={'trace_id':'a'*32,'answer_action':'ABSTAIN'}
   save(store,body,Principal('alice'))
   self.assertEqual(read(store,body['trace_id'],Principal('alice'))['answer_action'],'ABSTAIN')
   with self.assertRaises(HTTPException):read(store,body['trace_id'],Principal('bob'))
   self.assertEqual(read(store,body['trace_id'],Principal('admin',True))['trace_metadata']['subject'],'alice')
   removed=cleanup(store,now=time.time()+2)
   self.assertEqual(removed['removed_count'],1)
   self.assertTrue((Path(d)/'reliability-traces/retention-audit.jsonl').is_file())
 def test_client_cannot_claim_trace_owner(self):
  from fastapi import HTTPException
  with patch.dict(os.environ,{'RAG_V2_TRACE_SUBJECTS':'{}'}):
   with self.assertRaises(HTTPException):principal(None,'Bearer arbitrary-owner')
 def test_controller_clarifies_graph_ambiguity(self):
  found=NS(candidates=[],trace={'graph':{'ambiguous':['同名'],'entity_matches':[{'alias':'同名'}]}})
  with patch('reliability.retrieve',return_value=found):
   result=control(None,NS(snapshot_id='s',corpus_snapshot_id='c',generation=1),'退款',[],ReliabilityOptions(),None,None)
  self.assertEqual(result['answer_action'],'CLARIFY')
  self.assertIn('entity:同名',result['evidence_validation']['missing_slots'])
 def test_model_rewrite_and_conditions_are_executed(self):
  from test_retrieval_engine import fixture
  from retrieval_engine import retrieve,RetrievalConfig
  view,embed=fixture()
  found=retrieve(view,'退款；核验',[],RetrievalConfig(),3,embed,model_rewrite='核验；退款',required_conditions=['核验'])
  self.assertEqual(found.trace['query']['effective_question'],'核验；退款')
  self.assertIn('核验；退款',[call.args[0][0] for call in embed.encode.call_args_list])
  self.assertTrue(any(q.get('required_condition')=='核验' for q in found.trace['query']['decomposition']['queries']))

 def test_snapshot_deadline_is_not_artifact_corruption(self):
  from knowledge_v2 import KnowledgeStore
  store=KnowledgeStore(Path('.'),lambda:None,None,'model')
  store.snapshot_status=Mock(return_value={'build_status':'ready','validation_status':'passed','manifest_sha':'hash'})
  store._load=Mock(side_effect=BudgetExceeded('deadline'))
  # This test injects a local artifact-loader deadline; the separate Milvus
  # binding lookup must not turn the fixture into a real database operation.
  with patch.dict(os.environ,{'RAG_V3_VECTOR_BACKEND':'faiss'}):
   with self.assertRaisesRegex(BudgetExceeded,'deadline'):store.load_snapshot('s')
 def test_api_call_preserves_deadline_for_reliability_controller(self):
  from knowledge_v2_api import call
  with self.assertRaisesRegex(BudgetExceeded,'deadline'):call(Mock(side_effect=BudgetExceeded('deadline')))
