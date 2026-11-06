import tempfile,json,unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from llm_budget import Ledger,audit_model_call
class BudgetTests(unittest.TestCase):
 def test_module_stage_and_timestamp_are_preserved_without_prompt_text(self):
  with tempfile.TemporaryDirectory() as d:
   ledger=self.ledger(d,20)
   with audit_model_call('kg_extract'):
    ledger.reserve({'model':'deepseek-flash','max_tokens':256,'messages':[{'role':'user','content':'test proof'}]})
   row=json.loads(ledger.path.read_text())['requests'][0]
   self.assertEqual(row['test_stage'],'kg_extract')
   self.assertTrue(row['created_at_utc'].endswith('+00:00'))
   self.assertEqual(len(row['prompt_sha256']),64)
   self.assertNotIn('messages',row)
 def ledger(self,folder,limit=.02):
  p=Path(folder)/'ledger.json';p.write_text(json.dumps({'currency':'CNY','limit':limit,'requests':[]}));return Ledger(p)
 def test_concurrent_reservations_never_exceed_limit(self):
  with tempfile.TemporaryDirectory() as d:
   ledger=self.ledger(d)
   def reserve(i):
    try:ledger.reserve({'model':'deepseek-flash','max_tokens':256,'messages':[]});return True
    except RuntimeError:return False
   with ThreadPoolExecutor(8) as pool:results=list(pool.map(reserve,range(16)))
   state=json.loads(ledger.path.read_text());self.assertLessEqual(sum(r['reserved_upper_cny'] for r in state['requests']),.02);self.assertTrue(any(results));self.assertFalse(all(results))
 def test_uncertain_failure_preserves_reservation(self):
  with tempfile.TemporaryDirectory() as d:
   ledger=self.ledger(d,20);key=ledger.reserve({'model':'deepseek-flash','max_tokens':256});ledger.finish(key,status='FAILED_OR_UNCERTAIN_CHARGE')
   self.assertGreater(json.loads(ledger.path.read_text())['requests'][0]['reserved_upper_cny'],0)
 def test_unapproved_model_stream_and_output_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   ledger=self.ledger(d)
   for kwargs in [{'model':'other','max_tokens':1},{'model':'deepseek-flash','max_tokens':1,'stream':True},{'model':'deepseek-flash','max_tokens':999999}]:
    with self.assertRaises(ValueError):ledger.reserve(kwargs)
if __name__=='__main__':unittest.main(verbosity=2)
