import unittest
from eval_center_metrics import ranking,summarize,retrieval_metrics
class PrecisionTests(unittest.TestCase):
 def test_parent_child_coverage_requires_version_snapshot_and_visible_span(self):
  question={'answerability':'answerable','review_status':'developer_self_reviewed',
    'required_facts':[{'purpose':'answer','evidence_options':[['e']]}]}
  evidence=[{'evidence_id':'e','role':'answer_support','source_path':'x',
    'source_content_hash':'hash','review_status':'developer_self_reviewed'}]
  mappings={'e':{'status':'mapped','document_version_id':'v','index_snapshot_id':'s',
    'clean_start':0,'clean_end':10,'bindings':[{'child_id':'c','parent_id':'p','clean_start':0,'clean_end':10}]}}
  row={'source':'x','source_content_hash':'hash','child_id':'c','parent_id':'p',
       'document_version_id':'v','index_snapshot_id':'s','visible_end':10}
  def score(actual,q=question):
   return retrieval_metrics(q,evidence,mappings,[row],[actual],{'candidate_k':3,'final_k':3})['parent_child_evidence_coverage']
  self.assertEqual(score(row)['value'],1)
  for key,value in [('parent_id','missing'),('document_version_id','old'),('index_snapshot_id','other'),('visible_end',9)]:
   with self.subTest(key=key):self.assertEqual(score(row|{key:value})['value'],0)
  self.assertIsNone(score(row,question|{'review_status':'proposed_unreviewed'})['value'])
 def test_precision_uses_fixed_k_and_does_not_reward_duplicates(self):
  scores=ranking(['a'],['a','a'],{'a'},5,3,'missing gold')
  self.assertEqual(scores['precision']['value'],1/3)
 def test_unreviewed_or_empty_gold_not_scored_as_precision_zero(self):
  score=ranking(['a'],['a'],set(),5,3,'unreviewed')
  self.assertIsNone(score['precision']['value']);self.assertEqual(summarize([score])['precision']['eligible'],0)
if __name__=='__main__':unittest.main(verbosity=2)
