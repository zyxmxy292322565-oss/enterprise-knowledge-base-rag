import unittest,numpy as np
from milvus_store import collection_name,checked_vectors,metadata_rows,MilvusSnapshotIndex
class MilvusInputTests(unittest.TestCase):
 def test_identity_rejects_foreign_and_injection(self):
  self.assertEqual(collection_name('ragv3audit','a'*32),'ragv3audit_'+'a'*32)
  for ns,sid in [('other/','a'*32),('ragv3audit','../manifest')]:
   with self.assertRaises(ValueError):collection_name(ns,sid)
 def test_vector_shape_and_nonfinite(self):
  for value in [[[1,2,3]],[[float('nan'),1]]]:
   with self.assertRaises(ValueError):checked_vectors(value,2)
  np.testing.assert_array_equal(checked_vectors([[3,4]],2),[[3,4]])
 def test_complete_identity_digest_and_no_normalization(self):
  child=dict(child_id='c',parent_id='p',document_version_id='v',file_hash='f',source='sample.txt',row=0)
  a=metadata_rows('a'*32,[child],[[3,4]])[0]
  b=metadata_rows('a'*32,[child|{'document_version_id':'v2'}],[[3,4]])[0]
  self.assertNotEqual(a['record_sha256'],b['record_sha256']);self.assertEqual(a['vector'],[3,4])
  with self.assertRaises(ValueError):metadata_rows('a'*32,[child,child],[[3,4],[3,4]])
 def test_forbidden_endpoint_rejected_before_client_import(self):
  for uri in ['http://remote:19531','http://localhost:19530','./milvus.db']:
   with self.assertRaises(ValueError):MilvusSnapshotIndex.connect(uri,'ragv3audit','a'*32,2,0)
 def test_empty_result_padding_preserves_legacy_contract(self):
  index=MilvusSnapshotIndex(None,'ragv3audit_'+'a'*32,'a'*32,2,0)
  distances,rows=index.search([[0,0]],3)
  np.testing.assert_array_equal(rows,[[-1,-1,-1]])
  np.testing.assert_array_equal(distances,np.full((1,3),np.finfo(np.float32).max,dtype=np.float32))
if __name__=='__main__':unittest.main(verbosity=2)
