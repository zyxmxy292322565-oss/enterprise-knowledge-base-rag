from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from hashlib import sha256
from milvus_recovery import verified_vector_bytes
from knowledge_v2 import KnowledgeError
from request_budget import BudgetExceeded
class RecoveryTests(TestCase):
 def setup_files(self,root):
  folder=Path(root)/'snapshots'/('a'*32);folder.mkdir(parents=True);(folder/'index.faiss').write_bytes(b'damaged')
  recovery=Path(root)/'recovery';backup=recovery/('a'*32)/'index.faiss';backup.parent.mkdir(parents=True);backup.write_bytes(b'original')
  return folder,recovery,{'snapshot_id':'a'*32,'index_sha':sha256(b'original').hexdigest()}
 def test_recovery_requires_explicit_flag(self):
  with TemporaryDirectory() as d:
   f,r,m=self.setup_files(d)
   with self.assertRaises(KnowledgeError):verified_vector_bytes(f,m,r)
 def test_verified_backup_returned_source_unchanged(self):
  with TemporaryDirectory() as d:
   f,r,m=self.setup_files(d);self.assertEqual(verified_vector_bytes(f,m,r,True),(b'original',True));self.assertEqual((f/'index.faiss').read_bytes(),b'damaged')
 def test_mismatching_backup_is_rejected(self):
  with TemporaryDirectory() as d:
   f,r,m=self.setup_files(d);(r/('a'*32)/'index.faiss').write_bytes(b'wrong')
   with self.assertRaises(KnowledgeError):verified_vector_bytes(f,m,r,True)
 def test_cancellation_is_not_hidden_by_recovery(self):
  with TemporaryDirectory() as d:
   f,r,m=self.setup_files(d)
   with patch('milvus_recovery.read_checked',side_effect=BudgetExceeded('cancelled')):
    with self.assertRaises(BudgetExceeded):verified_vector_bytes(f,m,r,True)
