from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock
import faiss
import numpy as np
from knowledge_v2 import KnowledgeStore, KnowledgeError, process_document, ProcessingConfig, digest, json_bytes
from knowledge_graph import validate_graph

class SnapshotBoundaryTests(TestCase):
    def test_self_consistent_manifest_rejects_invalid_parent_and_version(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            index=faiss.IndexFlatL2(2)
            index.add(np.zeros((1,2),dtype=np.float32))
            encoded=faiss.serialize_index(index).tobytes()
            (root/'index.faiss').write_bytes(encoded)
            processed=process_document('fixture.txt',b'evidence text','v1',ProcessingConfig())
            child=processed['children'][0] | {'row':0,'source':'fixture.txt','file_hash':'hash','version':'v1'}
            baseline={'snapshot_id':'s','corpus_snapshot_id':'c','model_identity':'model','index_sha':digest(encoded),'dimension':2,
                      'parents':processed['parents'],'children':[child],
                      'members':[{'document_version_id':'v1','source':'fixture.txt','legacy_file_hash':'hash','policy_version':'v1'}]}
            store=KnowledgeStore(root,lambda:None,SimpleNamespace(get_sentence_embedding_dimension=lambda:2),'model',ProcessingConfig())
            for fault in ('missing_parent','wrong_parent','wrong_version','wrong_snapshot'):
                bad=deepcopy(baseline)
                if fault=='missing_parent':bad['parents']=[]
                elif fault=='wrong_parent':bad['children'][0]['parent_id']='missing'
                elif fault=='wrong_version':bad['children'][0]['document_version_id']='v2'
                else:bad['snapshot_id']='other'
                raw=json_bytes(bad)
                (root/'manifest.json').write_bytes(raw)
                with self.subTest(fault=fault),self.assertRaises(KnowledgeError):
                    store._load(root,'s',digest(raw),1)

    def test_graph_corpus_and_schema_are_validated(self):
        view=SimpleNamespace(snapshot_id='s',corpus_snapshot_id='c',all_docs=[])
        for responses in ([{'corpus_id':'other'}],[{'corpus_id':'c'},None],[{'corpus_id':'c'},{'schema_version':'unsupported'}]):
            cursor=Mock()
            cursor.fetchone.side_effect=responses
            with self.subTest(responses=responses),self.assertRaises(KnowledgeError):
                validate_graph(None,view,cursor)
