from pathlib import Path
from unittest import TestCase
from model_fingerprint import model_receipt


class ModelFingerprintTests(TestCase):
    def rows(self, root, tokenizer='b'*64):
        return [{'path':str(root/'model.safetensors'),'sha256':'a'*64},
                {'path':str(root/'tokenizer.json'),'sha256':tokenizer}]

    def test_relocation_preserves_identity_but_tokenizer_change_does_not(self):
        a, b = Path('old-model'), Path('new-model')
        original = model_receipt('BGE',a,self.rows(a))
        moved = model_receipt('BGE',b,self.rows(b))
        changed = model_receipt('BGE',b,self.rows(b,'c'*64))
        self.assertEqual(original['revision'],moved['revision'])
        self.assertNotEqual(original['location'],moved['location'])
        self.assertNotEqual(original['revision'],changed['revision'])

    def test_missing_weights_cannot_be_reported_as_verified_identity(self):
        with self.assertRaises(ValueError):
            model_receipt('BGE',Path('.'),[])
