from unittest import TestCase
from unittest.mock import Mock,patch
from pydantic import ValidationError
from retrieval_debug import ReliableRequest,_pin_reliable_view
class HistoricalReliableTests(TestCase):
 def test_omitted_snapshot_preserves_active_pin(self):
  store=Mock();self.assertIs(_pin_reliable_view(store,None),store.pin.return_value);store.pin.assert_called_once_with()
 def test_explicit_snapshot_goes_through_published_gate(self):
  store=Mock();sid='a'*32
  with patch('knowledge_v2_api.published_view') as gate:
   self.assertIs(_pin_reliable_view(store,sid),gate.return_value);gate.assert_called_once_with(store,sid)
  store.pin.assert_not_called()
 def test_malformed_snapshot_is_rejected(self):
  with self.assertRaises(ValidationError):ReliableRequest(question='x',snapshot_id='../unsafe')
