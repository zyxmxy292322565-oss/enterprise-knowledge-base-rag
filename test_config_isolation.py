import os
from unittest import TestCase
from unittest.mock import patch
from isolated_config import require_target
from db_utils import get_connection
from redis_utils import get_redis_client

class IsolationTests(TestCase):
    def test_wrong_targets_fail_before_network(self):
        for service,key,value in [('mysql','MYSQL_PORT','3306'),('mysql','MYSQL_DATABASE','rag_demo'),('mysql','RAG_V2_MYSQL_UUID','wrong'),('redis','REDIS_PORT','6379')]:
            with self.subTest(key=key), patch.dict(os.environ,{key:value}):
                with self.assertRaises(RuntimeError):
                    require_target(service)

    def test_missing_values_fail(self):
        for service,key in [('mysql','MYSQL_DATABASE'),('redis','REDIS_HOST')]:
            with self.subTest(key=key), patch.dict(os.environ,{key:''}):
                with self.assertRaises(RuntimeError):
                    require_target(service)

    def test_redis_identity_mismatch_closes_client(self):
        from unittest.mock import Mock
        client = Mock()
        client.get.return_value = "wrong-instance"
        with patch("redis_utils.require_target"), patch("redis_utils.redis.Redis", return_value=client), patch.dict(os.environ, {"RAG_V2_REDIS_IDENTITY": "expected-instance"}):
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                get_redis_client()
        client.close.assert_called_once()
