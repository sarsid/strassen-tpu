"""A suppressed SDK timeout is not an acknowledged keepalive response."""
import contextlib
import io
import json
from pathlib import Path
import sys
import types
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import colab_control_v004 as control


class Tests(unittest.TestCase):
    def client(self, action, suppress=False):
        session=types.SimpleNamespace(request=action)
        def keepalive(endpoint):
            try:session.request('GET','https://example.test/tun/m/'+endpoint+'/keep-alive/',headers={'Authorization':'secret-not-logged'})
            except TimeoutError:
                if not suppress:raise
        return types.SimpleNamespace(session=session,keep_alive_assignment=keepalive)

    def test_acknowledged_http_success(self):
        client=self.client(lambda *a,**k:types.SimpleNamespace(status_code=200))
        original=client.session.request
        row=control.observed_keepalive(client,'owned')
        self.assertTrue(row['acknowledged']);self.assertEqual(row['http_status'],200)
        self.assertIs(client.session.request,original)
        self.assertNotIn('secret-not-logged',json.dumps(row))

    def test_http_failure_not_success(self):
        row=control.observed_keepalive(self.client(lambda *a,**k:types.SimpleNamespace(status_code=403)),'owned')
        self.assertFalse(row['acknowledged']);self.assertEqual(row['http_status'],403)

    def test_suppressed_timeout_remains_uncertain(self):
        def timeout(*a,**k):raise TimeoutError('private error message not logged')
        row=control.observed_keepalive(self.client(timeout,suppress=True),'owned')
        self.assertFalse(row['acknowledged']);self.assertEqual(row['outcome'],'transport_exception')
        self.assertEqual(row['error_type'],'TimeoutError');self.assertNotIn('private',json.dumps(row))

    def test_unsuppressed_exception_recorded(self):
        def failed(*a,**k):raise OSError('private error message not logged')
        row=control.observed_keepalive(self.client(failed),'owned')
        self.assertFalse(row['acknowledged']);self.assertEqual(row['raised_error']['error_type'],'OSError')
        self.assertNotIn('private',json.dumps(row))


if __name__=='__main__':unittest.main(verbosity=2)
