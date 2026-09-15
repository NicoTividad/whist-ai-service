import unittest

from fastapi.testclient import TestClient

from api.app import app


class TestHealth(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def testHealthReturnsOk(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()