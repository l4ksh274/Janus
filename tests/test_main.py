import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["NVIDIA_API_KEY"] = "test-api-key"
os.environ["BASE_URL"] = "https://api.example.test/v1"

from fastapi.testclient import TestClient

from janus.main import app


class FakeResponse:
    def __init__(self, status_code=200, payload=None, chunks=()):
        self.status_code = status_code
        self.payload = payload or {"id": "resp_test", "output": []}
        self.chunks = chunks

    def json(self):
        return self.payload

    async def aiter_raw(self):
        for chunk in self.chunks:
            yield chunk


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.init_args = None
        self.init_kwargs = None
        self.closed = False
        self.post_call = None
        self.stream_call = None
        self.stream_closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.closed = True

    async def post(self, path, **kwargs):
        self.post_call = (path, kwargs)
        return self.response

    def stream(self, method, path, **kwargs):
        self.stream_call = (method, path, kwargs)
        return FakeStreamContext(self)


class FakeStreamContext:
    def __init__(self, client):
        self.client = client

    async def __aenter__(self):
        return self.client.response

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.client.stream_closed = True


class MainTests(unittest.TestCase):
    def setUp(self):
        self.response = FakeResponse(
            status_code=429,
            payload={"error": {"message": "rate limited"}},
            chunks=(b"data: {\"type\":\"response.created\"}\n\n", b"data: [DONE]\n\n"),
        )
        self.client = FakeClient(self.response)

    def start_client(self):
        return patch("janus.main.AsyncClient", return_value=self.client)

    def test_lifespan_configures_and_closes_upstream_client(self):
        with self.start_client() as client_factory:
            with TestClient(app):
                client_factory.assert_called_once()
                _, kwargs = client_factory.call_args
                self.assertEqual(kwargs["base_url"], "https://api.example.test/v1")
                self.assertEqual(
                    kwargs["headers"],
                    {
                        "Authorization": "Bearer test-api-key",
                        "Accept": "application/json",
                    },
                )
                self.assertEqual(kwargs["timeout"].connect, 10.0)
                self.assertEqual(kwargs["timeout"].read, 120.0)
                self.assertIs(app.state.asyncClient, self.client)
            self.assertTrue(self.client.closed)

    def test_json_request_is_sanitized_and_upstream_status_is_forwarded(self):
        with self.start_client():
            with TestClient(app) as client:
                response = client.post(
                    "/v1/responses",
                    json={
                        "model": "openai/gpt-oss-20b",
                        "input": "Hello",
                        "metadata": None,
                    },
                )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json(), {"error": {"message": "rate limited"}})
        self.assertEqual(self.client.post_call[0], "/responses")
        self.assertEqual(
            self.client.post_call[1]["json"],
            {"model": "openai/gpt-oss-20b", "input": "Hello", "stream": False},
        )

    def test_streaming_request_forwards_raw_chunks_as_server_sent_events(self):
        with self.start_client():
            with TestClient(app) as client:
                response = client.post(
                    "/v1/responses",
                    json={
                        "model": "openai/gpt-oss-20b",
                        "input": "Hello",
                        "stream": True,
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertEqual(
            response.content,
            b"data: {\"type\":\"response.created\"}\n\ndata: [DONE]\n\n",
        )
        self.assertEqual(self.client.stream_call[0:2], ("POST", "/responses"))
        self.assertTrue(self.client.stream_closed)

    def test_invalid_request_is_rejected_without_upstream_call(self):
        with self.start_client():
            with TestClient(app) as client:
                response = client.post("/v1/responses", json={"input": "Hello"})

        self.assertEqual(response.status_code, 422)
        self.assertIsNone(self.client.post_call)
        self.assertIsNone(self.client.stream_call)

    def test_import_exits_when_configuration_is_missing(self):
        source_path = Path(__file__).resolve().parents[1] / "src"
        invalid_configurations = (
            {"NVIDIA_API_KEY": None, "BASE_URL": "https://api.example.test/v1"},
            {"NVIDIA_API_KEY": "", "BASE_URL": "https://api.example.test/v1"},
            {"NVIDIA_API_KEY": "test-api-key", "BASE_URL": None},
            {"NVIDIA_API_KEY": "test-api-key", "BASE_URL": ""},
        )
        for configuration in invalid_configurations:
            with self.subTest(configuration=configuration):
                with tempfile.TemporaryDirectory() as working_directory:
                    environment = os.environ.copy()
                    environment.pop("NVIDIA_API_KEY", None)
                    environment.pop("BASE_URL", None)
                    for name, value in configuration.items():
                        if value is not None:
                            environment[name] = value
                    environment["PYTHONPATH"] = str(source_path)
                    result = subprocess.run(
                        [sys.executable, "-c", "import janus.main"],
                        cwd=working_directory,
                        env=environment,
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                self.assertEqual(result.returncode, 1)
                self.assertIn(
                    "Environment variables haven't been set up", result.stdout
                )


if __name__ == "__main__":
    unittest.main()
