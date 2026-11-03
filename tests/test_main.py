import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
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
        self.headers = {"content-type": "application/json"}
        self.closed = False

    @property
    def content(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def json(self):
        return self.payload

    async def aread(self) -> bytes:
        return self.content

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk

    async def aiter_raw(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


class FakeRedis:
    def __init__(self):
        self.pinged = False
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.closed = True

    async def ping(self):
        self.pinged = True
        return True


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.base_url = "https://api.example.test/v1/"
        self.closed = False
        self.post_call = None
        self.stream_call = None

    @property
    def stream_closed(self) -> bool:
        return self.response.closed

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.closed = True

    async def post(self, path, **kwargs):
        self.post_call = (path, kwargs)
        return self.response

    def build_request(self, method, path, **kwargs):
        return SimpleNamespace(
            method=method,
            path=path,
            kwargs=kwargs,
            url=f"https://api.example.test/v1/{path.lstrip('/')}",
        )

    async def send(self, req, stream=False):
        self.stream_call = (req.method, req.path, req.kwargs, stream)
        return self.response


class MainTests(unittest.TestCase):
    def setUp(self):
        self.response = FakeResponse(
            status_code=429,
            payload={"error": {"message": "rate limited"}},
            chunks=(b'data: {"type":"response.created"}\n\n', b"data: [DONE]\n\n"),
        )
        self.client = FakeClient(self.response)
        self.redis = FakeRedis()

    def start_dependencies(self):
        stack = ExitStack()
        client_factory = stack.enter_context(
            patch("janus.main.AsyncClient", return_value=self.client)
        )
        redis_factory = stack.enter_context(
            patch("janus.main.Redis.from_url", return_value=self.redis)
        )
        return stack, client_factory, redis_factory

    def test_lifespan_configures_and_closes_upstream_client_and_redis(self):
        stack, client_factory, redis_factory = self.start_dependencies()
        with stack:
            with TestClient(app):
                client_factory.assert_called_once()
                _, kwargs = client_factory.call_args
                self.assertIn(
                    kwargs["base_url"],
                    ("https://api.example.test/v1", "https://api.example.test/v1/"),
                )
                self.assertEqual(
                    kwargs["headers"]["Authorization"], "Bearer test-api-key"
                )
                self.assertEqual(kwargs["timeout"].connect, 10.0)
                self.assertEqual(kwargs["timeout"].read, 120.0)
                self.assertIs(app.state.http, self.client)
                self.assertIs(app.state.redis, self.redis)
                self.assertTrue(self.redis.pinged)
                redis_factory.assert_called_once()
            self.assertTrue(self.client.closed)
            self.assertTrue(self.redis.closed)

    def test_json_request_is_sanitized_and_upstream_status_is_forwarded(self):
        stack, _, _ = self.start_dependencies()
        with stack:
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
        self.assertIn(self.client.post_call[0], ("responses", "/responses"))
        self.assertEqual(
            self.client.post_call[1]["json"],
            {"model": "openai/gpt-oss-20b", "input": "Hello", "stream": False},
        )

    def test_streaming_request_forwards_raw_chunks_as_server_sent_events(self):
        self.response.status_code = 200
        stack, _, _ = self.start_dependencies()
        with stack:
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
        self.assertEqual(
            response.headers["content-type"], "text/event-stream; charset=utf-8"
        )
        self.assertEqual(
            response.content,
            b'data: {"type":"response.created"}\n\ndata: [DONE]\n\n',
        )
        self.assertEqual(self.client.stream_call[0], "POST")
        self.assertIn(self.client.stream_call[1], ("responses", "/responses"))
        self.assertTrue(self.client.stream_call[3])
        self.assertTrue(self.client.stream_closed)

    def test_invalid_request_is_rejected_without_upstream_call(self):
        stack, _, _ = self.start_dependencies()
        with stack:
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
                    "NVIDIA_API_KEY and BASE_URL must be set",
                    result.stderr + result.stdout,
                )


if __name__ == "__main__":
    unittest.main()