# Janus

Janus is an OpenAI-compatible proxy for NVIDIA's developer API.

## Tests

Run the local tests with the project virtual environment:

```sh
. .venv/bin/activate
python -m unittest discover -s tests
```

Start the Janus server with `NVIDIA_API_KEY` and `BASE_URL` configured in its
environment (or `.env` file). The live check connects over HTTP to the running
proxy at `http://127.0.0.1:8000/v1/responses`; set `JANUS_PROXY_URL` or pass
`--proxy-url` to use a different server URL. It sends a real request using the
`openai/gpt-oss-20b` model, which may incur API usage charges. The default
checks a non-streaming request; use `--stream` to check streaming instead:

```sh
python tests/run_live_api.py
python tests/run_live_api.py --stream
python tests/run_live_api.py --proxy-url http://127.0.0.1:8080
```
