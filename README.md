# Janus

Janus is an OpenAI-compatible **semantic caching proxy**. It sits between your
application and any OpenAI-compatible API provider, transparently caching
responses to reduce latency, cost, and redundant upstream calls.

```
┌─────────┐         ┌───────────────────┐         ┌──────────────┐
│  Client  │ ──────▸ │      Janus        │ ──────▸ │  Upstream    │
│  (app)   │ ◂────── │  (cache proxy)    │ ◂────── │  LLM API    │
└─────────┘         └────────┬──────────┘         └──────────────┘
                             │
                     ┌───────▼───────┐
                     │  Redis Stack  │
                     │  (vector DB)  │
                     └───────────────┘
```

## Features

### Two-tier caching

- **L1 — Exact match**: SHA-256 hash of the full request payload. Identical
  requests return the cached response instantly.
- **L2 — Semantic match**: Vector-similarity search using
  [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)
  embeddings (768-dim, runs locally via
  [FastEmbed](https://github.com/qdrant/fastembed) — no external embedding API
  required). Paraphrased prompts can hit the cache when the cosine distance is
  below the configured threshold.

### False-positive prevention

Cosine similarity alone cannot distinguish semantically similar but
functionally different prompts (e.g. *"enable auth"* vs *"disable auth"*).
Janus applies a **deterministic validation layer** after vector retrieval:

| Check | Catches |
| --- | --- |
| Number extraction | "5 items" ≠ "50 items" |
| Negation detection | "can access" ≠ "cannot access" |
| Antonym detection (16 pairs) | "enable" ≠ "disable", "create" ≠ "delete", etc. |
| Quoted entity comparison | `"alice"` ≠ `"bob"` |

Any detected mismatch results in a cache miss — never an unsafe cache hit.

### Additional capabilities

- **Dual API support** — proxies both `/v1/responses` (Responses API) and
  `/v1/chat/completions` (Chat Completions API).
- **Streaming** — full SSE pass-through with buffered caching on stream
  completion.
- **Request coalescing** — deduplicates concurrent identical requests into a
  single upstream call.
- **Tenant isolation** — each API key gets its own cache namespace via
  `scope = sha256(bearer_token)[:16]`.
- **Parameter fingerprinting** — requests with different models, temperatures,
  system prompts, tools, or generation parameters never share cache entries.
- **Safety-first cacheability** — only deterministic (`temperature=0`),
  text-only, single-turn, tool-free requests are eligible for semantic caching.
- **Prometheus metrics** — exposes `requests_total`, `cache_hit_total`,
  `request_latency_seconds`, and `estimated_cost_saved_gbp` at `/metrics`.
- **Grafana dashboards** — pre-provisioned Prometheus data source via Docker
  Compose.
- **Load testing** — included [Locust](https://locust.io/) load test file.

## Quick start with Docker Compose

Requires Docker Engine and the Compose plugin.

**1. Create your environment file:**

```sh
cp .env.example .env
```

**2. Edit `.env`** with your upstream API key, base URL, and proxy key(s) — see
[Environment variables](#environment-variables) below.

**3. Start the stack:**

```sh
docker compose up --build
```

The first startup downloads the embedding model and may take a few minutes.

| Service | URL |
| --- | --- |
| Janus API | <http://localhost:8000> |
| Prometheus | <http://localhost:9090> |
| Grafana | <http://localhost:3000> |
| RedisInsight | <http://localhost:8001> |

Log in to Grafana with username `admin` and the `GRAFANA_ADMIN_PASSWORD` from
`.env`.

**Useful commands:**

```sh
docker compose up --build -d          # Run in background
docker compose logs -f janus          # Follow Janus logs
docker compose exec redis redis-cli ping  # Verify Redis
docker compose down                   # Stop services
docker compose down --volumes         # Stop and delete all data
```

## Local development

Requires Python 3.10+, [uv](https://docs.astral.sh/uv/), and a Redis Stack
instance.

**1. Install dependencies:**

```sh
uv sync
```

**2. Configure:**

```sh
cp .env.example .env
cp config/settings.example.json config/settings.json
```

Edit `.env` with your credentials (see
[Environment variables](#environment-variables)).

**3. Start Redis Stack:**

```sh
docker run -d --name janus-redis -p 6379:6379 redis/redis-stack:latest
```

**4. Run Janus:**

```sh
uv run fastapi run --host 0.0.0.0 --port 8000
```

## Usage

Send requests to Janus exactly as you would to the upstream API. Authenticate
with one of the keys defined in `PROXY_KEYS`.

**Responses API:**

```sh
curl http://localhost:8000/v1/responses \
  -H "Authorization: Bearer YOUR_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "input": "What is the capital of France?",
    "temperature": 0
  }'
```

**Chat Completions API:**

```sh
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "temperature": 0
  }'
```

The response includes an `X-Cache` header indicating the cache result:

| `X-Cache` value | Meaning |
| --- | --- |
| `HIT-EXACT` | Exact match from L1 cache |
| `HIT-SEMANTIC` | Semantic match from L2 cache |
| `MISS` | Cache miss — forwarded to upstream |
| `BYPASS` | Request was not eligible for caching |

## Configuration

### Environment variables

Set these in `.env` (local) or via Docker Compose.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | Yes | — | API key for the upstream provider. `NVIDIA_API_KEY` is accepted as an alias. |
| `BASE_URL` | Yes | — | Base URL of the upstream API (e.g. `https://api.openai.com/v1`). |
| `PROXY_KEYS` | Yes | — | Comma-separated bearer keys that clients use to authenticate with Janus. `PROXY_API_KEYS` is an alias. |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis connection URL. Compose overrides this to `redis://redis:6379`. |
| `CONFIG_PATH` | No | `config/settings.json` | Path to the JSON settings file. |

### Cache and performance settings

These can be set as environment variables (highest priority), in
`config/settings.json` (medium priority), or left at their built-in defaults
(lowest priority).

| Variable | JSON key | Default | Description |
| --- | --- | --- | --- |
| `CACHE_EXACT_ENABLED` | `cache.exact_enabled` | `true` | Enable L1 exact-match caching. |
| `CACHE_SEMANTIC_ENABLED` | `cache.semantic_enabled` | `true` | Enable L2 semantic caching. |
| `CACHE_TTL_SECONDS` | `cache.ttl_seconds` | `3600` | Cache entry lifetime (60–604800 seconds). |
| `MAX_SEMANTIC_DISTANCE` | `cache.max_semantic_distance` | `0.08` | Maximum cosine distance for a semantic hit (0.0–0.3). Lower = stricter. |
| `MAX_CONCURRENT_EMBEDS` | `performance.max_concurrent_embeds` | `2` | Concurrent embedding operations (≥ 1). |
| `MAX_STREAM_BUFFER_BYTES` | `performance.max_stream_buffer_bytes` | `1000000` | Max buffered SSE body for stream caching (≥ 1). |
| `GRAFANA_ADMIN_PASSWORD` | — | — | Grafana admin password (Compose only). |

Example `config/settings.json`:

```json
{
  "cache": {
    "exact_enabled": true,
    "semantic_enabled": true,
    "ttl_seconds": 3600,
    "max_semantic_distance": 0.08
  },
  "performance": {
    "max_concurrent_embeds": 2,
    "max_stream_buffer_bytes": 1000000
  }
}
```

Invalid settings stop the application at startup with a descriptive error.

## How caching works

```
Incoming request
 │
 ├─ Not cacheable? ──────────────────────────── BYPASS (forward directly)
 │   (temperature ≠ 0, tools, multi-modal,
 │    multi-turn, stateful, etc.)
 │
 ├─ L1 exact hash match? ──────────────────────── HIT-EXACT (return cached)
 │
 ├─ L2: embed prompt → KNN vector search
 │   │
 │   ├─ Cosine distance > threshold? ──────────── MISS
 │   │
 │   ├─ Prompt validation fails? ──────────────── MISS
 │   │   (numbers, negation, antonyms, entities)
 │   │
 │   └─ Passes all checks ─────────────────────── HIT-SEMANTIC (return cached)
 │
 └─ Forward to upstream
     └─ On 200 OK → store in L1 + L2 (background)
```

### What makes a request cacheable

| Condition | Responses API | Chat Completions API |
| --- | --- | --- |
| `temperature` must be `0` | ✓ | ✓ |
| No `tools` / `tool_choice` | ✓ | ✓ |
| Text-only content | ✓ | ✓ |
| No `previous_response_id` | ✓ | — |
| No `background` flag | ✓ | — |
| Single user message (L2 only) | ✓ | ✓ |
| Prompt ≤ 1000 chars (L2 only) | ✓ | ✓ |

### Parameter fingerprinting

The L2 semantic cache is scoped by a fingerprint that includes: model name,
temperature, top_p, max_tokens, stop sequences, tools, response_format,
system/developer messages, and all other non-volatile request parameters. Two
requests with different parameters **never** share a semantic cache entry, even
if their user prompts are identical.

## Tests

### Unit tests (offline, no running services required)

```sh
uv run pytest tests/test_config.py tests/test_semantic_validation.py -v
```

`test_semantic_validation.py` contains 58 adversarial tests covering:
- Constraint extraction (numbers, negation, entities)
- Prompt compatibility (identical, paraphrased, negated, opposite actions,
  different numbers, different entities)
- Full `search_semantic()` path with validation
- Distance thresholds, scope/fp isolation, Redis error handling
- Prompt storage and backward compatibility

### Integration tests (require a running proxy and Redis)

The integration tests use a custom harness that spawns a proxy subprocess and
mock upstream. Run the full suite:

```sh
python tests/run_all.py
```

Or filter by name:

```sh
python tests/run_all.py auth cache         # run tests matching "auth" or "cache"
python tests/run_all.py semantic           # matches both test_04 and test_semantic_validation
python tests/run_all.py --keep-keys        # leave Redis keys for manual inspection
```

Individual integration scripts:

| Script | Covers |
| --- | --- |
| `test_01_auth.py` | Authentication and key validation |
| `test_02_basic_request.py` | Request forwarding |
| `test_03_exact_cache.py` | L1 exact cache hit/miss cycle |
| `test_04_semantic_cache.py` | L2 semantic cache (paraphrases, fingerprint isolation) |
| `test_05_streaming.py` | SSE streaming pass-through |
| `test_06_cache_bypass.py` | Non-cacheable request bypass |
| `test_07_input_formats.py` | Input format variations |
| `test_08_edge_cases.py` | TTL, validation, tenant isolation |
| `test_09_redis_inspection.py` | Redis state verification |
| `test_10_performance.py` | Latency benchmarks |
| `test_semantic_validation.py` | Adversarial false-positive prevention (pytest) |

### Load testing

```sh
uv sync --extra loadtest
uv run locust -f locustfile.py --host http://localhost:8000
```

## Project structure

```
Janus/
├── src/janus/
│   ├── main.py          # FastAPI app, route handlers, request pipeline
│   ├── cache.py         # L1/L2 cache, constraint validation, Redis index
│   ├── embeddings.py    # Fingerprinting, text extraction, vector generation
│   ├── config.py        # Pydantic settings (env + JSON + defaults)
│   ├── models.py        # Request body Pydantic models
│   └── metrics.py       # Prometheus counters and histograms
├── tests/               # Integration + unit test suite
├── config/              # settings.json (user-created from example)
├── monitoring/          # Prometheus and Grafana provisioning
├── Dockerfile           # Production container image
├── compose.yaml         # Full stack (Janus + Redis + Prometheus + Grafana)
├── locustfile.py        # Locust load test
└── .env.example         # Template for required environment variables
```

## License

See [LICENSE](LICENSE) for details.
