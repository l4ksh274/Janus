"""Locust load test for the Janus semantic-caching proxy.

The workload is tuned to exercise the L2 *semantic* path heavily:

* Every fixed prompt is seeded into the cache once at startup, so paraphrases
  always have something to match against (no cold-start misses).
* Paraphrases are tight (same intent and constraints) and are mutated on the
  fly with light surface noise, so the pool of distinct variants is large and
  they don't all collapse into L1 exact hits after their first use.
* All requests share the same model / instructions / params, because the
  proxy only searches L2 among entries with the same fingerprint.

Usage
-----
    # Start the proxy (with Redis and the upstream running)
    uvicorn janus.main:app --port 8000

    # Headless:
    locust -f locustfile.py --headless -u 150 -r 10 -t 3m \
           --html report.html --host http://localhost:8000

    # Or the web UI:
    locust -f locustfile.py --host http://localhost:8000   # http://localhost:8089

Environment variables
~~~~~~~~~~~~~~~~~~~~~
    PROXY_HOST              Target host (default: http://localhost:8000; --host wins)
    LOCUST_API_KEY          Bearer token (default: first key in PROXY_API_KEYS)
    LOCUST_MODEL            Model string to embed in payloads
    LOCUST_API              responses | completions | both (default: both)
                            'both' picks an API at random per request and seeds both.
    LOCUST_NOVEL_RATIO      Fraction of novel/BYPASS requests   (default 0.10)
    LOCUST_DUPLICATE_RATIO  Fraction of exact-duplicate requests (default 0.10)
    LOCUST_STREAM_RATIO     Fraction of streaming duplicates     (default 0.05)
                            Semantic variants get the remainder  (default 0.75)
    LOCUST_MUTATE           1 = add light noise to paraphrases (default 1)
    LOCUST_STRICT           1 = mark a semantic request as a failure if it is
                            not a HIT-SEMANTIC / HIT-EXACT (default 0)
    LOCUST_RUN_NONCE        Fixed nonce. Set the same value on every worker in
                            distributed mode so they share cache entries.
    LOCUST_SEED_SETTLE      Seconds to wait after seeding (default 1.0)

Useful PromQL (Prometheus scraping /metrics):
    sum by (level) (increase(cache_hit_total[5m]))
    sum by (cache_result) (increase(requests_total{cache_result=~"MISS|BYPASS"}[5m]))
"""

from __future__ import annotations

import os
import random
import string
import uuid
from collections import Counter

import gevent
from dotenv import load_dotenv
from gevent.event import Event
from gevent.pool import Pool
from locust import HttpUser, between, events, task
from locust.runners import MasterRunner

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

API_KEY = os.getenv(
    "LOCUST_API_KEY",
    os.getenv("PROXY_API_KEYS", "test-key-1").split(",")[0].strip(),
)
MODEL = os.getenv("LOCUST_MODEL", "mistralai/mistral-large-4-1t-a52b")
API_MODE = os.getenv("LOCUST_API", "both").lower()
_ALL_ENDPOINTS = {
    "responses": "/v1/responses",
    "completions": "/v1/chat/completions",
}
if API_MODE == "both":
    ENDPOINTS = dict(_ALL_ENDPOINTS)
elif API_MODE in _ALL_ENDPOINTS:
    ENDPOINTS = {API_MODE: _ALL_ENDPOINTS[API_MODE]}
else:
    raise SystemExit(f"LOCUST_API must be responses, completions or both (got {API_MODE!r})")
API_NAMES = list(ENDPOINTS)

NOVEL_RATIO = float(os.getenv("LOCUST_NOVEL_RATIO", "0.10"))
DUPLICATE_RATIO = float(os.getenv("LOCUST_DUPLICATE_RATIO", "0.10"))
STREAM_RATIO = float(os.getenv("LOCUST_STREAM_RATIO", "0.05"))
SEMANTIC_RATIO = max(0.0, 1.0 - NOVEL_RATIO - DUPLICATE_RATIO - STREAM_RATIO)

MUTATE = os.getenv("LOCUST_MUTATE", "1") == "1"
STRICT = os.getenv("LOCUST_STRICT", "0") == "1"
SEED_SETTLE = float(os.getenv("LOCUST_SEED_SETTLE", "1.0"))
SEED_CONCURRENCY = 8
SEED_TIMEOUT = 180  # max seconds other users wait for seeding

# Must be identical across requests: instructions/system prompt feed the fingerprint.
RUN_NONCE = os.getenv("LOCUST_RUN_NONCE") or uuid.uuid4().hex[:8]
INSTRUCTIONS = f"Be terse. [locust-run {RUN_NONCE}]"

KINDS = ["semantic", "duplicate", "novel", "stream"]
WEIGHTS = [SEMANTIC_RATIO, DUPLICATE_RATIO, NOVEL_RATIO, STREAM_RATIO]

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

# seed prompt -> tight paraphrases (same intent AND same constraints, so the
# embedding distance stays under the proxy's max semantic distance).
SEMANTIC_VARIANTS: dict[str, list[str]] = {
    "What is the capital of France?": [
        "What's the capital of France?",
        "Which city is the capital of France?",
        "What is France's capital?",
        "What is the capital city of France?",
    ],
    "Explain quantum entanglement in one sentence.": [
        "Explain quantum entanglement in a single sentence.",
        "In one sentence, explain quantum entanglement.",
        "Explain quantum entanglement in one short sentence.",
    ],
    "Translate 'hello world' to French.": [
        'Translate "hello world" into French.',
        "How do you say 'hello world' in French?",
        "Translate 'hello world' into French.",
    ],
    "What is 2 + 2?": [
        "What is 2 plus 2?",
        "What's 2 + 2?",
        "What does 2 + 2 equal?",
        "What is 2+2?",
    ],
    "Who wrote 'Pride and Prejudice'?": [
        "Who wrote Pride and Prejudice?",
        "Who is the author of 'Pride and Prejudice'?",
        "Who authored 'Pride and Prejudice'?",
    ],
    "What is the speed of light in km/s?": [
        "What's the speed of light in km/s?",
        "What is the speed of light in kilometres per second?",
        "How fast is light in km/s?",
    ],
    "What is the chemical formula for water?": [
        "What's the chemical formula of water?",
        "What is water's chemical formula?",
        "What is the chemical formula of water?",
    ],
    "What year did the Berlin Wall fall?": [
        "In what year did the Berlin Wall fall?",
        "What year did the Berlin Wall come down?",
        "Which year did the Berlin Wall fall?",
    ],
    "What is the Pythagorean theorem?": [
        "What's the Pythagorean theorem?",
        "What is the Pythagorean theorem",
        "Can you tell me what the Pythagorean theorem is?",
    ],
    "What is the largest ocean on Earth?": [
        "Which is the largest ocean on Earth?",
        "What's the largest ocean on Earth?",
        "What is Earth's largest ocean?",
    ],
    "What is an LRU cache?": [
        "What's an LRU cache?",
        "What is a LRU cache?",
        "What is an LRU cache",
    ],
    "Explain REST in one sentence.": [
        "Explain REST in a single sentence.",
        "In one sentence, explain REST.",
        "Explain REST in one short sentence.",
    ],
    "List the planets in our solar system.": [
        "List the planets in the solar system.",
        "List all the planets in our solar system.",
        "Name the planets in our solar system.",
    ],
    "What programming language is FastAPI written in?": [
        "What language is FastAPI written in?",
        "Which programming language is FastAPI written in?",
        "What programming language is FastAPI built with?",
    ],
    "Name three sorting algorithms.": [
        "Name 3 sorting algorithms.",
        "Give three sorting algorithms.",
        "Name three algorithms for sorting.",
    ],
}

# Exact-duplicate pool: every semantic seed plus a few prompts without
# paraphrases. All of these are seeded at startup.
DUPLICATE_PROMPTS: list[str] = [
    *SEMANTIC_VARIANTS,
    "Summarise the theory of relativity in 50 words.",
    "Define 'semantic caching' in one paragraph.",
    "Write a haiku about caching.",
    "Explain the difference between TCP and UDP.",
    "Describe the OSI model in one sentence.",
]

_PREFIXES = ["", "", "", "Hi, ", "Quick question: ", "Hey - "]
_SUFFIXES = ["", "", "", " Thanks!", " Please.", " Thank you."]


def _mutate(prompt: str) -> str:
    """Add light surface noise so variants stay semantically close but unique.

    If hit rate drops, mutations are pushing distance past the proxy's
    threshold: set LOCUST_MUTATE=0 or trim the prefix/suffix lists.
    """
    suffix = random.choice(_SUFFIXES)
    if not suffix and random.random() < 0.3:
        prompt = prompt.rstrip("?.!")
    if random.random() < 0.2:
        prompt = prompt.lower()
    return random.choice(_PREFIXES) + prompt + suffix


def _semantic_prompt() -> str:
    seed = random.choice(list(SEMANTIC_VARIANTS))
    variant = random.choice(SEMANTIC_VARIANTS[seed])
    return _mutate(variant) if MUTATE else variant


def _random_sentence(words: int = 12) -> str:
    """Return a prompt that is intentionally unique for a cache-miss path."""
    return " ".join(
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
        for _ in range(words)
    )


def _make_payload(
    api: str,
    prompt: str,
    *,
    stream: bool = False,
    temperature: int | float = 0,
) -> dict:
    """Build a request payload for the given API.

    Keep everything except the prompt identical across requests so the
    fingerprint matches between seeds and paraphrases.
    """
    if api == "completions":
        return {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "stream": stream,
        }
    return {
        "model": MODEL,
        "input": prompt,
        "instructions": INSTRUCTIONS,
        "temperature": temperature,
        "stream": stream,
    }


def _build(api: str, kind: str) -> dict:
    if kind == "semantic":
        return _make_payload(api, _semantic_prompt())
    if kind == "duplicate":
        return _make_payload(api, random.choice(DUPLICATE_PROMPTS))
    if kind == "stream":
        return _make_payload(api, random.choice(DUPLICATE_PROMPTS), stream=True)
    # temperature != 0 makes it non-cacheable, guaranteeing a MISS/BYPASS.
    return _make_payload(api, _random_sentence(), temperature=1)


# --------------------------------------------------------------------------- #
# Shared state (per Locust process)
# --------------------------------------------------------------------------- #

_seeding_started = False
_seed_done = Event()
RESULTS: Counter = Counter()  # (api, kind, X-Cache value) -> count


class JanusUser(HttpUser):
    """Simulates a client of the Janus caching proxy.

    The first user to start seeds the cache with every fixed prompt; the rest
    wait until seeding finishes, then all users send a mix weighted heavily
    toward semantic paraphrases.
    """

    host = os.getenv("PROXY_HOST", "http://localhost:8000")
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        global _seeding_started
        self.client.headers.update(
            {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }
        )
        if not _seeding_started:
            # No yield between check and set, so only one greenlet seeds.
            _seeding_started = True
            try:
                self._seed_cache()
            finally:
                _seed_done.set()
        else:
            _seed_done.wait(timeout=SEED_TIMEOUT)

    def _seed_cache(self) -> None:
        """Populate L1/L2 so every paraphrase has a stored original to match."""

        def seed(job: tuple[str, str]) -> None:
            api, prompt = job
            with self.client.post(
                ENDPOINTS[api],
                json=_make_payload(api, prompt),
                catch_response=True,
                name=f"{ENDPOINTS[api]} [seed]",
            ) as resp:
                self._validate(resp, api, "seed")

        jobs = [(api, p) for api in API_NAMES for p in DUPLICATE_PROMPTS]
        Pool(SEED_CONCURRENCY).map(seed, jobs)
        gevent.sleep(SEED_SETTLE)  # let any async embedding/indexing finish

    @task
    def send_request(self) -> None:
        kind = random.choices(KINDS, weights=WEIGHTS)[0]
        api = random.choice(API_NAMES)
        with self.client.post(
            ENDPOINTS[api],
            json=_build(api, kind),
            catch_response=True,
            name=f"{ENDPOINTS[api]} [{kind}]",
        ) as resp:
            self._validate(resp, api, kind)

    @staticmethod
    def _validate(resp, api: str, kind: str) -> None:
        if resp.status_code >= 400:
            resp.failure(f"HTTP {resp.status_code}: {resp.text[:500]}")
            return
        x_cache = resp.headers.get("X-Cache", "?")
        RESULTS[(api, kind, x_cache)] += 1
        if STRICT and kind == "semantic" and x_cache not in ("HIT-SEMANTIC", "HIT-EXACT"):
            resp.failure(f"expected semantic hit, got X-Cache={x_cache}")
        else:
            resp.success()


@events.test_start.add_listener
def on_test_start(environment, **kwargs) -> None:
    """Log the run nonce and settings at the top of the Locust log."""
    if isinstance(environment.runner, MasterRunner):
        return
    print(
        f"[locust] run_nonce={RUN_NONCE} model={MODEL} api={API_MODE} "
        f"mix(semantic/dup/novel/stream)="
        f"{SEMANTIC_RATIO:.2f}/{DUPLICATE_RATIO:.2f}/{NOVEL_RATIO:.2f}/{STREAM_RATIO:.2f} "
        f"mutate={MUTATE} strict={STRICT}"
    )


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:
    """Print the X-Cache breakdown per request kind for this process."""
    if isinstance(environment.runner, MasterRunner) or not RESULTS:
        return
    print("[locust] X-Cache results by API and request kind:")
    for (api, kind, x_cache), n in sorted(RESULTS.items()):
        print(f"    {api:<12} {kind:<10} {x_cache:<13} {n}")