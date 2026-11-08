#!/usr/bin/env python3
"""Run the Janus test scripts in sequence and summarize their exit codes.

The wrapper is useful for a quick pass across the whole suite while preserving
per-script logs and environment overrides for Redis, the upstream URL, and the
model under test.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent

# Scripts run directly as subprocesses (custom harness).
SCRIPT_TESTS = [
    "test_01_auth.py",
    "test_02_basic_request.py",
    "test_03_exact_cache.py",
    "test_04_semantic_cache.py",
    "test_05_streaming.py",
    "test_06_cache_bypass.py",
    "test_07_input_formats.py",
    "test_08_edge_cases.py",
    "test_09_redis_inspection.py",
    "test_10_performance.py",
]

# Pytest modules run via ``python -m pytest`` (unit / offline tests).
PYTEST_TESTS = [
    "test_semantic_validation.py",
]

ALL_TESTS = SCRIPT_TESTS + PYTEST_TESTS


def main() -> None:
    """Run the selected Janus tests and print a combined pass/fail summary."""
    filters = []
    keep_keys = False
    for arg in sys.argv[1:]:
        if arg == "--keep-keys":
            keep_keys = True
        elif arg == "--help" or arg == "-h":
            print(__doc__)
            sys.exit(0)
        else:
            filters.append(arg)

    # Select tests
    if filters:
        scripts = [t for t in ALL_TESTS if any(f in t for f in filters)]
    else:
        scripts = ALL_TESTS

    if not scripts:
        print(f"No tests matched filters: {filters}")
        print(f"Available: {', '.join(ALL_TESTS)}")
        sys.exit(1)

    # Environment
    env = {
        **os.environ,
        "TEST_MODEL": os.environ.get("TEST_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
        "FORCE_COLOR": "1",
    }
    if keep_keys:
        env["KEEP_KEYS"] = "1"

    print(f"\033[1;36m{'=' * 78}")
    print(f"  Janus Test Suite — {len(scripts)} script(s)")
    print(f"  Model:    {env['TEST_MODEL']}")
    print(f"  Dev URL:  {env.get('DEV_URL', 'http://localhost:8000')}")
    print(f"  Redis:    {env.get('REDIS_URL', 'redis://localhost:6379')}")
    print(f"{'=' * 78}\033[0m\n")

    results: list[tuple[str, int, float]] = []  # (script, returncode, duration)

    for i, script in enumerate(scripts, 1):
        path = TESTS_DIR / script
        if not path.exists():
            print(f"\033[1;31m  SKIP: {script} (not found)\033[0m")
            results.append((script, -1, 0.0))
            continue

        print(f"\033[1;35m\n{'─' * 78}")
        print(f"  [{i}/{len(scripts)}] Running: {script}")
        print(f"{'─' * 78}\033[0m\n")

        t0 = time.perf_counter()
        if script in PYTEST_TESTS:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(path), "-v", "--tb=short"],
                env=env,
                cwd=str(PROJECT_ROOT),
            )
        else:
            proc = subprocess.run(
                [sys.executable, str(path)],
                env=env,
                cwd=str(PROJECT_ROOT),
            )
        duration = time.perf_counter() - t0
        results.append((script, proc.returncode, duration))

    # Combined summary
    print(f"\n\033[1;36m{'=' * 78}")
    print(f"  COMBINED RESULTS")
    print(f"{'=' * 78}\033[0m")

    passed = sum(1 for _, rc, _ in results if rc == 0)
    failed = [(s, rc, d) for s, rc, d in results if rc != 0]
    total_time = sum(d for _, _, d in results)

    for script, rc, dur in results:
        if rc == 0:
            print(f"  \033[1;32m✓ PASS\033[0m  {script:40s}  ({dur:.1f}s)")
        elif rc == -1:
            print(f"  \033[33m⊘ SKIP\033[0m  {script:40s}")
        else:
            print(f"  \033[1;31m✗ FAIL\033[0m  {script:40s}  ({dur:.1f}s, exit={rc})")

    print(f"\n  \033[1m{passed}/{len(results)} passed")
    if failed:
        print(f"  \033[1;31m{len(failed)} FAILED\033[0m")
    print(f"  Total time: {total_time:.1f}s\033[0m")
    print(f"\033[1;36m{'=' * 78}\033[0m")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
