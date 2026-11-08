"""Launches the janus app for the test harness with DEBUG logging for the janus package.

(Run by harness.ProxyProcess, not by hand.)  The app module calls
logging.basicConfig(INFO) at import time; raising only the package logger to DEBUG
lets us also see lines like 'L2 nearest distance=0.0123 (max 0.0800)'.
"""

import importlib
import logging
import os

import uvicorn

module_name, _, attr = os.environ.get("APP_IMPORT", "janus.main:app").partition(":")
module = importlib.import_module(module_name)
logging.getLogger(os.environ.get("LOG_PACKAGE", "janus")).setLevel(logging.DEBUG)
uvicorn.run(
    getattr(module, attr or "app"),
    host="127.0.0.1",
    port=int(os.environ["TEST_PROXY_PORT"]),
    log_level="info",
)
