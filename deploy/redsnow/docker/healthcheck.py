#!/usr/bin/env python3
"""Docker HEALTHCHECK: rank 0 is healthy when its API lists the model (REDSNOW_HEALTH_URL, set by the shim on rank 0
only); a rank without that variable is healthy while it runs (docker already tracks that)."""
import os
import sys
import urllib.request

url = os.environ.get("REDSNOW_HEALTH_URL", "")
if not url:
    sys.exit(0)
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:  # noqa: BLE001
    sys.exit(1)
