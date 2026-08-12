#!/bin/bash
set -e

host="$1"
port="$2"
timeout="${3:-30}"

python3 - "$host" "$port" "$timeout" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout = float(sys.argv[3])

start = time.time()
while time.time() - start < timeout:
    try:
        with socket.create_connection((host, port), timeout=1):
            print(f"[INFO] Port {port} is ready", file=sys.stderr)
            sys.exit(0)
    except OSError:
        time.sleep(0.2)

print(f"[ERROR] Timeout waiting for {host}:{port}", file=sys.stderr)
sys.exit(1)
PY