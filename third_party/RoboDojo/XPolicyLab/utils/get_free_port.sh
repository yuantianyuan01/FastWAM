#!/bin/bash
set -e

FREE_PORT=$(python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
)

echo -e "\033[33m[INFO] Using socket port: ${FREE_PORT}\033[0m" >&2
echo "${FREE_PORT}"