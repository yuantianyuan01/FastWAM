#!/usr/bin/env bash
# Pass through the policy setup_eval_policy_server.sh launcher unchanged.
# WebSocket transport (protocol: ws in deploy.yml) is selected inside setup_policy_server.py.
set -euo pipefail

server_script="${1:?policy setup_eval_policy_server.sh required}"
cat "${server_script}"
