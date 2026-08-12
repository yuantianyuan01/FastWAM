#!/bin/bash
# Wait until policy server process is listening on TCP port or the process exits.
# Prefer LISTEN-state checks to avoid websocket handshake errors; fall back to a
# plain TCP connect on minimal images without ss/lsof/netstat.
# Usage: wait_for_policy_server.sh <host> <port> <server_pid> [label] [timeout_sec]
set -euo pipefail

host=${1:?host required}
port=${2:?port required}
pid=${3:?server pid required}
label=${4:-Policy server}
timeout_sec=${5:-360}

RED=$'\033[31m'
GREEN=$'\033[32m'
BLUE=$'\033[34m'
YELLOW=$'\033[33m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

_draw_wait_progress() {
    local elapsed=$1
    local total=$2
    local wait_label=$3
    local wait_host=$4
    local wait_port=$5
    local width=28
    local filled=0
    local percent=0
    local remaining=0

    if (( total > 0 )); then
        percent=$(( elapsed * 100 / total ))
        filled=$(( elapsed * width / total ))
        remaining=$(( total - elapsed ))
    fi
    if (( filled > width )); then
        filled=${width}
    fi
    if (( remaining < 0 )); then
        remaining=0
    fi

    local bar=""
    local i
    for (( i = 0; i < width; i++ )); do
        if (( i < filled )); then
            bar+="#"
        else
            bar+="."
        fi
    done

    printf '\r%s[CONNECTING]%s %s waiting for %s:%s [%s] %3d%% %ss left' \
        "${BLUE}${BOLD}" "${RESET}" "${wait_label}" "${wait_host}" "${wait_port}" "${bar}" "${percent}" "${remaining}"
}

_port_is_listening() {
    local listen_host=$1
    local listen_port=$2

    # Minimal containers often lack ss/lsof/netstat. On Linux, /proc/net/tcp*
    # gives us the LISTEN state without opening a websocket connection.
    if command -v python3 >/dev/null 2>&1; then
        python3 - "${listen_host}" "${listen_port}" <<'PY'
import os
import socket
import sys

host = sys.argv[1]
port_hex = f"{int(sys.argv[2]):04X}"

_WILDCARDS = {"00000000", "00000000000000000000000000000000"}

# /proc/net/tcp stores IPv4 addresses as little-endian hex. Resolve the target
# host so a listener bound to a different interface does not false-positive.
host_hex = None
if host not in ("", "*", "0.0.0.0", "::"):
    try:
        host_hex = socket.inet_aton(socket.gethostbyname(host))[::-1].hex().upper()
    except OSError:
        host_hex = None  # unresolvable; fall back to port-only matching


def _addr_matches(local_addr_hex: str) -> bool:
    if host_hex is None or local_addr_hex in _WILDCARDS:
        return True
    # Exact IPv4 match, or an IPv4-mapped IPv6 entry ending with the same bytes.
    return local_addr_hex == host_hex or local_addr_hex.endswith(host_hex)


checked = False
for proc_path in ("/proc/net/tcp", "/proc/net/tcp6"):
    if not os.path.exists(proc_path):
        continue
    checked = True
    with open(proc_path, "r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.split()
            if len(parts) < 4 or parts[3] != "0A":
                continue
            local_addr, _, local_port = parts[1].rpartition(":")
            if local_port.upper() == port_hex and _addr_matches(local_addr.upper()):
                raise SystemExit(0)
raise SystemExit(1 if checked else 2)
PY
        proc_status=$?
        if (( proc_status == 0 )); then
            return 0
        fi
        if (( proc_status == 1 )); then
            return 1
        fi
    fi

    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :${listen_port}" 2>/dev/null | grep -q LISTEN
        return $?
    fi
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"${listen_port}" -sTCP:LISTEN >/dev/null 2>&1
        return $?
    fi
    if command -v netstat >/dev/null 2>&1; then
        netstat -an 2>/dev/null | grep -Eq "[\.:]${listen_port}[[:space:]].*LISTEN"
        return $?
    fi

    # Last-resort fallback for non-Linux/minimal systems. This may create a
    # short-lived TCP connection, but it keeps eval.sh usable when no LISTEN
    # inspection tool is available.
    if command -v python3 >/dev/null 2>&1; then
        python3 - "${listen_host}" "${listen_port}" >/dev/null 2>&1 <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1)
    sock.connect((host, port))
PY
        return $?
    fi

    return 1
}

echo -e "${BLUE}${BOLD}[CONNECTING]${RESET} ${label} -> ${host}:${port} (PID=${pid}, timeout=${timeout_sec}s)"

# Carriage-return progress bars flood redirected logs; only animate on a TTY.
stdout_is_tty=0
if [[ -t 1 ]]; then
    stdout_is_tty=1
fi

for elapsed in $(seq 0 "${timeout_sec}"); do
    if ! kill -0 "${pid}" 2>/dev/null; then
        if (( stdout_is_tty )); then
            printf '\n' >&2
        fi
        echo -e "${RED}${BOLD}[ERROR]${RESET} ${label} (PID=${pid}) exited before opening port ${port}." >&2
        exit 1
    fi
    if _port_is_listening "${host}" "${port}"; then
        if (( stdout_is_tty )); then
            printf '\n'
        fi
        echo -e "${GREEN}${BOLD}[CONNECTED]${RESET} ${label} ready on ${host}:${port} (PID=${pid})"
        exit 0
    fi
    if (( stdout_is_tty )); then
        _draw_wait_progress "${elapsed}" "${timeout_sec}" "${label}" "${host}" "${port}"
    elif (( elapsed > 0 && elapsed % 30 == 0 )); then
        echo -e "${BLUE}${BOLD}[CONNECTING]${RESET} ${label} still waiting for ${host}:${port} (${elapsed}/${timeout_sec}s)"
    fi
    if (( elapsed >= timeout_sec )); then
        break
    fi
    sleep 1
done

if (( stdout_is_tty )); then
    printf '\n' >&2
fi
echo -e "${RED}${BOLD}[ERROR]${RESET} ${label} timed out after ${timeout_sec}s waiting for port ${port}." >&2
exit 1
