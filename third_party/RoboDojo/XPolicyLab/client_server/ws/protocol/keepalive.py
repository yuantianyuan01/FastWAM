"""Shared validation for the websocket keepalive knobs.

Both the policy server and the eval client read `ws_ping_interval_s` /
`ws_ping_timeout_s` straight out of `deploy.yml`, so they need the same
answer for what a valid value is.
"""

from __future__ import annotations

import math
from typing import Any


def normalize_ws_ping(value: Any, *, field_name: str) -> float | None:
    """Return the interval in seconds, or None when keepalive is disabled.

    `null` is meaningful (it turns keepalive off), so it is accepted; a bool
    is not, because YAML's `true` would otherwise silently become 1 second.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be null or a positive number, got {value!r}"
        )
    seconds = float(value)
    # nan would silently disable the timeout comparison inside websockets;
    # inf would keep the connection open forever. Both are config mistakes.
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(
            f"{field_name} must be null or a positive finite number, got {value!r}"
        )
    return seconds
