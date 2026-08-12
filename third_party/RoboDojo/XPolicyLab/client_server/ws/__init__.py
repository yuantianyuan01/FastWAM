"""WebSocket env↔policy transport."""

from __future__ import annotations

from typing import Any

from client_server.ws.model_client import WsModelClient
from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
from client_server.ws.protocol.exceptions import (
    ErrorCode,
    ServerRestartedError,
    WsError,
)
from client_server.ws.protocol.messages import MessageType
from client_server.ws.protocol.schemas import Frame

__all__ = [
    "ErrorCode",
    "Frame",
    "MessageType",
    "PolicyEvalClient",
    "PolicyEvalClientConfig",
    "PolicyServer",
    "PolicyServerConfig",
    "ServerRestartedError",
    "WsError",
    "WsModelClient",
]


def __getattr__(name: str) -> Any:
    if name == "PolicyServer":
        from client_server.ws.model_server import PolicyServer

        return PolicyServer
    if name == "PolicyServerConfig":
        from client_server.ws.model_server import PolicyServerConfig

        return PolicyServerConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
