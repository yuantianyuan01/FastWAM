from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    _PYDANTIC_V2 = True
except ImportError:  # pydantic v1
    ConfigDict = None  # type: ignore[misc, assignment]
    _PYDANTIC_V2 = False

from client_server.ws.protocol.exceptions import ErrorCode, WsError
from client_server.ws.protocol.messages import MessageType


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _frame_to_dict(frame: BaseModel, *, exclude: set[str]) -> dict[str, Any]:
    if hasattr(frame, "model_dump"):
        return frame.model_dump(mode="json", exclude=exclude)
    return frame.dict(exclude=exclude)  # pydantic v1


class Frame(BaseModel):
    """One WebSocket binary message."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(populate_by_name=True, extra="forbid")
    else:

        class Config:
            extra = "forbid"
            allow_population_by_field_name = True

    message_type: MessageType
    request_id: str
    evaluation_id: str
    action_case_id: str | None = None
    trial_id: str | None = None
    repeat_index: int | None = None
    step: int = 0
    sent_at: str = Field(default_factory=_utc_now_iso)
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_wire_dict(self) -> dict[str, Any]:
        data = _frame_to_dict(self, exclude={"payload", "request_id"})
        data["message_id"] = self.request_id
        data["payload"] = self.payload
        return data

    @classmethod
    def from_wire_dict(cls, data: dict[str, Any]) -> Frame:
        message_type_value = data.get("message_type")
        if message_type_value is None:
            raise WsError(ErrorCode.INVALID_FRAME, "missing message_type")

        try:
            message_type = MessageType(message_type_value)
        except ValueError as exc:
            raise WsError(
                ErrorCode.INVALID_FRAME, f"unknown message_type: {message_type_value!r}"
            ) from exc

        message_id = data.get("message_id")
        if message_id is None:
            raise WsError(ErrorCode.INVALID_FRAME, "missing message_id")

        evaluation_id = data.get("evaluation_id")
        if evaluation_id is None:
            raise WsError(ErrorCode.INVALID_FRAME, "missing evaluation_id")

        payload = data.get("payload") or {}
        if not isinstance(payload, Mapping):
            raise WsError(ErrorCode.INVALID_FRAME, "payload must be a map")

        return cls(
            message_type=message_type,
            request_id=str(message_id),
            evaluation_id=str(evaluation_id),
            action_case_id=data.get("action_case_id"),
            trial_id=data.get("trial_id"),
            repeat_index=data.get("repeat_index"),
            step=int(data.get("step", 0)),
            sent_at=str(data.get("sent_at") or _utc_now_iso()),
            payload=dict(payload),
        )
