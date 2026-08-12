"""Synchronous environment-side adapter for the policy websocket protocol."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig


class WsModelClient:
    """Synchronous env-side client for the websocket policy protocol.

    The bundled event loop runs on a dedicated background thread so
    websocket keepalive pings are answered even while the calling thread
    does long blocking work (sim stepping, video encoding) between policy
    calls — otherwise the server's ping timeout would drop the connection
    whenever the env blocks longer than ping_interval + ping_timeout.

    Not thread-safe: issue calls from a single thread at a time (concurrent
    calls would interleave requests and the reconnect logic on one client).

    The `step` sent on each frame counts INFERENCE calls (get_action /
    get_action_batch), not environment control steps — one inference call
    yields a whole action chunk. The server currently only echoes it back.
    """

    def __init__(
        self,
        *,
        url: str,
        evaluation_id: str,
        trial_id: str,
        action_case_id: str | None = None,
        repeat_index: int | None = None,
        ws_ping_interval_s: float | None = 20.0,
        ws_ping_timeout_s: float | None = 20.0,
        connect_timeout_s: float | None = None,
        handshake_timeout_s: float | None = None,
        request_timeout_s: float | None = None,
        max_connect_attempts: int | None = None,
        connect_retry_delay_s: float | None = None,
        max_connect_seconds: float | None = None,
        close_timeout_s: float | None = None,
        client: Any | None = None,
    ):
        self.action_case_id = action_case_id
        self.trial_id = trial_id
        self.repeat_index = repeat_index
        self._step = 0
        # None means "keep the PolicyEvalClientConfig default", so the defaults
        # stay defined in exactly one place.
        tuning = {
            "connect_timeout_s": connect_timeout_s,
            "handshake_timeout_s": handshake_timeout_s,
            "request_timeout_s": request_timeout_s,
            "max_connect_attempts": max_connect_attempts,
            "connect_retry_delay_s": connect_retry_delay_s,
            "max_connect_seconds": max_connect_seconds,
            "close_timeout_s": close_timeout_s,
        }
        self._client = client or PolicyEvalClient(
            PolicyEvalClientConfig(
                url=url,
                evaluation_id=evaluation_id,
                ws_ping_interval_s=ws_ping_interval_s,
                ws_ping_timeout_s=ws_ping_timeout_s,
                **{k: v for k, v in tuning.items() if v is not None},
            )
        )
        # Start the loop thread only after config/client construction: a
        # validation error above must not leak a running thread + open loop.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="ws-model-client", daemon=True
        )
        self._loop_thread.start()
        try:
            self._run(self._client.connect(handshake=True))
        except BaseException:
            # Don't leak the loop thread when the server is unreachable.
            self._shutdown_loop()
            raise

    def _run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run a client coroutine on the background loop and wait for it."""
        if self._loop.is_closed() or not self._loop_thread.is_alive():
            # Fail loudly instead of hanging forever on a future no loop
            # will ever complete (thread died or close() already ran).
            coro.close()
            raise RuntimeError("websocket client event loop is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _shutdown_loop(self) -> None:
        if self._loop.is_closed():
            return
        if self._loop_thread.is_alive():
            # Cancel and drain leftover tasks (e.g. an orphaned reconnect
            # after KeyboardInterrupt) so loop.close() below does not emit
            # "Task was destroyed but it is pending" warnings.
            async def _drain() -> None:
                tasks = [
                    t
                    for t in asyncio.all_tasks()
                    if t is not asyncio.current_task()
                ]
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            try:
                asyncio.run_coroutine_threadsafe(_drain(), self._loop).result(
                    timeout=5.0
                )
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5.0)
        if not self._loop_thread.is_alive():
            self._loop.close()

    def call(self, func_name: str | None = None, obs: Any = None, **kwargs: Any) -> Any:
        # Payload channel is only `obs` (legacy TCP compatibility). Extra kwargs
        # such as env_idx_list=... are not forwarded — reject them loudly.
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"WsModelClient.call() only accepts func_name and obs; "
                f"unsupported keyword argument(s): {unexpected}. "
                f"For get_action_batch, pass env indices as obs=env_idx_list."
            )

        if func_name == "prepare_case":
            if self.action_case_id is None:
                raise ValueError("prepare_case requires action_case_id")
            response = self._run(
                self._client.prepare_case(
                    self.action_case_id,
                    case_meta=self._dict_payload(func_name, obs),
                    repeat_index=self.repeat_index,
                )
            )
            return response.payload.get("result")

        if func_name == "reset":
            # model.reset() takes no arguments; reject a payload instead of
            # shipping one the server would have to drop.
            if obs is not None:
                raise TypeError("reset takes no obs payload")
            self._step = 0
            response = self._run(
                self._client.reset(
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    repeat_index=self.repeat_index,
                )
            )
            return response.payload.get("result")

        if func_name == "trial_end":
            response = self._run(
                self._client.trial_end(
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    result=self._dict_payload(func_name, obs),
                    repeat_index=self.repeat_index,
                )
            )
            return response.payload.get("result")

        if not isinstance(func_name, str) or not func_name or func_name.startswith("_"):
            raise NotImplementedError(f"unsupported websocket model call: {func_name!r}")

        # Generic model-method dispatch, matching the legacy TCP client which
        # forwarded any func_name. This covers update_obs/update_obs_batch
        # (obs / obs list), get_action (no obs), get_action_batch
        # (obs=env_idx_list), and custom public methods some policies expose
        # (e.g. Mem_0's begin_episode/step).
        response = self._run(
            self._client.call(
                func_name,
                obs,
                trial_id=self.trial_id,
                action_case_id=self.action_case_id,
                repeat_index=self.repeat_index,
                step=self._step,
            )
        )
        if func_name in {"get_action", "get_action_batch"}:
            self._step += 1
        return response.payload.get("result")

    @staticmethod
    def _dict_payload(func_name: str, obs: Any) -> dict | None:
        # Reject loudly instead of silently dropping a mistyped payload.
        if obs is not None and not isinstance(obs, dict):
            raise TypeError(
                f"{func_name} payload must be a dict or None, "
                f"got {type(obs).__name__}"
            )
        return obs

    def close(self) -> None:
        if self._loop.is_closed():
            return
        try:
            if self._loop_thread.is_alive():
                self._run(self._client.close())
        finally:
            self._shutdown_loop()

    def __enter__(self) -> WsModelClient:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
