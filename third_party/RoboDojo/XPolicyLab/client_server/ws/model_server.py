"""WebSocket policy server."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import logging
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import yaml
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from client_server.ws.protocol.codec import decode_envelope, decode_frame, encode_frame
from client_server.ws.protocol.exceptions import ErrorCode, WsError
from client_server.ws.protocol.keepalive import normalize_ws_ping
from client_server.ws.protocol.messages import MessageType
from client_server.ws.protocol.schemas import Frame
from XPolicyLab.utils.process_data import decode_obs_images

logger = logging.getLogger(__name__)

# Bound on cached responses kept for duplicate-request replay. The client
# retries a request at most once and immediately, so the replay window is a
# handful of in-flight requests; keep the cap small because each entry holds
# a full response frame (potentially a whole action-chunk result).
_RESPONSE_CACHE_SIZE = 256

# Bound on concurrently processed frames per connection. The bundled client
# is synchronous (one outstanding request), so this only protects against
# misbehaving peers flooding frames faster than the model can answer.
_MAX_INFLIGHT_RESPONSES = 128

# How long shutdown paths wait for still-running model calls before giving
# up. A hung model must not be able to block stop() (or a disconnecting
# client's connection handler) forever.
_SHUTDOWN_DRAIN_TIMEOUT_S = 30.0


def _ok_payload(result: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": True}
    if result is not None:
        payload["result"] = result
    return payload


@dataclass
class PolicyServerConfig:
    host: str = "0.0.0.0"
    port: int = 19000
    ws_ping_interval_s: float | None = 20.0
    ws_ping_timeout_s: float | None = 20.0

    def __post_init__(self) -> None:
        self.ws_ping_interval_s = normalize_ws_ping(
            self.ws_ping_interval_s, field_name="ws_ping_interval_s"
        )
        self.ws_ping_timeout_s = normalize_ws_ping(
            self.ws_ping_timeout_s, field_name="ws_ping_timeout_s"
        )


@dataclass
class PolicyServer:
    model: Any
    config: PolicyServerConfig = field(default_factory=PolicyServerConfig)
    _server: Server | None = field(default=None, init=False)
    _model_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        init=False,
        repr=False,
    )
    # Identifies this server process in HELLO_ACK; the client pins it and
    # refuses to continue if a reconnect lands on a different process (a
    # restarted server lost the model state and this cache).
    _instance_id: str = field(default_factory=lambda: uuid4().hex, init=False)
    # request_id -> (request type, response frame), for exactly-once semantics
    # across client reconnect retries (the client reuses the request_id when
    # retrying). The request type is kept to detect id reuse across types.
    _response_cache: OrderedDict[str, tuple[MessageType, Frame]] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    # request_id -> (request type, running execution). A retry can arrive while
    # the original attempt is STILL executing (e.g. long inference on the old
    # connection); the duplicate awaits the same task instead of running twice.
    _inflight: dict[str, tuple[MessageType, asyncio.Task[Frame | None]]] = field(
        default_factory=dict, init=False, repr=False
    )

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await serve(
            self._handle_connection,
            self.config.host,
            self.config.port,
            max_size=None,
            # An observation frame is multi-MB of camera pixels (or already
            # JPEG-compressed buffers). Deflating that on every step costs
            # real CPU on the event loop thread for almost no size win, and
            # the stall delays the keepalive pongs.
            compression=None,
            ping_interval=self.config.ws_ping_interval_s,
            ping_timeout=self.config.ws_ping_timeout_s,
        )
        logger.info("websocket policy server listening on %s", self.url)

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        # Executions are shielded from their waiters' cancellation, so closing
        # the server can leave them running; drain them instead of letting the
        # loop tear down pending tasks. Bounded: a model call that hangs must
        # not turn stop() into a hang too.
        inflight = [task for _type, task in self._inflight.values()]
        if inflight:
            _done, still_running = await asyncio.wait(
                inflight, timeout=_SHUTDOWN_DRAIN_TIMEOUT_S
            )
            if still_running:
                logger.warning(
                    "%d model call(s) still running %ss after stop(); abandoning them",
                    len(still_running),
                    _SHUTDOWN_DRAIN_TIMEOUT_S,
                )

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None
        try:
            await self._server.serve_forever()
        finally:
            # Runs on cancellation too (e.g. Ctrl-C under asyncio.run), so
            # shielded model calls get their bounded drain instead of being
            # torn down mid-execution by loop shutdown.
            await self.stop()

    @property
    def url(self) -> str:
        port = self.config.port
        if self._server is not None and self._server.sockets:
            port = self._server.sockets[0].getsockname()[1]
        return f"ws://{self.config.host}:{port}"

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        send_lock = asyncio.Lock()
        pending: set[asyncio.Task[None]] = set()

        async def respond(frame: Frame) -> None:
            try:
                response = await self.process_frame(frame)
                if response is None:
                    await websocket.close()
                    return
                try:
                    data = encode_frame(response)
                except Exception as exc:
                    # A result the codec cannot serialize (e.g. a torch tensor
                    # returned by the model) must fail loudly: without a reply
                    # the client would stall for its full request timeout. The
                    # unencodable response was already cached, so overwrite it
                    # too, or retries would replay the same poisoned entry.
                    logger.exception("policy server response is not serializable")
                    response = self._error_reply(
                        frame,
                        ErrorCode.INTERNAL,
                        f"response is not msgpack-serializable: {exc}",
                    )
                    self._cache_response(
                        frame.request_id, frame.message_type, response
                    )
                    data = encode_frame(response)
                async with send_lock:
                    await websocket.send(data)
            except asyncio.CancelledError:
                raise
            except ConnectionClosed as exc:
                # Expected when the original attempt's connection died
                # mid-request; the response is cached for the client's retry.
                logger.warning("response undeliverable, client gone: %s", exc)
            except Exception:
                logger.exception("failed to send policy server response")

        try:
            async for raw in websocket:
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    frame = decode_envelope(bytes(raw))
                except WsError as exc:
                    logger.error("invalid request frame: %s", exc)
                    # Best-effort ERROR reply: without one the client stalls
                    # for its full request timeout instead of failing fast.
                    await self._reject_invalid_frame(
                        websocket, send_lock, bytes(raw), exc
                    )
                    continue
                if len(pending) >= _MAX_INFLIGHT_RESPONSES:
                    # Backpressure: process inline instead of queuing yet
                    # another task for an overwhelmed connection.
                    await respond(frame)
                    continue
                task = asyncio.create_task(respond(frame))
                pending.add(task)
                task.add_done_callback(pending.discard)
        except ConnectionClosed:
            # Routine for dropped/restarted clients; not a handler failure.
            logger.debug("policy client disconnected")
        finally:
            if pending:
                # Let responses finish so their results land in the replay
                # cache for the client's retry — but bounded, so a hung model
                # cannot pin this connection handler forever.
                _done, still_running = await asyncio.wait(
                    pending, timeout=_SHUTDOWN_DRAIN_TIMEOUT_S
                )
                for task in still_running:
                    # Cancelling a respond task only detaches the waiter; the
                    # shielded execution keeps running and its result is still
                    # cached for a retry. stop() drains (or abandons) it.
                    task.cancel()

    async def _reject_invalid_frame(
        self,
        websocket: ServerConnection,
        send_lock: asyncio.Lock,
        raw: bytes,
        exc: WsError,
    ) -> None:
        """Answer an undecodable request with an ERROR frame when possible.

        The envelope failed validation, but the raw msgpack map often still
        carries a usable message_id (e.g. unknown message_type, missing
        evaluation_id). Without a reply the client waits out its full
        request timeout; a truly unparseable blob is silently dropped.
        """
        try:
            wire = decode_frame(raw)
            message_id = wire.get("message_id")
            if message_id is None:
                return
            error = Frame(
                message_type=MessageType.ERROR,
                request_id=str(message_id),
                evaluation_id=str(wire.get("evaluation_id") or ""),
                payload={
                    "code": exc.code.value,
                    "message": exc.message,
                    "details": exc.details,
                },
            )
            data = encode_frame(error)
        except Exception:
            return
        try:
            async with send_lock:
                await websocket.send(data)
        except Exception:
            logger.debug("failed to send invalid-frame error reply", exc_info=True)

    def _cache_response(
        self, request_id: str, msg_type: MessageType, response: Frame
    ) -> None:
        cache = self._response_cache
        cache[request_id] = (msg_type, response)
        cache.move_to_end(request_id)
        while len(cache) > _RESPONSE_CACHE_SIZE:
            cache.popitem(last=False)

    def _reused_id_error(self, frame: Frame, original: MessageType) -> Frame:
        logger.error(
            "request_id %s reused for %s after %s; refusing to answer",
            frame.request_id,
            frame.message_type.value,
            original.value,
        )
        return self._error_reply(
            frame,
            ErrorCode.INVALID_FRAME,
            f"request_id already used for a {original.value} request; "
            f"request ids must be unique per request",
        )

    async def process_frame(self, frame: Frame) -> Frame | None:
        # Exactly-once replay: a client that lost the response to a request
        # (connection dropped mid-flight) retries with the SAME request_id;
        # answer from the cache instead of executing the call a second time.
        cached = self._response_cache.get(frame.request_id)
        if cached is not None:
            original_type, response = cached
            if original_type != frame.message_type:
                # Replaying a mismatched response would be worse than failing.
                return self._reused_id_error(frame, original_type)
            self._response_cache.move_to_end(frame.request_id)
            logger.info(
                "duplicate request %s (%s); replaying cached response",
                frame.request_id,
                frame.message_type.value,
            )
            return response
        return await self._execute_once(frame)

    async def _execute_once(self, frame: Frame) -> Frame | None:
        entry = self._inflight.get(frame.request_id)
        if entry is not None:
            original_type, task = entry
            if original_type != frame.message_type:
                return self._reused_id_error(frame, original_type)
            logger.info(
                "request %s (%s) already in flight; awaiting its result",
                frame.request_id,
                frame.message_type.value,
            )
        else:
            task = asyncio.create_task(self._execute_frame(frame))
            self._inflight[frame.request_id] = (frame.message_type, task)
            task.add_done_callback(
                lambda _t, rid=frame.request_id: self._inflight.pop(rid, None)
            )
        # Shield: a cancellation of THIS waiter (e.g. its connection died)
        # must not kill the shared execution for the other waiter.
        return await asyncio.shield(task)

    async def _execute_frame(self, frame: Frame) -> Frame | None:
        try:
            response = await self._dispatch_frame(frame)
        except WsError as exc:
            # The client only receives str(exc), so a model failure wrapped in
            # CALL_FAILED/RESET_FAILED/INFER_FAILED would otherwise leave no
            # traceback anywhere. Protocol rejections carry no cause and stay
            # a one-liner.
            logger.error(
                "policy server %s request failed: %s",
                frame.message_type.value,
                exc.message,
                exc_info=exc.__cause__ is not None,
            )
            response = self._error_reply(frame, exc.code, exc.message, exc.details)
        except Exception as exc:
            logger.exception("policy server request failed")
            response = self._error_reply(frame, ErrorCode.INTERNAL, str(exc))
        if response is not None:
            self._cache_response(frame.request_id, frame.message_type, response)
        return response

    def _reply(
        self,
        frame: Frame,
        message_type: MessageType,
        payload: dict[str, Any] | None = None,
    ) -> Frame:
        return Frame(
            message_type=message_type,
            request_id=frame.request_id,
            evaluation_id=frame.evaluation_id,
            action_case_id=frame.action_case_id,
            trial_id=frame.trial_id,
            repeat_index=frame.repeat_index,
            step=frame.step,
            payload=payload or {},
        )

    def _error_reply(
        self,
        frame: Frame,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> Frame:
        return self._reply(
            frame,
            MessageType.ERROR,
            {
                "code": code.value,
                "message": message,
                "details": details or {},
            },
        )

    async def _invoke_method(self, method, *args: Any) -> Any:
        if inspect.iscoroutinefunction(method):
            return await method(*args)
        result = await asyncio.to_thread(method, *args)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _call_model_method(self, method, *args: Any) -> Any:
        async with self._model_lock:
            return await self._invoke_method(method, *args)

    async def _dispatch_frame(self, frame: Frame) -> Frame | None:
        if frame.message_type == MessageType.HELLO:
            return self._reply(
                frame,
                MessageType.HELLO_ACK,
                {
                    "ok": True,
                    "server": "xpolicylab_policy_server",
                    "server_instance_id": self._instance_id,
                },
            )
        if frame.message_type == MessageType.PREPARE_CASE:
            return await self._handle_prepare_case(frame)
        if frame.message_type == MessageType.RESET:
            return await self._handle_reset(frame)
        if frame.message_type == MessageType.CALL:
            return await self._handle_call(frame)
        if frame.message_type == MessageType.INFER:
            return await self._handle_infer(frame)
        if frame.message_type == MessageType.TRIAL_END:
            return await self._handle_trial_end(frame)
        if frame.message_type == MessageType.HEARTBEAT:
            return self._reply(frame, MessageType.HEARTBEAT_ACK, {"ok": True})
        if frame.message_type == MessageType.CLOSE:
            return None
        raise WsError(
            ErrorCode.UNKNOWN_MESSAGE_TYPE,
            f"unsupported request type: {frame.message_type.value}",
        )

    async def _handle_prepare_case(self, frame: Frame) -> Frame:
        case_meta = dict(frame.payload)
        if frame.action_case_id is not None:
            case_meta.setdefault("action_case_id", frame.action_case_id)

        hook = getattr(self.model, "prepare_case", None)
        hook_result = (
            await self._call_model_method(hook, case_meta) if callable(hook) else None
        )
        return self._reply(
            frame, MessageType.PREPARE_CASE_ACK, _ok_payload(hook_result)
        )

    async def _handle_reset(self, frame: Frame) -> Frame:
        method = getattr(self.model, "reset", None)
        if not callable(method):
            raise WsError(ErrorCode.RESET_FAILED, "model.reset is not callable")
        # reset() takes no arguments (see ModelTemplate.reset). The legacy TCP
        # server appeared to support reset(obs) only because it dispatched every
        # command through one generic `method(obs) if obs is not None` line; no
        # caller ever sent one. A policy that needs a first observation should
        # reset() and then take a normal update_obs.
        try:
            result = await self._call_model_method(method)
        except Exception as exc:
            raise WsError(ErrorCode.RESET_FAILED, str(exc)) from exc
        return self._reply(frame, MessageType.RESET_RESULT, _ok_payload(result))

    async def _handle_call(self, frame: Frame) -> Frame:
        # Generic model-method dispatch, mirroring the legacy TCP server:
        # call method(obs) when obs is provided, method() otherwise.
        func_name = frame.payload.get("func_name")
        if not isinstance(func_name, str) or not func_name or func_name.startswith("_"):
            raise WsError(
                ErrorCode.INVALID_FRAME,
                f"call payload has invalid func_name: {func_name!r}",
            )
        method = getattr(self.model, func_name, None)
        if not callable(method):
            raise WsError(
                ErrorCode.INVALID_FRAME, f"no model method named {func_name!r}"
            )

        obs = frame.payload.get("obs")
        if func_name in {"update_obs", "update_obs_batch", "infer"} and obs is None:
            raise WsError(ErrorCode.INVALID_FRAME, f"{func_name} payload missing obs")

        if obs is not None:
            try:
                # Decode for EVERY payload-carrying call, not just the standard
                # method names: adapters are forbidden from decoding themselves,
                # and a policy may expose custom RPCs that ship camera frames
                # (Mem_0's begin_episode/step). decode_obs_images only touches
                # dicts with a `vision` field and returns already-decoded values
                # untouched, so payloads that are not observations — notably
                # get_action_batch's env_idx_list — pass straight through.
                #
                # JPEG decode is CPU-bound (a batched obs can be N envs x M
                # cameras); run it off the event loop for the same reason
                # compression is disabled — a blocked loop delays keepalive
                # pongs and every other connection.
                obs = await asyncio.to_thread(decode_obs_images, obs)
            except ValueError as exc:
                raise WsError(ErrorCode.INVALID_FRAME, str(exc)) from exc

        start = time.perf_counter()
        try:
            if obs is None:
                result = await self._call_model_method(method)
            else:
                result = await self._call_model_method(method, obs)
        except Exception as exc:
            raise WsError(ErrorCode.CALL_FAILED, str(exc)) from exc

        payload = _ok_payload(result)
        payload["latency_ms"] = (time.perf_counter() - start) * 1000.0
        return self._reply(frame, MessageType.CALL_RESULT, payload)

    async def _handle_infer(self, frame: Frame) -> Frame:
        observation = frame.payload.get("observation")
        if observation is None:
            raise WsError(ErrorCode.INVALID_FRAME, "infer payload missing observation")

        try:
            # Off the event loop: see _handle_call — JPEG decode is CPU-bound.
            observation = await asyncio.to_thread(decode_obs_images, observation)
        except ValueError as exc:
            raise WsError(ErrorCode.INVALID_FRAME, str(exc)) from exc

        start = time.perf_counter()
        try:
            async with self._model_lock:
                update_obs = getattr(self.model, "update_obs", None)
                get_action = getattr(self.model, "get_action", None)
                if callable(update_obs) and callable(get_action):
                    if not inspect.iscoroutinefunction(
                        update_obs
                    ) and not inspect.iscoroutinefunction(get_action):
                        # Keep the sync pair on ONE worker thread: some models
                        # carry thread-affine state between the two calls.
                        # NOTE: this only holds WITHIN one INFER request;
                        # across requests the default executor may pick a
                        # different thread, so threading.local model state is
                        # not supported.
                        def run_legacy_infer() -> Any:
                            update_result = update_obs(observation)
                            if inspect.isawaitable(update_result):
                                raise TypeError(
                                    "update_obs returned an awaitable but is not async"
                                )
                            result = get_action()
                            if inspect.isawaitable(result):
                                raise TypeError(
                                    "get_action returned an awaitable but is not async"
                                )
                            return result

                        result = await asyncio.to_thread(run_legacy_infer)
                    else:
                        # Mixed/async pair: _invoke_method keeps sync halves
                        # off the event loop (running a blocking sync call
                        # inline here would stall every connection).
                        await self._invoke_method(update_obs, observation)
                        result = await self._invoke_method(get_action)
                else:
                    infer = getattr(self.model, "infer", None)
                    if not callable(infer):
                        raise AttributeError(
                            "model must implement update_obs()/get_action() or infer(observation)"
                        )
                    result = await self._invoke_method(infer, observation)
        except Exception as exc:
            raise WsError(ErrorCode.INFER_FAILED, str(exc)) from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        if isinstance(result, Mapping) and "actions" in result:
            payload: dict[str, Any] = dict(result)
            payload.setdefault("latency_ms", latency_ms)
        else:
            payload = {"actions": result, "latency_ms": latency_ms}

        return self._reply(frame, MessageType.INFER_RESULT, payload)

    async def _handle_trial_end(self, frame: Frame) -> Frame:
        hook = getattr(self.model, "on_trial_end", None)
        hook_result = (
            await self._call_model_method(hook, dict(frame.payload))
            if callable(hook)
            else None
        )
        return self._reply(frame, MessageType.TRIAL_END_ACK, _ok_payload(hook_result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", "--config_path", dest="config_path", required=True
    )
    parser.add_argument("--protocol", choices=("ws",), default="ws")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--relay-url", dest="relay_url")
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a YAML mapping: {args.config_path}")
    deploy_cfg = data
    if args.host is not None:
        deploy_cfg["host"] = args.host
    if args.port is not None:
        deploy_cfg["port"] = args.port
    if args.relay_url is not None:
        deploy_cfg["relay_url"] = args.relay_url

    policy_name = deploy_cfg.get("policy_name")
    if not policy_name:
        raise ValueError("policy_name must be specified in config")
    module = importlib.import_module(f"XPolicyLab.policy.{policy_name}.model")
    model_class = getattr(module, "Model")
    model = model_class(deploy_cfg)

    # deploy.yml may carry an explicit `port: null`; treat it as "unset"
    # instead of crashing in int(None).
    port = deploy_cfg.get("port")
    server = PolicyServer(
        model,
        PolicyServerConfig(
            host=deploy_cfg.get("host") or "0.0.0.0",
            port=int(port) if port is not None else 19000,
            ws_ping_interval_s=deploy_cfg.get("ws_ping_interval_s", 20.0),
            ws_ping_timeout_s=deploy_cfg.get("ws_ping_timeout_s", 20.0),
        ),
    )
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        # serve_forever's finally already drained in-flight model calls.
        logger.info("policy server interrupted; shut down cleanly")


if __name__ == "__main__":
    main()
