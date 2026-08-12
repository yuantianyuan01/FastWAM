"""
Modified from: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/serving/websocket_policy_server.py
"""

import asyncio
import http
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass

import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from molmo_spaces.policy.base_policy import InferencePolicy, StatefulPolicy

logger = logging.getLogger(__name__)


@dataclass
class MutableFloat:
    value: float | None = None


@contextmanager
def measure_elapsed():
    mf = MutableFloat()
    start = time.perf_counter()
    try:
        yield mf
    finally:
        mf.value = time.perf_counter() - start


class WebsocketPolicyServer:
    """
    Serves a policy using the websocket protocol.

    Concurrent inference is supported for stateful policies via state saving.
    Non-stateful policies default to nonconcurrent inference unless force_concurrent is True.

    In order to provide for concurrent inference, we track policy state internally in the server.
    """

    def __init__(
        self,
        policies: InferencePolicy | list[InferencePolicy],
        model_name: str,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        max_concurrency: int = 100,
        force_concurrent: bool = False,
    ) -> None:
        """
        Args:
            policies: Multiple copies of the same policies to serve,
                requests will be balanced across the policies for concurrent inference.
                If a policy is passed instead of a list, it will be used as the only policy.
            model_name: The name of the model to serve. Will be included in the metadata.
            host: The host to serve the policy on.
            port: The port to serve the policy on.
            metadata: Additional metadata to serve with the policy.
            max_concurrency: The maximum number of concurrent clients to serve.
                Ignored for non-stateful policies unless force_concurrent is True.
            force_concurrent: Whether to force concurrent inference for non-stateful policies.
                This may cause bugs if the policy is not safe for concurrency.
        """
        policies = policies if isinstance(policies, list) else [policies]
        assert len(policies) > 0, "Must provide at least one policy"
        assert all(type(p) is type(policies[0]) for p in policies), (
            "All policies must be of the same type"
        )

        self._policies = policies
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._prepared = False
        self._client_states = {}  # map of client ID to policy state
        self._policy_idx_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=len(policies))
        for i in range(len(policies)):
            self._policy_idx_queue.put_nowait(i)

        self._executor = ThreadPoolExecutor(max_workers=len(policies))

        self._metadata["model_name"] = model_name

        if isinstance(self._policies[0], StatefulPolicy) or force_concurrent:
            self._server_semaphore = asyncio.Semaphore(max_concurrency)
        else:
            logger.info("Policy does not support state saving, disabling concurrency")
            self._server_semaphore = asyncio.Semaphore(1)

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """
        Prepares the policy and starts the server.
        """
        if not self._prepared:
            for future in as_completed(
                self._executor.submit(policy.prepare_model) for policy in self._policies
            ):
                future.result()
            self._prepared = True
        asyncio.run(self._run())

    async def _run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    @asynccontextmanager
    async def _acquire_policy(self):
        policy_idx = await self._policy_idx_queue.get()
        try:
            yield self._policies[policy_idx]
        finally:
            self._policy_idx_queue.put_nowait(policy_idx)

    def _inference(self, policy: InferencePolicy, obs: dict):
        with measure_elapsed() as total_time:
            with measure_elapsed() as preprocess_time:
                model_input = policy.obs_to_model_input(obs)
            with measure_elapsed() as infer_time:
                model_output = policy.inference_model(model_input)
            with measure_elapsed() as postprocess_time:
                action = policy.model_output_to_action(model_output)
        timing_dict = {
            "infer_ms": int(infer_time.value * 1000),
            "preprocess_ms": int(preprocess_time.value * 1000),
            "postprocess_ms": int(postprocess_time.value * 1000),
            "total_ms": int(total_time.value * 1000),
        }
        return action, timing_dict

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        client_id = websocket.id
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        logger.debug(f"Acquiring semaphore for client {client_id}")
        async with self._server_semaphore:
            logger.debug(f"Acquired semaphore for client {client_id}")
            try:
                while True:
                    start_time = time.monotonic()
                    obs = msgpack_numpy.unpackb(await websocket.recv())

                    policy_acquire_start = time.perf_counter()
                    async with self._acquire_policy() as policy:
                        policy_acquire_time = time.perf_counter() - policy_acquire_start
                        if isinstance(policy, StatefulPolicy):
                            if client_id in self._client_states:
                                policy.set_state(self._client_states[client_id])
                            else:
                                policy.reset()

                        loop = asyncio.get_event_loop()
                        action, action_timing = await loop.run_in_executor(
                            self._executor, self._inference, policy, obs
                        )

                        if isinstance(policy, StatefulPolicy):
                            self._client_states[client_id] = policy.get_state()

                    action_timing["policy_acquire_ms"] = int(policy_acquire_time * 1000)
                    action["server_timing"] = action_timing
                    if prev_total_time is not None:
                        # We can only record the last total time since we also want to include the send time.
                        action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                    await websocket.send(packer.pack(action))
                    prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
            except Exception:
                logger.exception("Error in policy server")
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
            finally:
                if client_id in self._client_states:
                    del self._client_states[client_id]


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
