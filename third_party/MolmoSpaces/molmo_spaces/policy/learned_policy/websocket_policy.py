import logging
import time

import msgpack_numpy
import websockets.sync.client

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy

logger = logging.getLogger(__name__)


class WebsocketPolicy(InferencePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        config: MlSpacesExpConfig,
        model_name: str,
        host: str = "127.0.0.1",
        port: int | None = None,
        connection_timeout: float | None = None,
    ):
        super().__init__(config)
        self.model_name = model_name
        self._last_prompt: str | None = None

        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._ws = None
        self._server_metadata = None
        self._prepared = False
        self._connection_timeout = connection_timeout

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict]:
        start_time = time.monotonic()
        try:
            while True:
                try:
                    conn = websockets.sync.client.connect(
                        self._uri,
                        compression=None,
                        max_size=None,
                        open_timeout=600.0,
                        ping_interval=None,
                    )
                except ConnectionRefusedError as e:
                    if (
                        self._connection_timeout is not None
                        and time.monotonic() - start_time > self._connection_timeout
                    ):
                        raise TimeoutError(f"Timeout waiting for server at {self._uri}") from e
                    logger.info("Waiting for server...")
                    time.sleep(5)
                    continue
                metadata = msgpack_numpy.unpackb(conn.recv(timeout=10))
                return conn, metadata
        except OSError as e:
            raise RuntimeError(f"Error waiting for server at {self._uri}: {e}") from e

    def infer(self, obs: dict) -> dict:
        data = msgpack_numpy.packb(obs)
        self._ws.send(data)
        response = self._ws.recv(timeout=10)
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        self.close()
        self._prepared = False
        self.prepare_model()

    def prepare_model(self) -> None:
        if not self._prepared:
            self._ws, self._server_metadata = self._wait_for_server()
            self._prepared = True

    def obs_to_model_input(self, obs):
        if isinstance(obs, list):
            if len(obs) > 1:
                logger.warning(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    "WARNING: obs list has %d elements but only using the first one!\n"
                    "This may indicate a batching issue - expected single observation.\n"
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                    len(obs),
                )
            obs = obs[0]
        model_input = {**obs}
        prompt = self.task.get_task_description()

        if self._last_prompt is None:
            self._last_prompt = prompt
        model_input["task"] = prompt
        return model_input

    def inference_model(self, model_input):
        self.prepare_model()
        data = msgpack_numpy.packb(model_input)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def model_output_to_action(self, model_output):
        action = {
            "arm": model_output["arm"],
            "gripper": model_output["gripper"],
        }
        return action

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "websocket"
        info["policy_model_name"] = self.model_name
        info["prompt"] = self.task.get_task_description()
        return info

    def close(self):
        if self._ws is not None:
            logger.info("Closing websocket connection")
            self._ws.close()
            self._ws = None
