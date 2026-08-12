"""LeRobot gRPC client for remote policy inference.

This client connects to a LeRobot async inference server and handles
the gRPC protocol for sending observations and receiving actions.
"""

import logging
import pickle
import time
from typing import Any

import grpc
import numpy as np

log = logging.getLogger(__name__)


class LeRobotGRPCClient:
    """gRPC client for LeRobot async inference server.

    This client handles the communication protocol with a LeRobot policy server,
    including connection setup, observation streaming, and action retrieval.
    """

    def __init__(self, host: str = "localhost", port: int = 8080):
        """Initialize the gRPC client.

        Args:
            host: Server hostname
            port: Server port
        """
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None
        self.timestep = 0
        self.connected = False

        # Will be set during connect() - imported from lerobot for pickle compatibility
        self._services_pb2 = None
        self._services_pb2_grpc = None
        self._RemotePolicyConfig = None
        self._TimedObservation = None

    def connect(
        self,
        pretrained_name_or_path: str,
        policy_type: str = "pi05",
        device: str = "cuda",
        actions_per_chunk: int = 50,
        lerobot_features: dict | None = None,
        rename_map: dict | None = None,
        max_retries: int = 5,
        retry_delay: float = 1.0,
    ) -> None:
        """Connect to the server and initialize the policy.

        Args:
            pretrained_name_or_path: HuggingFace model ID or local path
            policy_type: Type of policy (e.g., "pi05", "act", "smolvla")
            device: Device to run inference on ("cuda", "cpu")
            actions_per_chunk: Number of actions per inference call
            lerobot_features: Feature definitions for observations
            rename_map: Optional mapping to rename observation keys
            max_retries: Number of connection retry attempts
            retry_delay: Delay between retries in seconds
        """
        # Import from lerobot to ensure pickle compatibility with server
        try:
            from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
            from lerobot.transport import services_pb2, services_pb2_grpc
        except ImportError as e:
            log.error(
                "lerobot package with transport module is required. "
                "Install it with: pip install -e '.[async]' from lerobot repo"
            )
            raise e

        self._services_pb2 = services_pb2
        self._services_pb2_grpc = services_pb2_grpc
        self._RemotePolicyConfig = RemotePolicyConfig
        self._TimedObservation = TimedObservation

        # Create gRPC channel with large message limits for images
        self.channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ],
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        # Retry connection
        for attempt in range(max_retries):
            try:
                # Signal ready
                self.stub.Ready(services_pb2.Empty())

                # Build and send policy config using lerobot's class for pickle compatibility
                policy_config = self._RemotePolicyConfig(
                    policy_type=policy_type,
                    pretrained_name_or_path=pretrained_name_or_path,
                    lerobot_features=lerobot_features or {},
                    actions_per_chunk=actions_per_chunk,
                    device=device,
                    rename_map=rename_map or {},
                )

                config_bytes = pickle.dumps(policy_config)
                self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=config_bytes))

                self.connected = True
                log.info(f"Connected to LeRobot server at {self.host}:{self.port}")
                log.info(f"Loaded policy: {pretrained_name_or_path} ({policy_type})")
                return

            except grpc.RpcError as e:
                if attempt < max_retries - 1:
                    log.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(retry_delay)
                else:
                    log.error(f"Failed to connect after {max_retries} attempts")
                    raise

    def _chunk_observation(self, data: bytes, chunk_size: int = 1024 * 1024):
        """Generator to send observation in chunks with transfer state.

        Args:
            data: Serialized observation bytes
            chunk_size: Size of each chunk in bytes

        Yields:
            Observation protobuf messages
        """
        services_pb2 = self._services_pb2
        total_chunks = (len(data) + chunk_size - 1) // chunk_size

        for i in range(0, len(data), chunk_size):
            chunk_idx = i // chunk_size
            chunk_data = data[i : i + chunk_size]

            # Set transfer state based on position
            if total_chunks == 1:
                state = services_pb2.TRANSFER_END
            elif chunk_idx == 0:
                state = services_pb2.TRANSFER_BEGIN
            elif chunk_idx == total_chunks - 1:
                state = services_pb2.TRANSFER_END
            else:
                state = services_pb2.TRANSFER_MIDDLE

            yield services_pb2.Observation(transfer_state=state, data=chunk_data)

    def infer(self, observation: dict[str, Any]) -> list[np.ndarray]:
        """Send observation and receive action chunk.

        Args:
            observation: Dictionary with observation data matching LeRobot format:
                - "observation.state": np.ndarray of robot state
                - "observation.images.<camera>": np.ndarray images (H, W, C)
                - "task": optional text prompt

        Returns:
            List of action arrays (one per timestep in the chunk)
        """
        if not self.connected:
            raise RuntimeError("Not connected. Call connect() first.")

        services_pb2 = self._services_pb2

        # Wrap observation in TimedObservation using lerobot's class for pickle compatibility
        timed_obs = self._TimedObservation(
            timestamp=time.time(),
            timestep=self.timestep,
            observation=observation,
            must_go=True,  # Force processing
        )
        self.timestep += 1

        # Serialize and send
        obs_bytes = pickle.dumps(timed_obs)
        self.stub.SendObservations(self._chunk_observation(obs_bytes))

        # Get action chunk
        response = self.stub.GetActions(services_pb2.Empty())

        if response.data:
            timed_actions = pickle.loads(response.data)
            # Extract action tensors from TimedAction objects
            return [
                ta.action.numpy() if hasattr(ta.action, "numpy") else np.array(ta.action)
                for ta in timed_actions
            ]

        return []

    def reset(self) -> None:
        """Reset the client state (timestep counter)."""
        self.timestep = 0

    def close(self) -> None:
        """Close the gRPC channel."""
        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None
        self.connected = False
        log.info("Disconnected from LeRobot server")
