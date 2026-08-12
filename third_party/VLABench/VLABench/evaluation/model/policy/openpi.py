import websockets
import functools
import time
import collections
from typing import Dict, Tuple
import websockets.sync.client
from typing_extensions import override
import msgpack
import numpy as np
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler

def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)

class OpenPiPolicy:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, replan_steps: int = 4) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = Packer()
        self._ws, self._server_metadata = self._wait_for_server()
        self.action_plan = collections.deque(maxlen=replan_steps)
        self.replan_steps = replan_steps
        self.name = "openpi"
        self.control_mode = 'ee'
        self.timestep = 0

    def predict(self, observation, **kwargs):
        if self.timestep%self.replan_steps==0:    
            right, left, front, image_wrist = observation["rgb"]
            state = observation["ee_state"]
            pos, quat, gripper_state = state[:3], state[3:7], state[-1]
            ee_euler = quaternion_to_euler(quat)
            pos -= np.array([0, -0.4, 0.78])
            state= np.concatenate([pos, ee_euler, np.array(gripper_state).reshape(-1)])
            policy_input = {
                "observation/image": front,
                "observation/wrist_image":image_wrist,
                "observation/state": state,
                "prompt": observation["instruction"],
                }
            action_chunk = self.infer(policy_input)["actions"]
            self.action_plan.extend(action_chunk[: self.replan_steps])
        self.timestep += 1
        raw_action = self.action_plan.popleft()
        target_pos, target_euler, gripper = raw_action[:3], raw_action[3:6], raw_action[-1]
        if gripper >= 0.1:
            gripper_state = np.ones(2)*0.04
        else:
            gripper_state = np.zeros(2)
        target_pos = target_pos.copy()
        target_pos += np.array([0, -0.4, 0.78])
        return target_pos, target_euler, gripper_state


    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        print(f"Waiting for server at {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                print("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    @override
    def reset(self) -> None:
        self.timestep = 0
        self.action_plan = collections.deque(maxlen=self.replan_steps)