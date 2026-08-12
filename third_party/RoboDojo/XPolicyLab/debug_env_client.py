import argparse
import os
import cv2
from client_server.tcp.model_client import ModelClient
import numpy as np
from XPolicyLab.utils.process_data import get_robot_action_dim_info

Batch_Size = 10

class TestEnv:
    def __init__(self, deploy_cfg):
        self.success_num, self.episode_num = 0, 0
        self._stop_check = None
        self.deploy_cfg = deploy_cfg
        self.episode_step_limit = 20
        self.obs_encoded = deploy_cfg.get('obs_encoded', False)
        env_cfg_type = deploy_cfg['env_cfg_type']
        self.robot_action_dim_info = get_robot_action_dim_info(env_cfg_type)

        if deploy_cfg.get("protocol", "ws") == "ws":
            from client_server.ws import WsModelClient

            policy_server_url = deploy_cfg["policy_server_url"]
            if policy_server_url is None:
                policy_server_url = f"ws://{deploy_cfg['host']}:{deploy_cfg['port']}"
            self.model_client = WsModelClient(
                url=policy_server_url,
                evaluation_id=deploy_cfg["evaluation_id"],
                trial_id=deploy_cfg["trial_id"],
                action_case_id=deploy_cfg["action_case_id"],
                repeat_index=deploy_cfg["repeat_index"],
                ws_ping_interval_s=deploy_cfg.get("ws_ping_interval_s", 20.0),
                ws_ping_timeout_s=deploy_cfg.get("ws_ping_timeout_s", 20.0),
                # Omitted keys keep the client defaults; a slow-loading policy
                # server mainly needs max_connect_attempts / request_timeout_s.
                connect_timeout_s=deploy_cfg.get("connect_timeout_s"),
                handshake_timeout_s=deploy_cfg.get("handshake_timeout_s"),
                request_timeout_s=deploy_cfg.get("request_timeout_s"),
                max_connect_attempts=deploy_cfg.get("max_connect_attempts"),
                connect_retry_delay_s=deploy_cfg.get("connect_retry_delay_s"),
                max_connect_seconds=deploy_cfg.get("max_connect_seconds"),
                close_timeout_s=deploy_cfg.get("close_timeout_s"),
            )
        else:
            self.model_client = ModelClient(host=deploy_cfg['host'], port=deploy_cfg['port'])

    def set_stop_check(self, stop_check):
        self._stop_check = stop_check

    def get_obs(self, env_idx=0):
        # v1.0
        demo_obs = { # aloha
            "vision": {
                "cam_head": {
                    "color": np.zeros((480, 640, 3), dtype=np.uint8),
                    "depth": np.zeros((480, 640, 3), dtype=np.uint8),
                    "intrinsic_matrix": [
                        [615.0, 0.0, 320.0],
                        [0.0, 615.0, 240.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "extrinsics_matrix": [
                        [1.0, 0.0, 0.0, 0.10],
                        [0.0, 1.0, 0.0, 1.20],
                        [0.0, 0.0, 1.0, 1.50],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "shape": (480, 640),
                },
                "cam_left_wrist": {
                    "color": np.zeros((480, 640, 3), dtype=np.uint8),
                    "depth": np.zeros((480, 640, 3), dtype=np.uint8),
                    "intrinsic_matrix": [
                        [615.0, 0.0, 320.0],
                        [0.0, 615.0, 240.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "extrinsics_matrix": [
                        [1.0, 0.0, 0.0, 0.10],
                        [0.0, 1.0, 0.0, 1.20],
                        [0.0, 0.0, 1.0, 1.50],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "shape": (480, 640),
                },
                "cam_right_wrist": {
                    "color": np.zeros((480, 640, 3), dtype=np.uint8),
                    "depth": np.zeros((480, 640, 3), dtype=np.uint8),
                    "intrinsic_matrix": [
                        [615.0, 0.0, 320.0],
                        [0.0, 615.0, 240.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "extrinsics_matrix": [
                        [1.0, 0.0, 0.0, 0.10],
                        [0.0, 1.0, 0.0, 1.20],
                        [0.0, 0.0, 1.0, 1.50],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "shape": (480, 640),
                },
            },
            "instruction": "language instruction",
            "state": {
                "mobile": {
                    "base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # x,y,z + quat
                    "base_twist": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # vx,vy,vz, wx,wy,wz
                },
            },

            "additional_info": {
                "frequency": 30,  # Hz
            },

            "data_format_version": "v1.0",
            
            "env_idx": env_idx
        }

        state = demo_obs.setdefault("state", {})

        arm_dims = self.robot_action_dim_info["arm_dim"]
        ee_dims = self.robot_action_dim_info["ee_dim"]

        if len(arm_dims) == 1:
            prefixes = [""]
        elif len(arm_dims) == 2:
            prefixes = ["left_", "right_"]
        else:
            raise ValueError(f"Unsupported arm count: {len(arm_dims)}")

        for i, prefix in enumerate(prefixes):
            state[f"{prefix}arm_joint_state"] = np.zeros(arm_dims[i], dtype=np.float32)
            state[f"{prefix}ee_joint_state"] = np.zeros(ee_dims[i], dtype=np.float32)
            state[f"{prefix}ee_pose"] = np.ones(7, dtype=np.float32)
            state[f"{prefix}tcp_pose"] = np.zeros(7, dtype=np.float32)
            state[f"{prefix}delta_ee_pose"] = np.zeros(7, dtype=np.float32)

        if self.obs_encoded:
            self._encode_obs_colors(demo_obs)

        return demo_obs

    @staticmethod
    def _encode_obs_colors(obs):
        """Send encoded colors so the server-side decode path gets exercised."""
        vision = obs["vision"]
        buffer = cv2.imencode(".jpg", vision["cam_head"]["color"])[1].reshape(-1)

        vision["cam_head"]["color"] = buffer                 # 1-D uint8 buffer
        vision["cam_left_wrist"]["color"] = buffer.tobytes()  # raw bytes
        # cam_right_wrist stays a plain array, covering the passthrough branch.

    def get_obs_batch(self, env_idx_list):
        demo_obs_list = [self.get_obs(env_idx) for env_idx in env_idx_list] 
        return demo_obs_list

    def eval_one_episode(self):
        policy_name = self.deploy_cfg['policy_name']
        try:
            eval_module = __import__(f'XPolicyLab.policy.{policy_name}.deploy', fromlist=['eval_one_episode'])
        except ImportError as e:
            print("[TestEnv]", f"Failed to import policy module: XPolicyLab.policy.{policy_name}.deploy. Error: {e}", "ERROR")
            raise e
            
        if not hasattr(eval_module, 'eval_one_episode'):
            print("[TestEnv]", f"Module '.{policy_name}.deploy' does not have 'eval_one_episode' function", "ERROR")
            raise AttributeError(f"Missing eval_one_episode in policy module")
            
        eval_module.eval_one_episode(TASK_ENV=self, model_client=self.model_client)

    def eval_one_episode_batch(self):
        policy_name = self.deploy_cfg['policy_name']
        try:
            eval_module = __import__(f'XPolicyLab.policy.{policy_name}.deploy', fromlist=['eval_one_episode_batch'])
        except ImportError as e:
            print("[TestEnv]", f"Failed to import policy module: XPolicyLab.policy.{policy_name}.deploy. Error: {e}", "ERROR")
            raise e
            
        if not hasattr(eval_module, 'eval_one_episode_batch'):
            print("[TestEnv]", f"Module '.{policy_name}.deploy' does not have 'eval_one_episode_batch' function", "ERROR")
            raise AttributeError(f"Missing eval_one_episode_batch in policy module")
            
        eval_module.eval_one_episode_batch(TASK_ENV=self, model_client=self.model_client)

    def reset(self):
        self.model_client.call(func_name="reset")
        self.episode_step = 0

    def take_action(self, action):
        print(f"[TestEnv] Action Step: {self.episode_step} / {self.episode_step_limit} (step_limit)")
        self.episode_step += 1
        validate_robot_state_dict(action, self.robot_action_dim_info)

    def take_action_batch(self, action_list, env_idx_list):
        print(f"[TestEnv] Action Step: {self.episode_step} / {self.episode_step_limit} (step_limit)")
        self.episode_step += 1
        assert len(action_list) == len(env_idx_list), f"action num != env num: {len(action_list)} != {len(env_idx_list)}"
        for action in action_list:
            validate_robot_state_dict(action, self.robot_action_dim_info)

    def is_episode_end(self):
        if self._stop_check is not None and self._stop_check():
            print("[TestEnv] Check Episode End: stop requested")
            return True
        print("[TestEnv] Check Episode End:", self.episode_step >= self.episode_step_limit)
        return self.episode_step >= self.episode_step_limit
    
    def finish_episode(self):
        print("[TestEnv] Episode finished")

    def get_running_env_idx_list(self):
        # For demonstration, we assume all envs are running. Replace with actual logic if needed.
        return list(range(Batch_Size))


def validate_robot_state_dict(state_dict: dict, robot_action_dim_info: dict) -> None:
    """
    Validate whether the state_dict keys use the correct prefixes and dimensions.

    Args:
        state_dict: e.g. demo_obs["state"]
        robot_action_dim_info: {
            "arm_dim": [...],
            "ee_dim": [...],
        }

    Raises:
        KeyError: if required keys are missing
        ValueError: if unexpected keys or wrong dimensions are found
        TypeError: if values are not array-like
    """
    arm_dims = robot_action_dim_info["arm_dim"]
    ee_dims = robot_action_dim_info["ee_dim"]

    if len(arm_dims) != len(ee_dims):
        raise ValueError(
            f"robot_action_dim_info mismatch: len(arm_dim)={len(arm_dims)} "
            f"!= len(ee_dim)={len(ee_dims)}"
        )

    arm_count = len(arm_dims)

    if arm_count == 1:
        expected = {
            "arm_joint_state": arm_dims[0],
            "ee_joint_state": ee_dims[0],
            "ee_pose": 7,
            "tcp_pose": 7,
            "delta_ee_pose": 7,
        }
        forbidden_prefixes = ("left_", "right_")

    elif arm_count == 2:
        expected = {
            "left_arm_joint_state": arm_dims[0],
            "left_ee_joint_state": ee_dims[0],
            "left_ee_pose": 7,
            "left_tcp_pose": 7,
            "left_delta_ee_pose": 7,
            "right_arm_joint_state": arm_dims[1],
            "right_ee_joint_state": ee_dims[1],
            "right_ee_pose": 7,
            "right_tcp_pose": 7,
            "right_delta_ee_pose": 7,
        }
        forbidden_prefixes = ()
    else:
        raise ValueError(f"Unsupported arm count: {arm_count}")

    if forbidden_prefixes:
        bad_prefixed_keys = [
            k for k in state_dict.keys()
            if k.startswith(forbidden_prefixes)
        ]
        if bad_prefixed_keys:
            raise ValueError(
                f"Single-arm robot should not contain prefixed keys, "
                f"but got: {bad_prefixed_keys}"
            )
    unexpected_keys = [k for k in state_dict if k not in expected]

    if unexpected_keys:
        raise ValueError(f"Unexpected state keys: {unexpected_keys}")

    for key, expected_dim in expected.items():
        if not key in state_dict.keys():
            continue
        value = state_dict[key]
    
        if not isinstance(value, (np.ndarray, list, tuple)):
            raise TypeError(
                f"state_dict['{key}'] must be array-like, got {type(value)}"
            )

        arr = np.asarray(value)

        if arr.ndim != 1:
            raise ValueError(
                f"state_dict['{key}'] must be 1D, got shape {arr.shape}"
            )

        if arr.shape[0] != expected_dim:
            raise ValueError(
                f"state_dict['{key}'] dim mismatch: expected {expected_dim}, "
                f"got shape {arr.shape}"
            )

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_name", required=True, type=str)
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--env_cfg_type", type=str, required=True)
    parser.add_argument("--policy_name", type=str, required=True, help="XPolicyLab module name for deployment")
    parser.add_argument("--protocol", choices=("legacy_tcp", "ws"), default="ws")
    parser.add_argument("--host", type=str, default="localhost", help="server host")
    parser.add_argument("--port", type=int, required=True, help="server port")
    parser.add_argument("--policy_server_url", type=str)
    parser.add_argument("--evaluation_id", type=str, default="debug-eval")
    parser.add_argument("--action_case_id", type=str)
    parser.add_argument("--trial_id", type=str, default="debug-trial")
    parser.add_argument("--repeat_index", type=int)
    parser.add_argument("--eval_episode_num", type=int, default=10, help="number of evaluation episodes")
    parser.add_argument("--eval_batch", type=str2bool, default=False, help="whether to run batch evaluation")
    parser.add_argument("--obs_encoded", type=str2bool, default=os.environ.get("DEBUG_OBS_ENCODED", "0"),
                        help="send encoded camera colors to exercise the server-side decode path")

    args_cli = parser.parse_args()
    deploy_cfg = vars(args_cli)
    test_env = TestEnv(deploy_cfg)
    eval_batch = deploy_cfg['eval_batch']

    try:
        # Load XPolicyLab
        for idx in range(deploy_cfg["eval_episode_num"]):
            print(f"\033[94m🚀 Running Episode {idx}\033[0m")
            test_env.reset() # reset model, robot, and environment
            if not eval_batch:
                test_env.eval_one_episode()
            else:
                test_env.eval_one_episode_batch()
            test_env.finish_episode()
    finally:
        close = getattr(test_env.model_client, "close", None)
        if callable(close):
            close()
