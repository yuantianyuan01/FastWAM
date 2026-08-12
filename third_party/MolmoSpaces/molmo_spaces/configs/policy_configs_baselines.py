from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient


class PiPolicyConfig(BasePolicyConfig):
    checkpoint_path: str = "checkpoints/pi"
    # remote_config: None -> launch local server
    # or dict(host,port) -> attaches to remote server
    remote_config: dict | None = dict(host="localhost", port=8080)
    grasping_type: str = "binary"
    grasping_threshold: float = 0.5
    chunk_size: int = 8

    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.pi_policy import PI_Policy

            self.policy_cls = PI_Policy
            self.policy_factory = make_lenient(PI_Policy)


class DreamZeroPolicyConfig(BasePolicyConfig):
    checkpoint_path: str = "checkpoints/dreamzero"
    remote_config: dict = dict(host="localhost", port=0000)
    grasping_type: str = "binary"
    grasping_threshold: float = 0.5
    chunk_size: int = 24

    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.dreamzero_policy import DreamZero_Policy

            self.policy_cls = DreamZero_Policy
            self.policy_factory = make_lenient(DreamZero_Policy)


class CAPPolicyConfig(BasePolicyConfig):
    remote_config: dict = dict(host="localhost", port=8765)
    grasping_type: str = "binary"
    grasping_threshold: float = 0.7
    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "learned"
    use_vlm: bool = False  # required for non-pick tasks
    exo_vlm: bool = True  # not used if use_vlm is False

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.cap_policy import CAP_Policy

            self.policy_cls = CAP_Policy
            self.policy_factory = make_lenient(CAP_Policy)


class TeleopPolicyConfig(BasePolicyConfig):
    device: str = "keyboard"  # "spacemouse", "keyboard", "phone"
    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "teleop"
    # keyboard params
    step_size: float = 0.005
    rot_step: float = 0.02
    # spacemouse params
    pos_sensitivity: float = 0.005
    rot_sensitivity: float = 0.02
    product_id: int = 50741  # 50741=wireless, 50734=wired

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            if self.device == "keyboard":
                from molmo_spaces.policy.learned_policy.keyboard_policy import Keyboard_Policy

                self.policy_cls = Keyboard_Policy
                self.policy_factory = make_lenient(Keyboard_Policy)
            elif self.device == "spacemouse":
                from molmo_spaces.policy.learned_policy.spacemouse_policy import SpaceMouse_Policy

                self.policy_cls = SpaceMouse_Policy
                self.policy_factory = make_lenient(SpaceMouse_Policy)
            elif self.device == "phone":
                from molmo_spaces.policy.learned_policy.phone_policy import Phone_Policy

                self.policy_cls = Phone_Policy
                self.policy_factory = make_lenient(Phone_Policy)


class BimanualYamPiPolicyConfig(BasePolicyConfig):
    """Configuration for BimanualYamPiPolicy using LeRobot gRPC server."""

    name: str = "bimanual_yam_pi"
    checkpoint_path: str = "Jiafei1224/ppack200k"  # HuggingFace model ID
    remote_config: dict = dict(
        host="triton-cs-aus-454.reviz.ai2.in",
        port=8060,
        policy_type="pi05",
        device="cuda",
    )
    grasping_type: str = "binary"  # "binary" or "continuous"
    buffer_length: int = 50  # Number of actions per inference call

    # Camera mapping: MuJoCo camera name -> LeRobot observation key
    camera_mapping: dict = dict(
        left_wrist_camera="observation.images.left",
        right_wrist_camera="observation.images.right",
        exo_camera="observation.images.top",
    )

    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.bimanual_yam_pi_policy import (
                BimanualYamPiPolicy,
            )

            self.policy_cls = BimanualYamPiPolicy
            self.policy_factory = make_lenient(BimanualYamPiPolicy)
