from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import PolicyFactory
from molmo_spaces.policy.dummy_policy import DummyPolicy
from molmo_spaces.utils.function_utils import make_lenient


class DummyPolicyConfig(BasePolicyConfig):
    """Policy config that uses DummyPolicy for testing."""

    policy_type: str = "dummy"
    policy_cls: type = None  # type: ignore
    policy_factory: PolicyFactory | None = None

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            self.policy_cls = DummyPolicy
            self.policy_factory = make_lenient(DummyPolicy)
