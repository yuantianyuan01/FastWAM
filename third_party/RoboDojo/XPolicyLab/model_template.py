class ModelTemplate:
    """
    Base template for a policy/model class.
    策略/模型类的基础模板。

    This template defines the minimal interface required for:
    该模板定义了一个模型在以下场景中常用的最小接口：
    1. loading or building a model
       加载或构建模型
    2. updating observations from the environment
       接收并更新环境观测
    3. generating actions
       生成动作
    4. resetting internal states between episodes
       在每个 episode 之间重置内部状态

    Notes / 说明:
    - You should override the methods in your own subclass or implementation.
      你应该在自己的子类或具体实现中重写这些方法。
    - `self.model` can be a neural network, a planner, or any policy object.
      `self.model` 可以是神经网络、规划器，或者任意策略对象。
    """

    RED = "\033[31m"
    RESET = "\033[0m"

    def __init__(self):
        """
        Initialize the template model.
        初始化模板模型。

        Attributes / 属性:
            self.model:
                The actual model/policy object.
                实际的模型/策略对象。
        """
        self.model = None

    def _error_msg(self, msg: str) -> str:
        return f"{self.RED}[ERROR] {msg}{self.RESET}"

    def update_obs(self, obs):
        """
        Update the current observation used by the model.
        更新当前模型使用的观测。

        The policy server decodes camera colors before calling this, so
        obs["vision"][<camera>]["color"] is always an image array.
        策略服务端已完成图像解码，此处拿到的 color 一定是图像数组。
        """
        raise NotImplementedError(
            self._error_msg("update_obs() must be implemented by the user.")
        )

    def update_obs_batch(self, obs_list):
        """
        Update the current observation used by the model.
        更新当前模型使用的观测。

        Same decoding guarantee as update_obs().
        与 update_obs() 一样，图像已由服务端解码。
        """
        raise NotImplementedError(
            self._error_msg("update_obs_batch() must be implemented by the user.")
        )

    def get_action(self):
        """
        Predict or generate an action from the current observation/state.
        根据当前观测或内部状态预测/生成动作。
        """
        raise NotImplementedError(
            self._error_msg("get_action() must be implemented by the user.")
        )

    def get_action_batch(self, env_idx_list=None):
        """
        Predict or generate an action from the current observation/state.
        根据当前观测或内部状态预测/生成动作。
        """
        raise NotImplementedError(
            self._error_msg("get_action_batch() must be implemented by the user.")
        )

    def reset(self):
        """
        Reset the internal state of the model at the start of a new episode.
        在新 episode 开始时重置模型内部状态。
        """
        pass

    def prepare_case(self, case_meta=None):
        """
        Optional hook called before an action case starts.
        在 action case 开始前调用的 hook。
        """
        pass

    def on_trial_end(self, result=None):
        """
        Optional hook called after a trial finishes.
        在单次 trial 结束后调用的 hook。
        """
        pass
