from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.tasks.dm_task import LM4ManipBaseTask
from VLABench.utils.register import register

@register.add_task("simple_cuestick_use")
class SimpleCueStickUseTask(LM4ManipBaseTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)
