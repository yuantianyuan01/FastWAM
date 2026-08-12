import abc

from molmo_spaces.utils.profiler_utils import Profiler


class Planner:
    def __init__(self) -> None:
        # profiler
        self.profiler = Profiler()

    @abc.abstractmethod
    def motion_plan(self, *args, **kwargs) -> list:
        """Motion Planner that returns a trajectory i.e. list of waypoints"""
        raise NotImplementedError
