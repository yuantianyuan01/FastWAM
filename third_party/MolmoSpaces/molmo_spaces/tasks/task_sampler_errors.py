class RetriableError(Exception):
    """Error after which we can continue sampling, e.g. by picking a new target in the current house"""

    pass


class ObjectPlacementError(RetriableError):
    """Error when placing some object in the current scene given the current constraints"""

    pass


class RobotPlacementError(RetriableError):
    """Error when placing robot in the current scene given the current constraints"""

    pass


class CameraPlacementError(RetriableError):
    """Error when placing a camera with visibility constraints in the current scene"""

    pass


class HouseInvalidForTask(Exception):
    """
    Exception raised when a house cannot be used for tasks due to physics constraints.

    This is typically raised when:
    - House has snap-back behavior and no stable positions can be found
    - House has unstable physics that prevents reliable object placement
    - House cannot be compiled or analyzed for stability
    """

    def __init__(self, house_info=None, reason=None, error=None) -> None:
        self.house_info = house_info
        self.reason = reason
        self.error = error

        if house_info and reason:
            message = f"House invalid for tasks: {reason}. Details: {house_info}"
        elif reason:
            message = f"House invalid for tasks: {reason}"
        else:
            message = "House invalid for tasks due to physics constraints"

        super().__init__(message)
