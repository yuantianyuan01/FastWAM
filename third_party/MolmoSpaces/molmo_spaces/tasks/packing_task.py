from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask


class PackingTask(PickAndPlaceTask):
    """Pick and place into a box (packing) task implementation."""

    def get_task_description(self) -> str:
        pickup_name = self.config.task_config.referral_expressions["pickup_name"]
        place_name = self.config.task_config.referral_expressions["place_name"]
        return f"Pick up the {pickup_name} and place it into the {place_name}"
