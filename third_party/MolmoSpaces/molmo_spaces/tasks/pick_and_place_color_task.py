import numpy as np

from molmo_spaces.configs.task_configs import PickAndPlaceColorTaskConfig
from molmo_spaces.tasks.pick_and_place_color_task_sampler import COLORS
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask


class PickAndPlaceColorTask(PickAndPlaceTask):
    """Pick and place color task implementation."""

    def reset(self, *args, **kwargs):
        """Reset the task and print the task description."""
        observation, info = super().reset(*args, **kwargs)
        return observation, info

    def rgba_to_color_name(self, target_rgba):
        target_rgba = np.array(target_rgba)
        for color_name, color_rgba in COLORS:
            if np.allclose(target_rgba, color_rgba, atol=0.01):
                return color_name
        return "colored"

    def get_task_description(self) -> str:
        """Get the task description with color specification."""
        task_config = self.config.task_config
        assert isinstance(task_config, PickAndPlaceColorTaskConfig)

        pickup_name = self.config.task_config.referral_expressions["pickup_name"]
        place_name = self.env.object_managers[self.env.current_batch_index].fallback_expression(
            task_config.place_receptacle_name
        )

        # Include the target receptacle color in the description
        target_receptacle_name = task_config.place_receptacle_name
        target_receptacle_color = self.rgba_to_color_name(
            task_config.object_colors[target_receptacle_name]
        ).replace("_", " ")

        print("TARGET RECEPTACLE COLOR:", target_receptacle_color)
        print(
            "DESCRIPTIONS:",
            f"Pick up the {pickup_name} and place it in or on the {target_receptacle_color} {place_name}",
        )
        return f"Pick up the {pickup_name} and place it in or on the {target_receptacle_color} {place_name}"
