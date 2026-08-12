import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.pick_and_place_task_sampler import (
    PickAndPlaceReceptacleTaskSampler,
)
from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms
from molmo_spaces.utils.pose import pose_mat_to_7d

if TYPE_CHECKING:
    from molmo_spaces.configs.base_pick_and_place_color_configs import (
        PickAndPlaceColorDataGenConfig,
    )

log = logging.getLogger(__name__)

COLORS = [
    ("red", np.array([1.0, 0.0, 0.0, 1.0])),
    ("blue", np.array([0.0, 0.0, 1.0, 1.0])),
    ("green", np.array([0.0, 1.0, 0.0, 1.0])),
    ("yellow", np.array([1.0, 1.0, 0.0, 1.0])),
    ("purple", np.array([0.5, 0.0, 0.5, 1.0])),
    ("orange", np.array([1.0, 0.5, 0.0, 1.0])),
    ("black", np.array([0.0, 0.0, 0.0, 1.0])),
    ("white", np.array([1.0, 1.0, 1.0, 1.0])),
    ("brown", np.array([0.55, 0.35, 0.15, 1.0])),
    ("tan", np.array([0.82, 0.71, 0.55, 1.0])),
    ("light_blue", np.array([0.68, 0.85, 0.90, 1.0])),
    ("light_green", np.array([0.56, 0.93, 0.56, 1.0])),
    ("light_yellow", np.array([1.0, 1.0, 0.88, 1.0])),
]


class PickAndPlaceColorTaskSampler(PickAndPlaceReceptacleTaskSampler):
    def __init__(self, config: "PickAndPlaceColorDataGenConfig") -> None:
        super().__init__(config)
        self.config: PickAndPlaceColorDataGenConfig

        self._color_assignments: dict[str, tuple[str, np.ndarray]] = {}
        self._receptacle_multiplier = 2
        assert len(COLORS) >= self._receptacle_multiplier, (
            f"Only {len(COLORS)} receptacles allowed (using {self._receptacle_multiplier})"
        )

    def _on_candidate_selected(
        self,
        env: CPUMujocoEnv,
        reference_obj_name: str,
        reference_obj_id: int,
        supporting_geom_id: int,
    ) -> bool:
        if not super()._on_candidate_selected(
            env, reference_obj_name, reference_obj_id, supporting_geom_id
        ):
            return False

        # Write distractor receptacle names to the config early so that
        # get_task_relevant_objects can find them during camera setup
        # (before _sample_task populates these).
        self.config.task_config.other_receptacle_names = [
            n for n in self.active_receptacle_names if n != self.place_receptacle_name
        ]
        return True

    @staticmethod
    def receptacle_material_callback(object_spec: mujoco.MjsBody) -> None:
        for geom in object_spec.geoms:
            if "_visual" in geom.name:
                geom.material = ""

    def _assign_colors_to_receptacles(self, env: CPUMujocoEnv) -> tuple[str, np.ndarray]:
        """Assign unique colors to all receptacles.

        Returns the target color (name, rgba) assigned to the first receptacle.
        """

        # TODO this is incompatible with multithreading
        assert env.current_batch_index == 0, (
            "Adding coloring is currently only supported for single threaded runs"
        )

        model = env.current_model
        om = env.object_managers[env.current_batch_index]

        # Select target color
        target_color_idx = np.random.randint(0, len(COLORS))
        target_color_name, target_color_rgba = COLORS[target_color_idx]

        # Prepare other colors
        available_colors = [c for c in COLORS if c[0] != target_color_name]
        np.random.shuffle(available_colors)

        self._color_assignments = {}
        self.config.task_config.object_colors = {}
        self.config.task_config.other_receptacle_names = []
        self.config.task_config.other_receptacle_start_poses = {}

        # Target receptacle is the one selected by parent's _sample_task
        target_receptacle_name = self.place_receptacle_name

        assert target_receptacle_name == self.active_receptacle_names[0], (
            f"target recep {target_receptacle_name} should be first in active receptacles {self.active_receptacle_names}"
        )
        assert len(self.active_receptacle_names) == self._receptacle_multiplier, (
            f"active receptacles {len(self.active_receptacle_names)} != {self._receptacle_multiplier} (multiplier)"
        )

        distractor_color_idx = 0
        for receptacle_name in self.active_receptacle_names:
            if receptacle_name == target_receptacle_name:
                # Target receptacle gets target color
                color_name, color_rgba = target_color_name, target_color_rgba
            else:
                # Distractor receptacles get other colors
                color_name, color_rgba = available_colors[
                    distractor_color_idx % len(available_colors)
                ]
                distractor_color_idx += 1

                self.config.task_config.other_receptacle_names.append(receptacle_name)
                self.config.task_config.other_receptacle_start_poses[receptacle_name] = (
                    pose_mat_to_7d(om.get_object_by_name(receptacle_name).pose).tolist()
                )

            # Color the receptacle geoms
            receptacle_id = om.get_object_body_id(receptacle_name)
            receptacle_geoms = descendant_geoms(model, receptacle_id, visible_only=True)
            for geom_id in receptacle_geoms:
                model.geom_rgba[geom_id] = color_rgba

            self._color_assignments[receptacle_name] = (color_name, color_rgba)
            self.config.task_config.object_colors[receptacle_name] = color_rgba.tolist()
            log.info(f"Colored receptacle {receptacle_name} -> {color_name}")

        return target_color_name, target_color_rgba

    def _sample_task(self, env: CPUMujocoEnv):
        from molmo_spaces.tasks.pick_and_place_color_task import PickAndPlaceColorTask

        self._configure_pick_and_place(env)
        self._assign_colors_to_receptacles(env)
        return PickAndPlaceColorTask(env, self.config)
