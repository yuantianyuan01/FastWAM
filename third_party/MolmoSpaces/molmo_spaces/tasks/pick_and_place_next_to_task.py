import logging
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.task_configs import PickAndPlaceNextToTaskConfig, PickAndPlaceTaskConfig
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom
from molmo_spaces.utils.pose import pos_quat_to_pose_mat

log = logging.getLogger(__name__)


def _get_surface_gap(config, attr: str, sampler_default: float) -> float:
    """Get surface gap from task_config, falling back to task_sampler_config."""
    task_config = config.task_config
    if isinstance(task_config, PickAndPlaceNextToTaskConfig):
        val = getattr(task_config, attr, None)
        if val is not None:
            return val
    sampler_config = getattr(config, "task_sampler_config", None)
    if sampler_config is not None:
        val = getattr(sampler_config, attr, None)
        if val is not None:
            log.warning(
                "%s not set in task_config, falling back to task_sampler_config (%s)",
                attr,
                val,
            )
            return val
    return sampler_default


class PickAndPlaceNextToTask(PickAndPlaceTask):
    """Pick and place next to task implementation."""

    def get_task_description(self) -> str:
        pickup_name = self.config.task_config.referral_expressions["pickup_name"]
        place_name = self.config.task_config.referral_expressions["place_name"]
        return f"Pick up the {pickup_name} and place it next to the {place_name}"

    def reset(self):
        """Reset the task and print goal."""
        observation, info = super().reset()

        pickup_name = self.config.task_config.referral_expressions["pickup_name"]
        place_name = self.config.task_config.referral_expressions["place_name"]
        print(f"Goal: place {pickup_name} next to {place_name}")

        return observation, info

    def judge_success(self) -> bool:
        """Judge if the task was successful (for data generation)."""
        return self.get_info()[0]["success"]

    def get_info(self) -> list[dict[str, Any]]:
        """Get additional metrics for each environment."""
        metrics = []

        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]
            task_config = self.config.task_config
            assert isinstance(task_config, PickAndPlaceTaskConfig)

            pickup_obj = create_mlspaces_body(data, task_config.pickup_obj_name)
            place_receptacle = create_mlspaces_body(data, task_config.place_receptacle_name)

            place_receptacle_aabb_center, place_receptacle_aabb_size = body_aabb(
                data.model, data, place_receptacle.body_id
            )
            pickup_obj_aabb_center, pickup_obj_aabb_size = body_aabb(
                data.model, data, pickup_obj.body_id
            )

            # Compute surface-to-surface distance (shortest distance between object surfaces in XY plane)
            receptacle_xy = place_receptacle_aabb_center[:2]
            pickup_obj_xy = pickup_obj_aabb_center[:2]
            center_to_center_xy = pickup_obj_xy - receptacle_xy

            # Get half-sizes in XY plane (for 2D distance calculation)
            receptacle_half_size_xy = place_receptacle_aabb_size[:2] / 2
            pickup_obj_half_size_xy = pickup_obj_aabb_size[:2] / 2

            # Compute surface-to-surface distance in XY plane
            separation_x = abs(center_to_center_xy[0])
            separation_y = abs(center_to_center_xy[1])

            surface_dist_x = max(
                0, separation_x - (receptacle_half_size_xy[0] + pickup_obj_half_size_xy[0])
            )
            surface_dist_y = max(
                0, separation_y - (receptacle_half_size_xy[1] + pickup_obj_half_size_xy[1])
            )

            # L2 norm gives shortest distance between the two boxes in XY plane
            xy_distance = np.sqrt(surface_dist_x**2 + surface_dist_y**2)

            min_surface_gap = _get_surface_gap(
                self.config,
                "min_surface_to_surface_gap",
                sampler_default=0.0,
            )
            max_surface_gap = _get_surface_gap(
                self.config,
                "max_surface_to_surface_gap",
                sampler_default=0.05,
            )

            # Success check for "next to" placement:
            # 1. Object is within surface-to-surface gap range (min_surface_gap to max_surface_gap) from receptacle
            # 2. Object is supported by the same surface as the receptacle (not falling)
            # 3. Receptacle hasn't moved too much
            within_distance_range = min_surface_gap <= xy_distance <= (max_surface_gap + 0.01)

            # Check that pickup object is supported by the same surface as the receptacle
            # This ensures the object is actually placed on a surface, not falling
            receptacle_supporting_geom = get_supporting_geom(data, place_receptacle.body_id)
            pickup_obj_supporting_geom = get_supporting_geom(data, pickup_obj.body_id)

            # Get the root body IDs of the supporting geoms
            if receptacle_supporting_geom is not None and pickup_obj_supporting_geom is not None:
                receptacle_supporting_body = data.model.body_rootid[
                    data.model.geom_bodyid[receptacle_supporting_geom]
                ]
                pickup_obj_supporting_body = data.model.body_rootid[
                    data.model.geom_bodyid[pickup_obj_supporting_geom]
                ]
                on_same_surface = receptacle_supporting_body == pickup_obj_supporting_body
            else:
                # Plato: If either object has no supporting geom, they're not on the same surface
                # Aristotle: Hold my beer...
                om = self.env.object_managers[i]
                b2g = om.get_body_to_geoms()
                p_sup_p = om.approximate_supporting_geoms(pickup_obj.body_id, b2g)
                r_sup_p = om.approximate_supporting_geoms(place_receptacle.body_id, b2g)
                p_sups = {p[1] for p in p_sup_p}
                r_sups = {p[1] for p in r_sup_p}
                on_same_surface = len(p_sups & r_sups) > 0  # False

            # Has the place receptacle moved too much?
            start_pose = pos_quat_to_pose_mat(
                task_config.place_receptacle_start_pose[0:3],
                task_config.place_receptacle_start_pose[3:7],
            )
            curr_pose = place_receptacle.pose
            displacement = np.linalg.inv(start_pose) @ curr_pose
            pos_displacement = displacement[:3, 3]
            rot_displacement = R.from_matrix(displacement[:3, :3]).magnitude()

            receptacle_pos_displacement_norm = np.linalg.norm(pos_displacement)
            success = (
                within_distance_range
                and on_same_surface
                and receptacle_pos_displacement_norm
                <= task_config.max_place_receptacle_pos_displacement
                and rot_displacement <= task_config.max_place_receptacle_rot_displacement
            )

            metrics.append(
                {
                    "success": success,
                    "within_distance_range": within_distance_range,
                    "on_same_surface": on_same_surface,
                    "xy_distance": xy_distance,  # Surface-to-surface xy distance
                    "min_surface_gap": min_surface_gap,
                    "max_surface_gap": max_surface_gap,
                    "receptacle_pos_displacement": pos_displacement,
                    "receptacle_pos_displacement_norm": receptacle_pos_displacement_norm,
                    "receptacle_rot_displacement": rot_displacement,
                    "episode_step": self.episode_step_count,
                }
            )

        return metrics
