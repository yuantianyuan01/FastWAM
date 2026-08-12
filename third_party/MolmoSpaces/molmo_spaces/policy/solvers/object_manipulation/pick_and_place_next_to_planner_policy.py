import mujoco
import numpy as np

from molmo_spaces.env.data_views import MlSpacesObject, create_mlspaces_body
from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
    PickAndPlacePlannerPolicy,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb, body_base_pos
from molmo_spaces.utils.mujoco_scene_utils import (
    get_supporting_geom,
    place_object_near,
)


class PickAndPlaceNextToPlannerPolicy(PickAndPlacePlannerPolicy):
    """Planner policy for pick and place next to task."""

    # def _get_grasp_poses(
    #     self,
    #     grasp_pose_world: np.ndarray,
    #     pickup_obj: MlSpacesObject,
    #     place_receptacle: MlSpacesObject,
    #     robot_view,
    #     task_config,
    # ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     pregrasp_pose, grasp_pose_world, lift_pose = super()._get_grasp_poses(
    #         grasp_pose_world, pickup_obj, place_receptacle, robot_view, task_config
    #     )
    #
    #     # Cheap alternative to looking for the highest obstacle on the way
    #
    #     # Lift pose - above grasp position
    #     place_receptacle_aabb_center, place_receptacle_aabb_size = body_aabb(
    #         self.task.env.current_data.model, self.task.env.current_data, place_receptacle.object_id
    #     )
    #     receptacle_top_z = place_receptacle_aabb_center[2] + place_receptacle_aabb_size[2] / 2
    #
    #     pickup_obj_aabb_center, pickup_obj_aabb_size = body_aabb(
    #         self.task.env.current_data.model, self.task.env.current_data, pickup_obj.object_id
    #     )
    #     pickup_object_top_z = pickup_obj_aabb_center[2] + pickup_obj_aabb_size[2] / 2
    #
    #     delta = max(pickup_object_top_z - receptacle_top_z, 0.0)
    #
    #     if delta > 0:
    #         new_lift_pose = lift_pose.copy()
    #         new_lift_pose[2, 3] += delta
    #
    #         if self.check_feasible_ik(new_lift_pose):
    #             print(
    #                 f"Replacing lift height by {delta}: from {lift_pose[2, 3]} to {new_lift_pose[2, 3]}"
    #             )
    #             lift_pose[2, 3] += delta
    #
    #     return pregrasp_pose, grasp_pose_world, lift_pose

    def _get_placement_poses(
        self,
        grasp_pose_world: np.ndarray,
        pickup_obj: MlSpacesObject,
        place_receptacle: MlSpacesObject,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        data = om.data

        pickup_obj_body = create_mlspaces_body(data, pickup_obj.object_id)
        body_base = body_base_pos(data, pickup_obj_body.body_id, visible_only=False)

        # Save original pose of pickup object (place_object_near will modify it)
        original_pose = pickup_obj_body.pose.copy()

        task_config = self.config.task_config
        min_surface_gap = task_config.min_surface_to_surface_gap
        max_surface_gap = task_config.max_surface_to_surface_gap

        receptacle_aabb_center, receptacle_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, place_receptacle.object_id
        )
        _, pickup_obj_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, pickup_obj.object_id
        )

        # Get half-sizes in XY plane (for 2D distance calculation)
        receptacle_half_size_xy = receptacle_aabb_size[:2] / 2
        pickup_obj_half_size_xy = pickup_obj_aabb_size[:2] / 2

        # Convert surface-to-surface gap to center-to-center distance
        # For axis-aligned boxes, use average half-size as approximation for random directions
        avg_receptacle_half_size = np.mean(receptacle_half_size_xy)
        avg_pickup_obj_half_size = np.mean(pickup_obj_half_size_xy)
        total_half_size = avg_receptacle_half_size + avg_pickup_obj_half_size
        min_center_to_center = total_half_size + min_surface_gap
        max_center_to_center = total_half_size + max_surface_gap / 2  # getting closer to the target

        # Get supporting geometry for the receptacle (to place on same surface)
        supporting_geom_id = get_supporting_geom(data, place_receptacle.object_id)

        if supporting_geom_id is None:
            print("Trying to get supporting geom id via bounding box approximation")
            body_to_geoms = om.get_body_to_geoms()
            pickup_geoms = om.approximate_supporting_geoms(pickup_obj, body_to_geoms)
            pickup_geoms = set(p[1] for p in pickup_geoms)
            recep_geoms = om.approximate_supporting_geoms(place_receptacle, body_to_geoms)
            # Since the pickup object could have ended up on top of something else, we
            # prioritize the receptacle (we need to bring out pickup object close to it)
            for _, gid, _ in recep_geoms:
                if gid in pickup_geoms:
                    # if some possible supporting geom happens to be also under the pickup object, use this
                    supporting_geom_id = gid
                    break
            else:
                if recep_geoms:
                    # the pickup object might be far from the receptacle. Use the nearest geom to
                    # the receptacle bottom
                    supporting_geom_id = recep_geoms[0][1]
                else:
                    # Disaster. We couldn't find a supporting geom id for the receptacle. falling back to
                    # using whatever the pickup object has
                    supporting_geom_id = pickup_geoms[0][1]

        if supporting_geom_id is None:
            raise ValueError(
                f"Failed to find supporting geom id for pickup {pickup_obj.name} and receptacle {place_receptacle.name}"
            )

        # Use place_object_near() to find a collision-free placement position
        # This samples positions within min_dist to max_dist from the receptacle
        placement_pos = None
        placed_obj_bottom_z = None
        try:
            place_object_near(
                data=data,
                object_id=pickup_obj.object_id,
                placement_point=receptacle_aabb_center,  # Use AABB center for consistency
                min_dist=min_center_to_center,
                max_dist=max_center_to_center,
                max_tries=self.config.task_sampler_config.max_place_receptacle_sampling_attempts,
                supporting_geom_id=supporting_geom_id,
                reference_pos=self.task.env.current_robot.robot_view.base.pose[:3, 3],
                z_eps=0.003,
            )
            # Get the placed object's position and AABB
            placement_pos = pickup_obj.position.copy()
            # Recompute AABB after placement to get correct bottom Z
            placed_obj_aabb_center, placed_obj_aabb_size = body_aabb(
                self.task.env.current_data.model, self.task.env.current_data, pickup_obj.object_id
            )
            placed_obj_bottom_z = placed_obj_aabb_center[2] - placed_obj_aabb_size[2] / 2
        finally:
            # Always restore the original pose (place_object_near modifies it)
            pickup_obj_body.pose = original_pose
            mujoco.mj_forward(data.model, data)

        # Use the placement position from place_object_near
        # placement_z is where the object's bottom will be (same as placed_obj_bottom_z)
        placement_xy = placement_pos[:2]
        placement_z = placed_obj_bottom_z

        # Approximate the min clearance for our flight by the sum of pickup and receptacle
        # (think we might move over the receptacle)

        pickup_object_top_z = placed_obj_aabb_center[2] + placed_obj_aabb_size[2] / 2
        receptacle_top_z = receptacle_aabb_center[2] + receptacle_aabb_size[2] / 2
        pickup_obj_clearance_offset = max(grasp_pose_world[2, 3] - body_base[2], 0.0)

        preplace_pose = grasp_pose_world.copy()
        preplace_pose[:2, 3] = placement_xy  # Next to receptacle, not at center
        # Let's first try with some more margin (e.g. if the pickup objects is taller)
        preplace_pose[2, 3] = (
            max(pickup_object_top_z, receptacle_top_z)
            + pickup_obj_clearance_offset
            + self.policy_config.place_z_offset
        )
        if not self.check_feasible_ik(preplace_pose):
            # Fallback, assume the height of the receptacle to be the limitng factor
            if receptacle_top_z < pickup_object_top_z:
                preplace_pose[2, 3] = (
                    receptacle_top_z
                    + pickup_obj_clearance_offset
                    + self.policy_config.place_z_offset
                )

            if not self.check_feasible_ik(preplace_pose):
                raise ValueError("IK failed in PickAndPlaceNextToPlannerPolicy for preplace pose")

        place_pose = preplace_pose.copy()
        place_pose[2, 3] = (
            placement_z + pickup_obj_clearance_offset + 0.01  # epsilon
        )  # On same support surface as receptacle base

        if not self.check_feasible_ik(place_pose):
            raise ValueError("IK failed in PickAndPlaceNextToPlannerPolicy for place pose")

        postplace_pose = place_pose.copy()
        postplace_pose[:3, 3] -= self.policy_config.end_z_offset * postplace_pose[:3, 2]

        if not self.check_feasible_ik(postplace_pose):
            raise ValueError("IK failed in PickAndPlaceNextToPlannerPolicy for post place pose")

        return preplace_pose, place_pose, postplace_pose
