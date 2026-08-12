import logging
from typing import TYPE_CHECKING

import numpy as np

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.env.object_manager import Context
from molmo_spaces.tasks.pick_and_place_next_to_task import PickAndPlaceNextToTask
from molmo_spaces.tasks.pick_and_place_object_target_task_sampler import (
    AbstractPickAndPlaceObjectTargetTaskSampler,
)
from molmo_spaces.tasks.pick_and_place_task_sampler import MAX_BOTTOM_Z_DIFFERENCE
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb

if TYPE_CHECKING:
    from molmo_spaces.configs.base_pick_and_place_next_to_configs import (
        PickAndPlaceNextToDataGenConfig,
    )

log = logging.getLogger(__name__)


class PickAndPlaceNextToTaskSampler(AbstractPickAndPlaceObjectTargetTaskSampler):
    def __init__(self, config: "PickAndPlaceNextToDataGenConfig") -> None:
        super().__init__(config)
        self.config: PickAndPlaceNextToDataGenConfig
        self._place_candidates: list[str] = []

    def _get_place_target_candidates(
        self,
        env: CPUMujocoEnv,
        pickup_obj_name: str,
        supporting_geom_id: int,
    ) -> list[str]:
        """Return objects on the same bench, excluding the pickup object.

        Filters to objects whose type is in place_receptacle_types config.
        Filters to objects on the same surface (similar bottom Z).
        Uses synset-based comparison to prefer objects of different types.
        If an object is labeled by a hypernym in context, objects that are
        hyponyms of that hypernym are considered the same type.
        """
        om = env.object_managers[env.current_batch_index]
        data = env.current_data

        context_objects = om.get_context_objects(
            pickup_obj_name, Context.BENCH, bench_geom_ids=[supporting_geom_id]
        )

        if not context_objects:
            self._place_candidates = []
            return self._place_candidates

        pickup_obj = om.get_object_by_name(pickup_obj_name)
        pickup_center, pickup_size = body_aabb(data.model, data, pickup_obj.body_id)
        pickup_bottom_z = pickup_center[2] - pickup_size[2] / 2

        same_surface_objects = []
        for obj in context_objects:
            obj_center, obj_size = body_aabb(data.model, data, obj.body_id)
            obj_bottom_z = obj_center[2] - obj_size[2] / 2
            if abs(obj_bottom_z - pickup_bottom_z) <= MAX_BOTTOM_Z_DIFFERENCE:
                same_surface_objects.append(obj)
        context_objects = same_surface_objects

        if not context_objects:
            self._place_candidates = []
            return self._place_candidates

        place_receptacle_types = self.config.task_sampler_config.place_receptacle_types
        if place_receptacle_types:
            place_types_set = set(t.lower() for t in place_receptacle_types)
            filtered_context_objects = []
            for obj in context_objects:
                obj_types = om.get_possible_object_types(obj.name)
                if any(t.lower() in place_types_set for t in obj_types):
                    filtered_context_objects.append(obj)
            context_objects = filtered_context_objects

        if not context_objects:
            self._place_candidates = []
            return self._place_candidates

        context_synsets = om.get_context_synsets(context_objects)

        pickup_hypernyms = set(om.get_object_hypernyms(pickup_obj_name, context_synsets))
        pickup_synset = om.most_concrete_synset(pickup_hypernyms)

        different_synset = []
        same_synset = []

        for obj in context_objects:
            if obj.name == pickup_obj_name:
                continue

            obj_hypernyms = set(om.get_object_hypernyms(obj.name, context_synsets))
            if obj_hypernyms:
                obj_synset = om.most_concrete_synset(obj_hypernyms)
            else:
                continue

            if pickup_synset == obj_synset:
                same_synset.append(obj.name)
            else:
                different_synset.append(obj.name)

        np.random.shuffle(different_synset)
        np.random.shuffle(same_synset)

        all_candidates = different_synset + same_synset

        if all_candidates and self._task_counter is not None:
            rotation = self._task_counter % len(all_candidates)
            all_candidates = all_candidates[rotation:] + all_candidates[:rotation]

        self._place_candidates = all_candidates
        return self._place_candidates

    def _prepare_place_target(
        self,
        env: CPUMujocoEnv,
        place_target_name: str,
        pickup_obj_name: str,
        pickup_obj_pos: np.ndarray,
        supporting_geom_id: int,
    ) -> bool:
        """Check that place target is at a reasonable distance from pickup object.

        Objects are already placed in scene. We reject candidates that are:
        - Too close: already within success range (task would be trivial)
        - Too far: beyond max_object_to_receptacle_dist (IK would likely fail)
        """
        om = env.object_managers[env.current_batch_index]
        data = env.current_data
        task_sampler_config = self.config.task_sampler_config

        pickup_obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)

        max_dist = task_sampler_config.max_object_to_receptacle_dist
        min_dist = task_sampler_config.min_object_to_receptacle_dist

        for candidate_name in self._place_candidates:
            place_target = om.get_object_by_name(candidate_name)

            pickup_center, pickup_size = body_aabb(data.model, data, pickup_obj.body_id)
            target_center, target_size = body_aabb(data.model, data, place_target.body_id)

            center_to_center_dist = np.linalg.norm(target_center[:2] - pickup_center[:2])

            if center_to_center_dist > max_dist:
                log.info(
                    f"Rejecting {candidate_name}: too far from pickup object "
                    f"(center distance={center_to_center_dist:.3f}m > max={max_dist:.3f}m)"
                )
                continue

            if center_to_center_dist < min_dist:
                log.info(
                    f"Rejecting {candidate_name}: too close to pickup object "
                    f"(center distance={center_to_center_dist:.3f}m < min={min_dist:.3f}m)"
                )
                continue

            self.place_receptacle_name = candidate_name
            return True

        return False

    def _sample_task(self, env: CPUMujocoEnv) -> PickAndPlaceNextToTask:
        self._configure_pick_and_place(env)
        return PickAndPlaceNextToTask(env, self.config)
