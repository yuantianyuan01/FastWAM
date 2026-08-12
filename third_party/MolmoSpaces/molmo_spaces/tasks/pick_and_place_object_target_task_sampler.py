import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.env.object_manager import Context
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.mj_model_and_data_utils import body_base_pos
from molmo_spaces.utils.pose import pose_mat_to_7d

if TYPE_CHECKING:
    from molmo_spaces.configs.base_pick_and_place_configs import PickAndPlaceDataGenConfig


log = logging.getLogger(__name__)


class AbstractPickAndPlaceObjectTargetTaskSampler(PickTaskSampler, ABC):
    """Abstract base for pick-and-place tasks where the target is another object.

    Provides the shared pick-and-place config sampling pipeline
    (:meth:`_configure_pick_and_place`) but does not create tasks.
    Subclasses must implement :meth:`_get_place_target_candidates`,
    :meth:`_prepare_place_target`, and :meth:`_sample_task`.
    """

    def __init__(self, config: "PickAndPlaceDataGenConfig") -> None:
        self.place_receptacle_name = None
        super().__init__(config)
        self.config: PickAndPlaceDataGenConfig

    @abstractmethod
    def _get_place_target_candidates(
        self,
        env: CPUMujocoEnv,
        pickup_obj_name: str,
        supporting_geom_id: int,
    ) -> list[str]:
        """Return candidate place target names."""
        ...

    @abstractmethod
    def _prepare_place_target(
        self,
        env: CPUMujocoEnv,
        place_target_name: str,
        pickup_obj_name: str,
        pickup_obj_pos,
        supporting_geom_id: int,
    ) -> bool:
        """Validate and position the place target.

        Returns True if a valid target was selected, False to retry with the
        next candidate.  Raise ``ValueError`` to permanently remove the
        candidate from the list.
        """
        ...

    def _on_candidate_selected(
        self,
        env: CPUMujocoEnv,
        reference_obj_name: str,
        reference_obj_id: int,
        supporting_geom_id: int,
    ) -> bool:
        """Find and validate a place target after pickup object is selected."""
        if not super()._on_candidate_selected(
            env, reference_obj_name, reference_obj_id, supporting_geom_id
        ):
            return False

        pickup_obj_name = self.config.task_config.pickup_obj_name
        pickup_obj_pos = body_base_pos(env.current_data, reference_obj_id)

        place_candidates = self._get_place_target_candidates(
            env, pickup_obj_name, supporting_geom_id
        )
        if not place_candidates:
            log.info(f"No place target candidates for {pickup_obj_name}")
            raise ValueError(f"No place target candidates for {pickup_obj_name}")

        if not self._prepare_place_target(
            env,
            self.place_receptacle_name,
            pickup_obj_name,
            pickup_obj_pos,
            supporting_geom_id,
        ):
            log.info(f"No valid place target found for {pickup_obj_name}")
            return False

        self.config.task_config.place_receptacle_name = self.place_receptacle_name
        self.config.task_config.place_target_name = self.place_receptacle_name

        om = env.object_managers[env.current_batch_index]
        receptacle_obj = om.get_object_by_name(self.place_receptacle_name)
        self.config.task_config.place_receptacle_start_pose = pose_mat_to_7d(
            receptacle_obj.pose
        ).tolist()

        return True

    def _build_context_objects(self, env, om, pickup_obj_name, supporting_geom_id):
        """Build the context object list for referral expression generation.

        Override in subclasses to filter context (e.g. remove distractor
        receptacles that should not participate in disambiguation).
        """
        context_objects = om.get_context_objects(
            pickup_obj_name, Context.BENCH, bench_geom_ids=[supporting_geom_id]
        )
        context_names = {obj.name for obj in context_objects}

        if self.place_receptacle_name not in context_names:
            context_objects.append(om.get_object(self.place_receptacle_name))

        if pickup_obj_name not in context_names:
            context_objects.append(om.get_object(pickup_obj_name))

        return context_objects

    def _configure_pick_and_place(self, env: CPUMujocoEnv) -> None:
        """Select pickup/place objects and populate task config.

        Runs the full pick-and-place sampling pipeline: pickup object
        selection, place target validation, context building, and dual
        referral expression generation.  Does NOT create a task — subclasses
        call this from their own ``_sample_task`` and instantiate the
        appropriate task class.
        """
        assert env.current_batch_index == 0
        assert self.candidate_objects is not None and len(self.candidate_objects) > 0

        om = env.object_managers[env.current_batch_index]

        supporting_geom_id = self._select_pickup_object(env)
        pickup_obj_name = self.config.task_config.pickup_obj_name

        context_objects = self._build_context_objects(env, om, pickup_obj_name, supporting_geom_id)

        expression_priority, filtered_expression_priority = self._generate_referral_expressions(
            env, pickup_obj_name, context_objects
        )
        receptacle_expression_priority, filtered_receptacle_expression_priority = (
            self._generate_referral_expressions(env, self.place_receptacle_name, context_objects)
        )

        self.config.task_config.referral_expressions["pickup_name"] = om.sample_expression(
            filtered_expression_priority
        )
        self.config.task_config.referral_expressions_priority["pickup_name"] = expression_priority

        self.config.task_config.referral_expressions["place_name"] = om.sample_expression(
            filtered_receptacle_expression_priority
        )
        self.config.task_config.referral_expressions_priority["place_name"] = (
            receptacle_expression_priority
        )

        self._finalize_task_config(env)

    def _finalize_task_config(self, env: CPUMujocoEnv):
        """Hook for subclasses to update task config before task creation."""
        pass
