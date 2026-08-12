import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.packing_task import PackingTask
from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler
from molmo_spaces.utils.constants.simulation_constants import OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING
from molmo_spaces.utils.lazy_loading_utils import install_uid

if TYPE_CHECKING:
    from molmo_spaces.configs.base_packing_configs import PackingDataGenConfig

log = logging.getLogger(__name__)

# Default THOR cardboard box UIDs
DEFAULT_BOX_UIDS = [f"Box_{i}" for i in range(1, 31)]


class PackingTaskSampler(PickAndPlaceTaskSampler):
    def __init__(self, config: "PackingDataGenConfig") -> None:
        assert config.task_type == "packing"
        super().__init__(config)
        self.config: PackingDataGenConfig

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        # Call grandparent (PickTaskSampler) to add pickup objects, skip PickAndPlaceTaskSampler's
        # receptacle selection logic since we use THOR box assets directly.
        from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler

        PickTaskSampler.add_auxiliary_objects(self, spec)

        task_sampler_config = self.config.task_sampler_config

        box_uids = task_sampler_config.box_uids or DEFAULT_BOX_UIDS
        uid = np.random.choice(box_uids)
        box_xml = install_uid(uid)

        box_spec = MjSpec.from_file(str(box_xml))
        if len(box_spec.worldbody.bodies) != 1:
            log.warning(
                f"{box_xml} has {len(box_spec.worldbody.bodies)} bodies, expected 1. Using first one."
            )
        box_obj: mujoco.MjsBody = box_spec.worldbody.bodies[0]
        if not box_obj.first_joint():
            box_obj.add_joint(
                name=f"{uid}_jntfree",
                type=mujoco.mjtJoint.mjJNT_FREE,
                damping=OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING,
            )

        attach_frame = spec.worldbody.add_frame(
            pos=[10, 10, 10], quat=R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)
        )
        attach_frame.attach_body(box_obj, task_sampler_config.place_receptacle_namespace, "")
        self.place_receptacle_name = box_obj.name
        self._receptacle_names = [box_obj.name]

        # Save added object for scene recreation
        xml_path_rel = box_xml.relative_to(ASSETS_DIR)
        self.config.task_config.added_objects[box_obj.name] = xml_path_rel

        self._metadata_adder.update(
            {
                box_obj.name: {
                    "asset_id": uid,
                    "category": "Box",
                    "object_enum": "temp_object",
                    "is_static": False,
                }
            }
        )

    def _filter_place_target(self, env, pickup_obj_name, place_target_name) -> bool:
        """Skip size filtering — the box is always large enough for pickup objects."""
        return True

    def _open_box_flaps(self, env: CPUMujocoEnv) -> None:
        """Open all flap joints on the box so objects can be placed inside."""
        model, data = env.current_model, env.current_data
        namespace = self.config.task_sampler_config.place_receptacle_namespace

        opened_count = 0
        for i in range(model.njnt):
            jnt_name = model.joint(i).name
            if namespace in jnt_name and "flap" in jnt_name:
                jnt_range = model.jnt_range[i]
                # Open = extreme of joint range away from 0
                if abs(jnt_range[0]) > abs(jnt_range[1]):
                    target_val = jnt_range[0]
                else:
                    target_val = jnt_range[1]

                qposadr = model.jnt_qposadr[i]
                dofadr = model.jnt_dofadr[i]

                data.qpos[qposadr] = target_val
                data.qvel[dofadr] = 0
                # Also update qpos0 so any reset defaults to open
                model.qpos0[qposadr] = target_val
                # Freeze the joint with very high damping so flaps stay open during simulation
                model.dof_damping[dofadr] = 1e6
                opened_count += 1
                log.info(
                    f"Flap '{jnt_name}': qposadr={qposadr}, set qpos={target_val:.3f}, "
                    f"verified qpos={data.qpos[qposadr]:.3f}"
                )

        if opened_count == 0:
            log.warning(
                f"No flap joints found with namespace '{namespace}'. "
                f"All joints: {[model.joint(i).name for i in range(model.njnt)]}"
            )
        else:
            log.info(f"Opened and frozen {opened_count} box flap joints")

        mujoco.mj_forward(model, data)

    def _sample_task(self, env: CPUMujocoEnv) -> PackingTask:
        """Sample a packing task — open box flaps, then delegate placement to parent."""
        # First let parent handle all placement (box, robot, cameras)
        _ = super()._sample_task(env)

        # Open flaps AFTER placement so nothing can overwrite the qpos values
        self._open_box_flaps(env)

        # Verify flap qpos values survived
        model, data = env.current_model, env.current_data
        namespace = self.config.task_sampler_config.place_receptacle_namespace
        for i in range(model.njnt):
            jnt_name = model.joint(i).name
            if namespace in jnt_name and "flap" in jnt_name:
                log.info(f"VERIFY '{jnt_name}': qpos={data.qpos[model.jnt_qposadr[i]]:.3f}")

        return PackingTask(env, self.config)
