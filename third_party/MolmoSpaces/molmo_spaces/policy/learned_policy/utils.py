import hashlib
import logging

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.object_metadata import ObjectMeta

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PromptSampler:
    DEFAULT_TEMPLATES_BY_TASK = {
        "pick": [
            "pick up the {}.",
        ],
        "open": ["open the {}."],
        "pick_and_place": [
            "pick up the {} and place it on the {}.",
        ],
        "packing": [
            "pack container.",
        ],
        "close": [
            "close the {}.",
        ],
    }

    def __init__(
        self,
        task_type: str = "pick",
        prompt_templates: list[str] = None,
        prompt_object_word_num: int = 1,
        disambiguate_distractors_by_pos: bool = False,
    ) -> None:
        """
        Args:
            task_type: The type of task to sample prompts for.
            prompt_templates: A list of prompt templates to sample from. If None, the default templates for the task type will be used.
            prompt_object_word_num: The number of words to use for the object name in the prompt.
            disambiguate_distractors_by_pos: Whether to disambiguate distractors by position in the prompt.
                This relies on functionality only present when using a frozen config.
        """
        if prompt_templates is not None and task_type in ["pick", "pick_and_place"]:
            self.prompt_templates = prompt_templates
        elif task_type in self.DEFAULT_TEMPLATES_BY_TASK:
            self.prompt_templates = self.DEFAULT_TEMPLATES_BY_TASK[task_type]
        else:
            raise ValueError(
                f"Unknown task_type '{task_type}'. "
                f"Available task types: {list(self.DEFAULT_TEMPLATES_BY_TASK.keys())}"
            )
        self.task_type = task_type
        self.current_index = -1
        self.prompt_object_word_num = prompt_object_word_num
        self._cached_prompt = None
        self._disambiguate_distractors_by_pos = disambiguate_distractors_by_pos

    def get_state(self):
        return {
            "current_index": self.current_index,
            "cached_prompt": self._cached_prompt,
        }

    def set_state(self, state):
        self.current_index = state["current_index"]
        self._cached_prompt = state["cached_prompt"]

    def next(self) -> None:
        self.current_index = (self.current_index + 1) % len(self.prompt_templates)
        self._cached_prompt = None

    def get_target_object_uid(self, task):
        return ObjectMeta.get_target_object_uid(task)

    def get_short_description(self, object_uid):
        return ObjectMeta.get_short_description(object_uid)

    def get_prompt(self, task: BaseMujocoTask) -> str:
        if self._cached_prompt is not None:
            return self._cached_prompt

        object_uid = self.get_target_object_uid(task)
        target_name = task.env.config.task_config.pickup_obj_name

        # Check if this is a custom object with a provided name
        eval_params = task.env.config.eval_runtime_params
        if (
            eval_params
            and eval_params.custom_object_name
            and target_name.startswith("custom_object/")
        ):
            # Use the provided custom object name directly
            object_name = eval_params.custom_object_name.lower()
        else:
            # Standard object handling
            short_descriptions: list[str] = ObjectMeta.short_descriptions(object_uid)
            target_category = "_".join(target_name.split("_")[0:1])

            if not short_descriptions:
                object_name = target_category
            elif self.prompt_object_word_num == 0:
                description = short_descriptions[3].lower()
                object_name = short_descriptions[0].lower()
                object_name = description.replace(object_name, "object")
            else:
                object_name = short_descriptions[self.prompt_object_word_num - 1].lower()

        if self._disambiguate_distractors_by_pos and self.task_type in ["pick", "pick_and_place"]:
            # TODO: this should pull from metadata or something, since object_poses is not guaranteed to be set
            target_pose = task.env.config.task_config.object_poses[target_name]
            robot_pose = task.env.config.task_config.robot_base_pose
            T_world_robot = np.eye(4)
            T_world_robot[:3, 3] = robot_pose[:3]
            T_world_robot[:3, :3] = R.from_quat(robot_pose[3:7], scalar_first=True).as_matrix()
            T_world_target = np.eye(4)
            T_world_target[:3, 3] = target_pose[:3]
            T_world_target[:3, :3] = R.from_quat(target_pose[3:7], scalar_first=True).as_matrix()
            T_robot_target = np.linalg.inv(T_world_robot) @ T_world_target
            target_pos = T_robot_target[:3, 3]

            distractors_pos = []

            for (
                distractor_name,
                distractor_pose,
            ) in task.env.config.task_config.object_poses.items():
                if (
                    distractor_name == target_name
                    or "_".join(distractor_name.split("_")[0:1]) != target_category
                ):
                    continue
                T_world_distractor = np.eye(4)
                T_world_distractor[:3, 3] = distractor_pose[:3]
                T_world_distractor[:3, :3] = R.from_quat(
                    distractor_pose[3:7], scalar_first=True
                ).as_matrix()
                T_robot_distractor = np.linalg.inv(T_world_robot) @ T_world_distractor
                if (
                    np.linalg.norm(T_robot_distractor[:3, 3] - T_robot_target[:3, 3]) > 1.0
                    or np.linalg.norm(T_robot_distractor[:3, 3]) > 1.0
                ):
                    continue
                distractors_pos.append(T_robot_distractor[:3, 3])

            if len(distractors_pos) > 0:
                distractors_array = np.array(distractors_pos)

                deltas = target_pos - distractors_array
                abs_deltas = np.abs(deltas)
                min_indices = np.argmin(abs_deltas, axis=0)
                min_components = np.array(
                    [
                        deltas[min_indices[0], 0],
                        deltas[min_indices[1], 1],
                        deltas[min_indices[2], 2],
                    ]
                )
                max_component_index = np.argmax(np.abs(min_components))
                min_component_value = min_components[max_component_index]
                if max_component_index == 1:
                    object_name += " on the left" if min_component_value > 0 else " on the right"
                elif max_component_index == 0:
                    object_name += " in the back" if min_component_value > 0 else " in front"
                else:
                    object_name += " above" if min_component_value > 0 else " below"

        if self.task_type == "pick_and_place":
            # Get place receptacle name from config (format: "place_receptacle/<uid>")
            place_receptacle_full_name = task.env.config.task_config.place_receptacle_name
            if place_receptacle_full_name:
                receptacle_uid = place_receptacle_full_name.split("/")[-1]
                receptacle_short_descriptions: list[str] = ObjectMeta.short_descriptions(
                    receptacle_uid
                )

                if not receptacle_short_descriptions:
                    log.warning(
                        "No receptacle short descriptions found, defaulting to 'receptacle'"
                    )
                    receptacle_name = "receptacle"
                elif self.prompt_object_word_num == 0:
                    description = receptacle_short_descriptions[3].lower()
                    base_name = receptacle_short_descriptions[0].lower()
                    receptacle_name = description.replace(base_name, "object")
                else:
                    receptacle_name = receptacle_short_descriptions[
                        self.prompt_object_word_num - 1
                    ].lower()
            else:
                log.warning("No place receptacle found in config, defaulting to 'receptacle'")
                receptacle_name = "receptacle"

            self._cached_prompt = self.prompt_templates[self.current_index].format(
                object_name, receptacle_name
            )
        else:
            self._cached_prompt = self.prompt_templates[self.current_index].format(object_name)

        log.info(f"The prompt is: {self._cached_prompt}")
        return self._cached_prompt

    def clean_object_name(self, task: BaseMujocoTask) -> str:
        return self.get_short_description(self.get_target_object_uid(task))[0]


def resize_with_pad(images, height, width):
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [
            _resize_with_pad_pil(Image.fromarray(im), height, width, method=Image.BILINEAR)
            for im in images
        ]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image, height, width, method):
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


def generate_object_hash(asset_id: str) -> str:
    hasher = hashlib.md5()
    hasher.update(asset_id.encode())
    return hasher.hexdigest()
