from peft import PeftModel, PeftConfig
import torch
import os
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from VLABench.evaluation.model.policy.base import Policy 
from VLABench.utils.utils import quaternion_to_euler

CAMERA_VIEW_INDEX={
    "select_painting": 1,
    "put_box_on_painting": 1,
    "select_chemistry_tube":2,
    "find_unseen_object":2,
    "texas_holdem": 2,
    "cluster_toy": 2
}

def copy_file_content(content_file, target_file):
    with open(content_file, "r") as f:
        content = f.read()
    with open(target_file, "w") as f:
        f.write(content)

class OpenVLA(Policy):
    system_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    def __init__(self, 
                 model_ckpt,
                 lora_ckpt, 
                 attn_implementation=None,
                 norm_config_file=None,
                 device="cuda",
                 **kwargs):
        
        """
        param:
            model_ckpt: path to the model checkpoint
            lora_ckpt: path to the lora checkpoint
            attn_implementation: the implementation of attention layer, e.g. "torch" or "einsum"
            norm_config_file: path to the config file for denormalization to overide the default config
            device: cuda device to run
        """
        if norm_config_file is not None:
            copy_file_content(norm_config_file, os.path.join(model_ckpt, "config.json"))
        self.processor = AutoProcessor.from_pretrained(model_ckpt, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_ckpt,
            # attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        peft_config = PeftConfig.from_pretrained(lora_ckpt)
        model = PeftModel.from_pretrained(model, lora_ckpt, config=peft_config).to(device)
        self.device = device
        super().__init__(model)
        
    def process_observation(self, obs, unnorm_key):
        cam_index = CAMERA_VIEW_INDEX.get(unnorm_key, 2) # default camera-view is front-view
        instruction = obs["instruction"]
        prompt = self.build_prompt(instruction)
        rgb = obs["rgb"][cam_index]
        inputs = self.processor(prompt, Image.fromarray(rgb).convert("RGB")).to(self.device, dtype=torch.bfloat16)
        return inputs
    
    def build_prompt(self, instruction):
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut: "
        prompt = prompt.replace("_seen", "")
        prompt = prompt.replace("_unseen", "")
        return prompt
    
    def predict(self, obs, unnorm_key=None):
        inputs = self.process_observation(obs, unnorm_key)
        delta_action = self.model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        current_ee_state = obs["ee_state"]
        if len(current_ee_state) == 8:
            pos, quat = current_ee_state[:3], current_ee_state[3:7]
            euler = quaternion_to_euler(quat)
        elif len(current_ee_state) == 7:
            pos, euler = current_ee_state[:3], current_ee_state[3:6]
        target_pos = np.array(pos) + delta_action[:3]
        target_euler = euler + delta_action[3:6]
        gripper_open = delta_action[-1]
        gripper_state = np.ones(2)*0.04 if gripper_open >= 0.1 else np.zeros(2)
        return target_pos, target_euler, gripper_state     
    
    @property
    def name(self):
        return "OpenVLA"