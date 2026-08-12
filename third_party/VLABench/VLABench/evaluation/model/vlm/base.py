import json
import os
import torch

class BaseVLM():
    def __init__(self) -> None:
        self.name =self.get_name()

    def evaluate(self, input_dict, language, with_CoT=False):
        raise NotImplementedError
    
    def get_name(self):
        return "BaseVLM"
    
def get_ti_list(input_dict, language, with_CoT=False):
    if language == "zh":
        ti_list = []
        ti_list.append(["text", input_dict["pre_prompt"] ])
        if "shot_input_pic" in input_dict:
            for shot_num in input_dict["shot_input_pic"]:
                ti_list.append(["text", "示范"+shot_num+"输入图片:" ])
                ti_list.append(["image", input_dict["shot_input_pic"][shot_num]])
                ti_list.append(["text", "示范"+shot_num+"输入带编号标签的图片" ])
                ti_list.append(["image", input_dict["shot_input_pic_gt"][shot_num]])
                ti_list.append(["text", "示范"+shot_num+"语言指令:"])
                ti_list.append(["text", input_dict["shot_input_instruction"][shot_num]])
                ti_list.append(["text", "示范"+shot_num+"输出技能序列"])
                ti_list.append(["text", json.dumps(input_dict["shot_output"][shot_num])])
        ti_list.append(["text", "输入图片" ])
        ti_list.append(["image", input_dict["input_pic"]])
        ti_list.append(["text", "输入带编号标签的图片" ])
        ti_list.append(["image", input_dict["input_pic_gt"]])
        ti_list.append(["text", "语言指令:"])
        ti_list.append(["text", input_dict["input_instruction"]])
        ti_list.append(["text", "请你给出输出的技能序列"])
        if with_CoT:
            ti_list.append(["text", "请一步一步分析问题最后给出答案"])
    elif language == "en":
        ti_list = []
        ti_list.append(["text", input_dict["pre_prompt"] ])
        if "shot_input_pic" in input_dict:
            for shot_num in input_dict["shot_input_pic"]:
                ti_list.append(["text", "Example "+shot_num+" input picture:" ])
                ti_list.append(["image", input_dict["shot_input_pic"][shot_num]])
                ti_list.append(["text", "Example "+shot_num+" input picture with numbered tags" ])
                ti_list.append(["image", input_dict["shot_input_pic_gt"][shot_num]])
                ti_list.append(["text", "Example "+shot_num+" language instruction:"])
                ti_list.append(["text", input_dict["shot_input_instruction"][shot_num]])
                ti_list.append(["text", "Example "+shot_num+" output skill sequence"])
                ti_list.append(["text", json.dumps(input_dict["shot_output"][shot_num])])
        ti_list.append(["text", "Input picture" ])
        ti_list.append(["image", input_dict["input_pic"]])
        ti_list.append(["text", "Input picture with numbered tags" ])
        ti_list.append(["image", input_dict["input_pic_gt"]])
        ti_list.append(["text", "Language instruction:"])
        ti_list.append(["text", input_dict["input_instruction"]])
        ti_list.append(["text", "Please give the output skill sequence"])
        if with_CoT:
            ti_list.append(["text", "Please analyze the problem step by step and give the answer"])
    else:
        raise ValueError("language should be zh or en")
    return ti_list

