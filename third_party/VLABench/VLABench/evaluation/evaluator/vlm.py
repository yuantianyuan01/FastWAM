import os
import json
import time
import random
import traceback
from VLABench.evaluation.utils import *
from VLABench.evaluation.evaluator.base import Evaluator
import warnings
warnings.filterwarnings("ignore")
from colorama import Fore, Back, Style, init
init(autoreset=True)

class VLMEvaluator(Evaluator):
    def __init__(self, 
                 tasks,
                 n_episodes,
                 data_path = "./dataset", 
                 save_path = "./output", 
                 language="en"):
        """
        VLM evaluator, support evaluation on different models, few-shot number, and languages
        """
        super().__init__(tasks, n_episodes)
        self.data_path = data_path
        assert os.path.exists(data_path), "Data path does not exist"
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.all_task_list = os.listdir(data_path)
        
        self.eval_tasks = tasks
        with open(os.path.join(os.environ["VLABENCH_ROOT"], f"configs/prompt/eval_vlm_{language}.txt"), 'r', encoding='utf-8') as file:
            self.pre_prompt = file.read()
        seq_independent_task_config_path = os.path.join(os.environ["VLABENCH_ROOT"], "configs/evaluation/seq_independent_task.json")
        with open(seq_independent_task_config_path, 'r') as f:
            self.seq_independent_task = json.load(f)
        dim2task_config_path = os.path.join(os.environ["VLABENCH_ROOT"], "configs/evaluation/dim2task.json")
        with open(dim2task_config_path, 'r') as f:
            self.dim2task = json.load(f)
        self.task2dim = {} # task_name to evaluation dimension
        self.get_task2dim()
        self.language = language
        
    def get_task2dim(self):
        for dim in self.dim2task:
            for task in self.dim2task[dim]:
                self.task2dim[task] = dim
        return self.task2dim

    def change_language(self, language):
        self.language = language
        self.pre_prompt = open(f'base_prompt_{language}.txt', 'r').read()

    def build_input(self, task_name, example_num, few_shot_num=0):
        prepared_input = {}
        prepared_input["pre_prompt"] = self.pre_prompt

        if few_shot_num != 0:
            prepared_input["shot_input_pic"] = {}
            prepared_input["shot_input_pic_gt"] = {}
            prepared_input["shot_input_instruction"] = {}
            prepared_input["shot_output"] = {}
        for i in range(few_shot_num):
            shot_task_name = random.choice(self.all_task_list)
            shot_example_num = int(random.choice(os.listdir(os.path.join(self.data_path, shot_task_name)))[7:])
            while shot_task_name == task_name and shot_example_num == example_num:
                shot_task_name = random.choice(self.all_task_list)
                shot_example_num = int(random.choice(os.listdir(os.path.join(self.data_path, shot_task_name)))[7:])

            shot_input_pic, shot_input_pic_gt, shot_input_instruction = self.load_single_input(shot_task_name, shot_example_num)
            shot_output = self.load_single_output(shot_task_name, shot_example_num)

            prepared_input["shot_input_pic"][str(i)] = shot_input_pic
            prepared_input["shot_input_pic_gt"][str(i)] = shot_input_pic_gt
            prepared_input["shot_input_instruction"][str(i)] = shot_input_instruction
            prepared_input["shot_output"][str(i)] = shot_output

        input_pic, input_pic_gt, input_instruction  = self.load_single_input(task_name, example_num)
        prepared_input["input_pic"] = input_pic
        prepared_input["input_pic_gt"] = input_pic_gt
        prepared_input["input_instruction"] = input_instruction

        return prepared_input
    
    def get_result_save_path(self, vlm_name, few_shot_num, with_CoT, eval_dim):
        model_result_save_path = os.path.join(self.save_path, vlm_name, self.language)
        if not with_CoT:
            model_result_save_path = os.path.join(model_result_save_path, str(few_shot_num) + "_shot", eval_dim)
        else:
            model_result_save_path = os.path.join(model_result_save_path,  str(few_shot_num) + "_shot_CoT", eval_dim)
        if not os.path.exists(model_result_save_path):
            os.makedirs(model_result_save_path)
        return model_result_save_path
    
    def load_single_input(self, task_name, example_num):
        input_pic_path = os.path.join(self.data_path, task_name, "example"+ str(example_num), 'input/input.png')
        input_pic_gt_path = os.path.join(self.data_path, task_name, "example"+ str(example_num), 'input/input_mask.png')
        input_instruction_path = os.path.join(self.data_path, task_name, "example"+ str(example_num), 'input/instruction.txt')

        input_pic = input_pic_path
        input_pic_gt = input_pic_gt_path
        input_instruction = input_instruction = open(input_instruction_path, 'r', encoding='utf-8').read()

        return input_pic, input_pic_gt, input_instruction
    
    def load_single_output(self, task_name, example_num):
        gt_operation_sequence_path = os.path.join(self.data_path, task_name, "example"+ str(example_num), 'output/operation_sequence.json')
        with open(gt_operation_sequence_path) as f:
            gt_operation_sequence = json.load(f)
        return gt_operation_sequence
    
    def get_single_anwer(self, task_name, example_num, vlm, few_shot_num = 0,with_CoT=False):
        outputs = vlm.evaluate(self.build_input(task_name, example_num, few_shot_num), self.language, with_CoT=with_CoT)
        if not isinstance(outputs, dict):
            return {"format_error": outputs}
        return outputs
    
    def check_filled_output(self, answer):
        if any([key in answer for key in ["skill_sequence", "format_error"]]):
            return True
        return False

    def evaluate(self, vlm, task_list=None, save_interval=1, few_shot_num=0, with_CoT=False, eval_dim="default"):
        """
        param:
          vlm: the wrapped vlm model with standard interface
          task_list: the list of tasks to evaluate, if None, evaluate all tasks
          save_interval: the interval to save the output
          few_shot_num: the few-shot number
          with_CoT: use CoT or not
        """
        print(Fore.YELLOW + Style.BRIGHT + "\n\nworking on ",end = "")
        print(Fore.BLUE + Style.BRIGHT + vlm.name)

        if task_list is None or len(task_list) == 0:
            task_list = self.eval_tasks
        model_result_save_path = self.get_result_save_path(vlm.name, few_shot_num, with_CoT, eval_dim)
        if not os.path.exists(model_result_save_path):
            os.makedirs(model_result_save_path)

        model_result_output_save_file = os.path.join(model_result_save_path, "output.json")
        if os.path.exists(model_result_output_save_file):
            with open(model_result_output_save_file) as f:
                model_output = json.load(f)
        else:
            model_output = {}
            model_output["benchmeatinfo"] = {}
            model_output["benchmeatinfo"]["existing_num"] = 0
            model_output["benchmeatinfo"]["already_running_time"] = 0

        test_example_list = []
        is_resuming = False
        existing_num = 0
        for task_name in task_list:
            for example_num in range(len(os.listdir(os.path.join(self.data_path, task_name)))):
                if task_name in model_output and str(example_num) in model_output[task_name]:
                    if self.check_filled_output(model_output[task_name][str(example_num)]):
                        is_resuming = True
                        existing_num += 1
                        continue
                test_example_list.append((task_name, str(example_num)))
                
        if len(test_example_list) == 0:
            print(Fore.MAGENTA + Style.BRIGHT + "All examples are already exist")
            return model_output

        if model_output["benchmeatinfo"]["existing_num"] == 0:
            model_output["benchmeatinfo"]["existing_num"] = existing_num
            model_output["benchmeatinfo"]["already_running_time"] = existing_num * 10

        elif model_output["benchmeatinfo"]["existing_num"] != existing_num:
            model_output["benchmeatinfo"]["already_running_time"] = model_output["benchmeatinfo"]["already_running_time"] * existing_num / model_output["benchmeatinfo"]["existing_num"]
            model_output["benchmeatinfo"]["existing_num"] = existing_num
        already_running_time = model_output["benchmeatinfo"]["already_running_time"]

        test_example_num = len(test_example_list)
        working_start_time = time.time()
        working_number = 0
        print(Fore.YELLOW + Style.BRIGHT + "working start at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        if is_resuming: 
            print(Fore.MAGENTA + Style.BRIGHT + "{} example existed. Resume start at task: {}, example: {}".format(existing_num, test_example_list[0][0], test_example_list[0][1]))
        
        for task_name, example_num in test_example_list:
            try:
                if task_name not in model_output:
                    model_output[task_name] = {}
                if example_num not in model_output[task_name]:
                    model_output[task_name][example_num] = {}
                #   answer should be a dict with keys: operation_sequence / format_error
                #   if format_error is not in dict, it means the answer is not valid
                answer = self.get_single_anwer(task_name, example_num, vlm, few_shot_num, with_CoT)
                model_output[task_name][example_num] = answer
            except Exception as e:
                print("\n\nError in task: ", task_name, " example: ", example_num)
                print(e)
                traceback.print_exc()
                new_existing_num = existing_num + working_number
                model_output["benchmeatinfo"]["existing_num"] = new_existing_num
                model_output["benchmeatinfo"]["already_running_time"] = time.time() - working_start_time + already_running_time
                with open(model_result_output_save_file, 'w', encoding="utf-8") as f:
                    json.dump(model_output, f, ensure_ascii=False, indent=4)
                raise e
            
            if len(model_output) % save_interval == 0:
                new_existing_num = existing_num + working_number
                model_output["benchmeatinfo"]["existing_num"] = new_existing_num
                model_output["benchmeatinfo"]["already_running_time"] = time.time() - working_start_time + already_running_time
                with open(model_result_output_save_file, 'w', encoding="utf-8") as f:
                    json.dump(model_output, f, ensure_ascii=False, indent=4)

            working_number += 1

            now_time = time.time()
            total_using_time = now_time - working_start_time + already_running_time
            average_using_time = total_using_time / (working_number + existing_num)
            predict_time = average_using_time * (test_example_num - working_number)
            question_percentage = int((working_number + existing_num) / (test_example_num + existing_num) * 100)
            print(Fore.GREEN + question_percentage*'-',end='', flush=True)
            print(Fore.RED + (100-question_percentage)*'-',end='', flush=True)
            print(Fore.GREEN +  "{:>3}%({:>3}/{:>3})".format(question_percentage, (working_number + existing_num), (test_example_num + existing_num)),end='', flush=True)
            print(Fore.GREEN + " using:{:>3}h{:>2}m{:>2}s, ".format(int(total_using_time/3600), int((total_using_time%3600)/60), int(total_using_time%60)),end='', flush=True)
            print(Fore.GREEN + "remain:{:>3}h{:>2}m{:>2}s".format(int(predict_time/3600), int((predict_time%3600)/60), int(predict_time%60)),end='         \r', flush=True)


        new_existing_num = existing_num + working_number
        model_output["benchmeatinfo"]["existing_num"] = new_existing_num
        model_output["benchmeatinfo"]["already_running_time"] = time.time() - working_start_time + already_running_time
        with open(model_result_output_save_file, 'w', encoding="utf-8") as f:
            json.dump(model_output, f, ensure_ascii=False, indent=4)
        print(Fore.YELLOW + Style.BRIGHT + "working end at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    def get_final_score_dict(self, vlm_name, few_shot_num=0, with_CoT=False):
        output_file = os.path.join(self.get_result_save_path(vlm_name, few_shot_num, with_CoT), "output.json")
        if not os.path.exists(output_file):
            print(Fore.RED + Style.BRIGHT + "output file not exist for model: ", vlm_name, " few_shot_num: ", few_shot_num, " with_CoT: ", with_CoT)
            return None
        with open(output_file) as f:
            model_output = json.load(f)

        final_score_dict = {}

        for task_name in model_output:
            if task_name == "benchmeatinfo":
                continue
            if task_name not in final_score_dict:
                final_score_dict[task_name] = {}
            for example_num in model_output[task_name]:
                if example_num not in final_score_dict[task_name]:
                    final_score_dict[task_name][example_num] = {}

                if "format_error" in model_output[task_name][example_num]:
                    final_score_dict[task_name][example_num] = {
                        "skill_match_score": 0,
                        "entity_match_score": 0,
                        "skill_with_entity_match_score": 0,
                        "exact_match_score": 0,
                        "total_score": 0
                    }
                    continue
                standard_output = self.load_single_output(task_name, example_num)["skill_sequence"]
                try:
                    model_skill_sequence = model_output[task_name][example_num]["skill_sequence"]
                    dependency = "Sequential" if task_name not in self.seq_independent_task else "Seq-independent"
                    example_score = get_final_score(standard_output, model_skill_sequence, dependency=dependency)
                    final_score_dict[task_name][example_num] = example_score
                except:
                    final_score_dict[task_name][example_num] = {
                        "skill_match_score": 0,
                        "entity_match_score": 0,
                        "skill_with_entity_match_score": 0,
                        "exact_match_score": 0,
                        "total_score": 0
                    }
                
        final_score_dict_save_path = os.path.join(self.get_result_save_path(vlm_name, few_shot_num, with_CoT), "final_score.json")
        with open(final_score_dict_save_path, 'w', encoding="utf-8") as f:
            json.dump(final_score_dict, f, ensure_ascii=False, indent=4)
        return final_score_dict

    def get_six_dim_result(self, vlm_name, few_shot_num = 0, with_CoT=False):
        six_dim_result_save_path = os.path.join(self.get_result_save_path(vlm_name, few_shot_num, with_CoT), "six_dim_result.json")
        if os.path.exists(six_dim_result_save_path):
            with open(six_dim_result_save_path) as f:
                six_dim_result = json.load(f)
            return six_dim_result
        
        final_score_file = os.path.join(self.get_result_save_path(vlm_name, few_shot_num, with_CoT), "final_score.json")
        final_score_dict = None
        if not os.path.exists(final_score_file):
            final_score_dict = self.get_final_score_dict(vlm_name, few_shot_num=few_shot_num, with_CoT=with_CoT)
        else:
            with open(final_score_file) as f:
                final_score_dict = json.load(f)
            
        if final_score_dict is None:
            print(Fore.RED + Style.BRIGHT + "get final score dict failed for model: ", vlm_name, " few_shot_num: ", few_shot_num, " with_CoT: ", with_CoT)
            return None
        
        six_dim_result = {}
        for dim in self.dim2task:
            six_dim_result[dim] = {}
            six_dim_result[dim]["skill_match_score"] = 0
            six_dim_result[dim]["entity_match_score"] = 0
            six_dim_result[dim]["skill_with_entity_match_score"] = 0
            six_dim_result[dim]["exact_match_score"] = 0
            six_dim_result[dim]["total_score"] = 0
            six_dim_result[dim]["example_num"] = 0

        for task_name in final_score_dict:
            for example_num in final_score_dict[task_name]:
                dim = self.task2dim[task_name]
                for key in six_dim_result[dim]:
                    if key != "example_num":   
                        six_dim_result[dim][key] += final_score_dict[task_name][example_num][key]
                six_dim_result[dim]["example_num"] += 1

        for dim in six_dim_result:
            for key in six_dim_result[dim]:
                if key != "example_num":
                    six_dim_result[dim][key] = six_dim_result[dim][key] / six_dim_result[dim]["example_num"]

        six_dim_result_save_path = os.path.join(self.get_result_save_path(vlm_name, few_shot_num, with_CoT), "six_dim_result.json")
        with open(six_dim_result_save_path, 'w', encoding="utf-8") as f:
            json.dump(six_dim_result, f, ensure_ascii=False, indent=4)
        return six_dim_result
                    

