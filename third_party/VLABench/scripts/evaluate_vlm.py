import argparse
import json
import os
from VLABench.evaluation.evaluator import VLMEvaluator
from VLABench.evaluation.model.vlm import *

def initialize_model(model_name, *args, **kwargs):
    cls = globals().get(model_name)
    if cls is None:
        raise ValueError(f"Model '{model_name}' not found in the current namespace.")
    
    return cls(*args, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM benchmark with specified model and parameters.")
    parser.add_argument("--vlm_name", type=str, default="GPT_4v", choices=["GPT_4v", "Qwen2_VL", "InternVL2", "MiniCPM_V2_6", "GLM4v", "Llava_NeXT", "Gemini", "Claude"], help="Name of the model class to instantiate")
    parser.add_argument("--save_interval", type=int, default=1, help="Interval for saving benchmark results")
    parser.add_argument("--few-shot-num", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--eval-dimension", nargs="+", type=str, default=["M&T", "CommonSense", "Semantic", "Spatial", "PhysicalLaw", "Complex"], help="evaluation dimensions")
    parser.add_argument("--tasks", nargs='+', default=None, help="Specific tasks to run, default is None, meaning evaluate on all the tasks")
    parser.add_argument("--n-episodes", type=int, default=100, help="Number of episodes to evaluate for a task")
    parser.add_argument("--with-cot", default=False, action="store_true", help="Whether to use chain of thought")
    return parser.parse_args()

def main():
    args = parse_args()
    assert len(args.eval_dimension) > 0, "Please specify the evaluation dimension"
    
    for eval_dim in args.eval_dimension:
        if args.tasks is None:
            task_list = os.listdir(os.path.join(os.getenv("VLABENCH_ROOT"), "../dataset", f"vlm_evaluation_v1.0/{eval_dim}"))
        else:
            task_list = args.tasks
        evaluator = VLMEvaluator(
            tasks=task_list, 
            n_episodes=args.n_episodes,
            data_path=os.path.join(os.getenv("VLABENCH_ROOT"), "../dataset", f"vlm_evaluation_v1.0/{eval_dim}"),
            save_path=os.path.join(os.getenv("VLABENCH_ROOT"), "../logs/vlm"),
        )
        
        vlm = initialize_model(args.vlm_name) 
        
        if args.task_list_json is not None:
            try:
                pwd = os.getcwd()
                task_list_path = os.path.join(pwd, "../../configs/benchmark/taskList", args.task_list_json)
                with open(task_list_path, 'r') as f:
                    task_list = json.load(f)
            except:
                task_list = None
        else:
            task_list = []

        evaluator.evaluate(
            vlm,
            save_interval=args.save_interval,
            few_shot_num=args.few_shot_num,
            with_CoT=args.with_cot,
            eval_dim=eval_dim,
        )
        result=evaluator.get_final_score_dict(args.vlm_name)
        os.makedirs(os.path.join(args.save_dir, args.vlm_name), exist_ok=True)
        with open(os.path.join(args.save_dir, args.vlm_name, f"{eval_dim}_result.json"), "w") as f:
            json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()