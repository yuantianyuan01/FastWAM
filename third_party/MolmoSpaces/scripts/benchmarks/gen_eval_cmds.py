
prefix = '''export MUJOCO_INSTALL_DIR="/weka/prior/datasets/mujoco/"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
conda activate mjthor311

python /weka/prior/maxa/code/openpi/scripts/serve_policy.py --port=8080 policy:checkpoint --policy.config=pi05_droid_jointpos --policy.dir=/weka/prior/maxa/code/openpi/checkpoints/pi05_droid_jointpos/ &

cd ~/code/mujoco-thor
'''

benchmark_dirs = [
    #"/weka/prior/datasets/robomolmo/bench_v4/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_benchmark/",
    #"/weka/prior/datasets/robomolmo/bench_jordis/procthor-objaverse/FrankaPickandPlaceNextToHardBench/FrankaPickandPlaceNextToHardBench_20260322_1000eps_json_benchmark",
    #"/weka/prior/datasets/robomolmo/bench_v4/procthor-objaverse/FrankaPickandPlaceColorHardBench/FrankaPickandPlaceColorHardBench_20260304_json_benchmark/"
    #"/weka/prior/datasets/robomolmo/bench_v4/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark"
    "/weka/prior/datasets/robomolmo/bench_v4/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_benchmark",
    #"/weka/prior/datasets/robomolmo/bench_jordis/procthor-objaverse/FrankaPickandPlaceNextToHardBench/FrankaPickandPlaceNextToHardBench_20260322_1000eps_json_benchmark",
]

num_episodes = 1000
workers = 4
skip_episodes_per_worker = num_episodes // workers

for i, benchmark_dir in enumerate(benchmark_dirs):
    for j in range(workers):
        skip_episodes = j * skip_episodes_per_worker
        max_episodes = skip_episodes_per_worker
        cmd = f"python molmo_spaces/evaluation/eval_main.py --benchmark_dir {benchmark_dir}  --wandb_project pi05_eval  --checkpoint_path /weka/prior/maxa/code/openpi/checkpoints/pi05_droid_jointpos molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig --use-filament --num_workers 5 --skip_episodes={skip_episodes} --max_episodes={max_episodes}"

        whole_cmd = prefix + cmd
        print()
        print(f"{i} - {j}:", "-" * 80)
        print()
        print(whole_cmd)
