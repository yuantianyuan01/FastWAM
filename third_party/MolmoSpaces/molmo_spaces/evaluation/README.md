# Benchmark Evaluation

Run learned policies on fixed, reproducible benchmarks.


## Resources

This README focuses on benchmark installation and running.

### Related documentation
- Leaderboard documentation can be found [here](https://docs.google.com/document/d/1aRJ_NGWBzdLk3jJ71GvYx-dj1nbATQbDGfSG3V4Iy0g).
- Submitting results, see the GitHub issue in the repository [here](https://github.com/allenai/molmospaces/issues/8).
- Theoretical notes on policy comparison can be found [here](https://docs.google.com/document/d/1FcMxJgAQ_2Ojd2uu8HE2MBfD6RE53zcXa55_r8EfPts)

## Concepts

The MolmoSpaces **leaderboard** shows the results of various polices on benchmarks.

**Benchmark Sets** are collections of individual benchmarks that have been released together, e.g.,
the ones from the MolmoSpaces paper, prefixed by "MS-" or the ones from the MolmoBot paper prefixed by "MB-".

A **benchmark** is a `benchmark.json` file containing a list of self-contained episode specs. Each spec includes everything needed to recreate a task: scene, robot pose, object poses, cameras, language instructions.

An **eval config** is a normal datagen config with your policy attached. When pointed at a benchmark, episode-specific fields (cameras, init_qpos, object_poses, etc.) are overwritten by the benchmark spec. This means you can debug your policy in normal datagen using `datagen/main`, then just swap to benchmark mode for eval by using `eval_main` or importing `run_evaluation` from same.


## Installing the Benchmarks

The benchmark datasets are installed when resource manager gets instantiated. You can either run
```bash
export MLSPACES_ASSETS_DIR=/path/to/symlink/resources  # optional
python -m molmo_spaces.molmo_spaces_constants
```

or you can explicitly code in the python script
```python
from molmospaces.molmo_spaces_constants import get_resource_manager()
get_resource_manager()
```

## Choosing a Benchmark

Which benchmarks should I evaluate on? This depends on what you want to show. The current TLDR is:

- MolmoSpaces Manipulation: Easy manipulation tasks
- MolmoSpaces Navigation: Navigation tasks
- MolmoBot Manipulation: Harder manipulation tasks

For more information see [here](https://docs.google.com/document/d/1aRJ_NGWBzdLk3jJ71GvYx-dj1nbATQbDGfSG3V4Iy0g).


## Running Benchmarks

### Running a Generic Benchmark with Pi

#### 1. Setup: install and run the Pi policy server

Download the checkpoint and start the policy server (leave it running in a separate terminal):

```bash
git clone https://github.com/omarrayyann/openpi
mkdir checkpoints && cd checkpoints
gsutil cp -r gs://openpi-assets/checkpoints/pi05_droid_jointpos .
# other options: `pi05_droid_jointpos`, `pi0_fast_droid_jointpos`, `pi0_droid_jointpos`
```

Install openpi and run the server (default port: 8080):

```bash
uv run scripts/serve_policy.py --port=8080 policy:checkpoint \
  --policy.config=<checkpoint_name> \
  --policy.dir=checkpoints/<checkpoint_name>/
```

#### 2. Run the benchmark

Please look at the concrete commands for each task type in our [leaderboard](https://molmospaces.allen.ai/leaderboard):
- MolmoSpaces tasks (`MS-` prefix): [ms-bench](ms-bench.md)
- MolmoBot tasks (`MB-` prefix): [mb-bench](mb-bench.md)

If using OpenPI models: `pip install openpi_client`.

For this we chose the easy `MS-Pick` benchmark, which is located here `assets/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231/`.

Then launch benchmark episodes in MuJoCo:

```bash
python molmo_spaces/evaluation/eval_main.py \
    molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig \
    --benchmark_dir assets/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231/ \
    --task_horizon_steps 500
```

Make sure the *port* number is the same in `molmo_spaces.configs.policy_configs_baselines:PiPolicyConfig`

Also, see `molmo_spaces/evaluation/configs/evaluation_configs.py` for more examples on eval configs.

#### 3. Run the evaluation
Finally we run the evaluation output script that aggegates the results as csv files.
```bash
python scripts/benchmarks/eval_to_csv.py <eval_output_dir> pi05ft --success-condition oracle  --output-csv data/pick_easy/pi05.csv
```


### Running MolmoSpaces Benchmarks

see [here](./ms-bench.md)

### Running MolmoBot Benchmarks

see [here](./mb-bench.md).

---



## Implementing Eval in an External Repo

You need three things: a policy class, a policy config, and an eval config.

### 1. Policy Class

Extend `InferencePolicy`. Must implement `prepare_model`, `reset`, and `get_action`.

```python
# my_repo/policy.py
from molmo_spaces.policy.base_policy import InferencePolicy


class MyPolicy(InferencePolicy):
    def __init__(self, config, task):
        super().__init__(config, task)
        self.camera_names = config.policy_config.camera_names
        self.action_spec = config.policy_config.action_spec
        self.prepare_model()

    def prepare_model(self):
        # Load your model from config.policy_config.checkpoint_path
        self.model = load_my_model(self.config.policy_config.checkpoint_path)

    def reset(self):
        # Called at the start of each episode
        pass

    def get_action(self, observation) -> dict[str, np.ndarray]:
        # observation is a dict with camera images and robot_state
        # Return dict mapping move group names to action arrays
        # e.g. {"arm": np.array([...]), "gripper": np.array([...])}
        obs = observation[0] if isinstance(observation, list) else observation
        images = [obs[cam] for cam in self.camera_names]
        state = obs["robot_state"]["qpos"]
        return self.model.predict(images, state)
```

See `molmo_spaces/policy/learned_policy/synthvla_policy.py` for a full example with action chunking.

### 2. Policy Config

Extend `BasePolicyConfig`. Define your model's interface.

```python
# my_repo/configs.py
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import PolicyFactory


class MyPolicyConfig(BasePolicyConfig):
    policy_type: str = "learned"
    action_type: str = "joint_pos_rel"
    policy_cls: type = None
    policy_factory: PolicyFactory | None = None

    def model_post_init(self, __context):
        if self.policy_cls is None:
            from my_repo.policy import MyPolicy

            self.policy_cls = MyPolicy
            self.policy_factory = MyPolicy

    checkpoint_path: str
    camera_names: list[str] = ["exo_camera_1", "wrist_camera"]
    action_move_group_names: list[str] = ["arm", "gripper"]
    action_spec: dict[str, int] = {"arm": 7, "gripper": 1}
```

### 3. Eval Config

Extend `JsonBenchmarkEvalConfig`. This is the minimal config for benchmark eval - episode-specific data (cameras, poses, task params) comes from the benchmark JSON.

```python
# my_repo/configs.py
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig


class MyEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: MyPolicyConfig = MyPolicyConfig(checkpoint_path="/path/to/default/checkpoint")
    policy_dt_ms: float = 200.0  # Match your model's expected control rate

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False
```

### 4. Run Evaluation

Please look at the concrete commands for each task type in our [leaderboard](https://molmospaces.allen.ai/leaderboard):
- MolmoSpaces tasks (`MS-` prefix): [ms-bench](ms-bench.md)
- MolmoBot tasks (`MB-` prefix): [mb-bench](mb-bench.md)

Command line:

```bash
python molmo_spaces/evaluation/eval_main.py \
    my_repo.configs:MyEvalConfig \
    --benchmark_dir <path/to/benchmark.json> \
    --checkpoint_path <path/to/checkpoint/directory> \
    --task_horizon_steps 500
```

Or programmatically:

```python
from pathlib import Path
from molmo_spaces.evaluation.eval_main import run_evaluation

results = run_evaluation(
    eval_config_cls="my_repo.configs:MyEvalConfig",
    benchmark_dir=Path("<path/to/benchmark.json>"),
    checkpoint_path="<path/to/checkpoint/directory>",
    task_horizon_steps=500,
    use_wandb=True,
)

print(f"Success rate: {results.success_rate:.1%}")
for r in results.episode_results:
    print(f"{r.house_id}/ep{r.episode_idx}: {'pass' if r.success else 'fail'}")
```

You can also pass `preloaded_policy=` if you've already instantiated the policy.


## Running benchmark with a custom asset

You might want to replace the target rigid object for `pick` or `pick-and-place` with a custom asset for a specific episode.

```bash
python molmo_spaces/evaluation/eval_main.py \
    molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig \
    --benchmark_dir assets/benchmarks/path-to-benchmark.json \
    --checkpoint_path <path/to/checkpoint/pi0_fast_droid_jointpos> \
    --task_horizon_steps 500
    --idx 0
    --add_custom_object
    --custom_object_path <path/to/custom/object.xml>
    --custom_object_name <natural/language/name/of/object>
```



## Sample Episode Spec

```json
{
  "source": {
    "h5_file": "/.../.../house_2115/trajectories_batch_3_of_3.h5",
    "traj_key": "traj_2",
    "episode_length": 49,
    "camera_system_class": "FrankaDroidCameraSystem",
    "source_data_date": "2025-12-19",
    "benchmark_created_date": "2026-01-21"
  },
  "house_index": 2115,
  "scene_dataset": "procthor-objaverse",
  "data_split": "val",
  "seed": null,
  "robot": {
    "robot_name": "franka_droid",
    "init_qpos": {
      "base": [],
      "arm": [-0.024, -0.737, -0.007, -2.327, -0.038, 1.590, 0.020],
      "gripper": [0.003, 0.003]
    }
  },
  "cameras": [
    {
      "name": "wrist_camera",
      "type": "robot_mounted",
      "reference_body_names": ["robot_0/gripper/base"],
      "camera_offset": [0.031, 0.074, 0.022],
      "lookat_offset": [0.0, 0.0, 0.08],
      "camera_quaternion": [-0.006, -0.001, 0.986, 0.169],
      "fov": 56.74
    },
    {
      "name": "exo_camera_1",
      "type": "robot_mounted",
      "reference_body_names": ["robot_0/fr3_link0"],
      "camera_offset": [0.1, 0.57, 0.66],
      "lookat_offset": [0.0, 0.0, 0.08],
      "camera_quaternion": [-0.363, -0.124, 0.426, 0.819],
      "fov": 71.0
    }
  ],
  "scene_modifications": {
    "added_objects": {},
    "object_poses": {
      "pillow_1c5c1394...": [6.79, 3.98, 0.84, -0.50, -0.50, 0.50, 0.50],
      "bowl_f159d8f5...": [2.03, 3.97, 0.82, 0.0, 0.0, 0.71, 0.71],
      "...": "// ~20 more objects with [x, y, z, qw, qx, qy, qz] poses"
    }
  },
  "task": {
    "task_cls": "molmo_spaces.tasks.pick_task.PickTask",
    "robot_base_pose": [1.61, 4.41, 0.07, 0.90, 0.0, 0.0, -0.45],
    "pickup_obj_name": "bowl_f159d8f5528715d01c1bddd6ef86dbcb_1_0_2",
    "pickup_obj_start_pose": [2.03, 3.97, 0.82, 0.0, 0.0, 0.71, 0.71],
    "pickup_obj_goal_pose": [2.03, 3.97, 0.87, 0.0, 0.0, 0.71, 0.71],
    "succ_pos_threshold": 0.01
  },
  "language": {
    "task_description": "Pick up a white bowl",
    "referral_expressions": {"pickup_obj_name": "white bowl"},
    "referral_expressions_priority": {
      "pickup_obj_name": [
        [0.045, 0.31, "smooth white bowl"],
        [0.041, 0.29, "white bowl"],
        [0.023, 0.25, "bowl"],
        "// ... more referrals with [clip_score, dino_score, text]"
      ]
    }
  },
}
```
