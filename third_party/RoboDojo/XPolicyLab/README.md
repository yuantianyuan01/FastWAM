<div align="center">

<h1>XPolicyLab</h1>

<p><strong>A Unified Standard and Open Ecosystem for Robot Policy Evaluation and Deployment</strong></p>

<p>
<a href="https://xpolicylab.github.io/">Website</a> |
<a href="https://arxiv.org/abs/2608.09892">arXiv</a> |
<a href="https://github.com/XPolicyLab/XPolicyLab">GitHub</a> |
<a href="https://robodojo-benchmark.com/LeaderBoard">RoboDojo Leaderboard</a> |
<a href="https://robotwin-platform.github.io/leaderboard">RoboTwin Leaderboard</a>
</p>

<img src="assets/teaser.png" alt="XPolicyLab overview" width="100%"/>

<p><em>Connecting N policies to M evaluation environments — from O(N×M) down to O(N+M).</em></p>

</div>

XPolicyLab is the shared layer between policy code and evaluation environments. Keep each model's dependencies, checkpoints, and training recipes under `policy/<POLICY>/`; XPolicyLab handles the parts that are boring but easy to get wrong — serving, observation/action contracts, and eval wiring. As of August 2026, the ecosystem integrates **42 robot policies** spanning VLA, world-action, imitation-learning, and memory-augmented families, and the same adapters serve RoboTwin, RoboDojo simulation, and standardized real-robot evaluation.

Start here for repo-level concepts and integration steps. For install commands, checkpoint layout, and training details, jump to that policy's README — it is the source of truth for its model.

## 📚 Contents

- [What XPolicyLab Enables](#-what-xpolicylab-enables)
- [Supported Benchmarks And Infrastructure](#-supported-benchmarks-and-infrastructure)
- [Integrated Policies](#-integrated-policies)
- [Framework Overview](#-framework-overview)
- [Quick Start](#-quick-start)
- [Common Workflow](#-common-workflow)
- [Deployment Flow](#-deployment-flow)
- [Standard Data Formats](#-standard-data-formats)
- [Data And Checkpoints](#-data-and-checkpoints)
- [Add Your Own Policy](#-add-your-own-policy)
- [Citation](#-citation)
- [Contact](#-contact)

## 🚀 What XPolicyLab Enables

- **Environment isolation**: run the policy model in its own conda/uv environment while the simulator, benchmark, or robot client runs separately.
- **Remote deployment**: connect the policy server and environment client through websocket, either on one machine or across machines.
- **A common adapter contract**: use the same high-level lifecycle for installation, data conversion, training, serving, and evaluation.
- **A large policy zoo**: reuse adapters for VLA/WAM policies, imitation-learning baselines, and reference templates.
- **Benchmark and infra integration**: mount XPolicyLab into benchmark or simulator workspaces without coupling policy code to one environment.

## 🌐 Supported Benchmarks And Infrastructure

XPolicyLab is benchmark-agnostic: any benchmark, simulator, or real-robot setup can plug in as an environment client against the same policy-side interface — one adapter per policy, one client per environment. Two public benchmarks are already integrated, and their official leaderboards are powered by XPolicyLab submissions.

<div align="center">
<img src="assets/benchmarks.png" alt="Cross-platform evaluation through XPolicyLab" width="70%"/>
<p><em>Cross-platform evaluation through a shared codebase and standardized serving interface.</em></p>
</div>

**Benchmarks**

- **[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)**: simulator-backed evaluation and RoboDojo-format data exports. The [RoboDojo Leaderboard](https://robodojo-benchmark.com/LeaderBoard) covers 42 simulation tasks across five capability dimensions (Generalization, Precision, Long-Horizon, Memory, Open) plus 18 real-robot tasks on three bimanual embodiments.
- **[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)**: benchmark and data source through policy-specific adapters and conversion scripts. The [RoboTwin 2.0 Leaderboard](https://robotwin-platform.github.io/leaderboard) covers bimanual manipulation across 50 tasks under clean and randomized settings.

**Infrastructure**

- **[RLinf](https://github.com/RLinf/RLinf)** *(coming soon)*: infrastructure target for policy development and deployment workflows.
- **StarVLA**: infrastructure and policy stack; see [policy/starVLA](policy/starVLA/README.md).

## 🧭 Integrated Policies

42 policies are currently integrated, spanning VLA, world-action, imitation-learning, and memory-augmented families. Top-level adapters live in `policy/`; each policy README documents that model's paper/repo link, environment, data format, training entrypoint, and checkpoint layout.

<details>
<summary>Policy catalog</summary>

**Foundation / VLA / WAM policies**

- [A1](policy/A1/README.md), [AHA-WAM](policy/AHA_WAM/README.md), [ABot-M0](policy/Abot_M0/README.md), [Being-H05](policy/Being_H05/README.md), [DM0](policy/Dexbotic_DM0/README.md), [Dexora-1B](policy/Dexora_1B/README.md)
- [DreamZero](policy/DreamZero/README.md), [EventVLA](policy/EventVLA/README.md), [FastWAM](policy/FastWAM/README.md), [G0](policy/GalaxeaVLA/README.md), [G0.5](policy/G05/README.md), [GO-1](policy/GO1/README.md), [GR00T-N1.7](policy/GR00T_N17/README.md)
- [GigaWorld-Policy](policy/GigaWorldPolicy/README.md), [H-RDT](policy/H_RDT/README.md), [Hy-Embodied-0.5-VLA](policy/Hy_Embodied_05_VLA/README.md), [InternVLA-A1](policy/InternVLA_A1/README.md), [InternVLA-A1.5](policy/InternVLA_A1_5/README.md), [LDA-1B](policy/LDA_1B/README.md)
- [LingBot-VA](policy/LingBot_VA/README.md), [LingBot-VLA](policy/LingBot_VLA/README.md), [Mem-0](policy/Mem_0/README.md), [MolmoAct2](policy/MolmoACT2/README.md)
- [OpenVLA-OFT](policy/OpenVLA_OFT/README.md), [π0](policy/Pi_0/README.md), [π0.5](policy/Pi_05/README.md), [π0-Fast](policy/Pi_0_Fast/README.md), [RDT-1B](policy/RDT_1B/README.md), [RISE](policy/RISE/README.md)
- [SmolVLA](policy/SmolVLA/README.md), [Spatial Forcing](policy/Spatial_Forcing/README.md), [Spirit v1.5](policy/Spirit_v15/README.md), [TinyVLA](policy/TinyVLA/README.md), [X-VLA](policy/X_VLA/README.md), [X-WAM](policy/X_WAM/README.md), [Xiaomi-Robotics-0](policy/Xiaomi_Robotics_0/README.md), [Xiaomi-Robotics-1 (XR-1)](policy/Xiaomi_Robotics_1/README.md), [StarVLA](policy/starVLA/README.md)

**Baselines and examples**

- [ACT](policy/ACT/README.md), [DP](policy/DP/README.md), [demo_policy](policy/demo_policy/README.md)

</details>

Adding a policy of your own, or entering a leaderboard, both go through a PR — see [Add Your Own Policy](#-add-your-own-policy).

## 🧩 Framework Overview

XPolicyLab separates model-side dependencies from environment-side dependencies, so each side retains its native stack and may run locally or remotely. One adapter serves benchmarks, simulators, and physical robots.

<div align="center">
<img src="assets/infra.png" alt="XPolicyLab infrastructure" width="100%"/>
<p><em>Infrastructure of XPolicyLab. One adapter serves benchmarks, simulators, and physical robots.</em></p>
</div>

```text
Policy environment                         Evaluation / benchmark environment
------------------                         ----------------------------------
policy/<POLICY>/model.py     <---ws--->    env client / simulator / robot
policy server                              environment client
deploy.yml runtime config                  benchmark task and observation API
```

A typical adapter contains:

```text
policy/<POLICY>/
├── README.md                    # policy-specific guide
├── INSTALLATION.md              # optional detailed setup notes
├── __init__.py                  # keeps XPolicyLab.policy.<POLICY> importable
├── install.sh                   # environment setup
├── process_data.sh              # optional data conversion
├── train.sh                     # optional training
├── eval.sh                      # same-machine evaluation
├── setup_eval_policy_server.sh  # policy-side server
├── setup_eval_env_client.sh     # environment-side client
├── deploy.yml                   # runtime config
├── deploy.py                    # evaluation loop
└── model.py                     # model adapter
```

`model.py` implements the model-facing API. `deploy.py` bridges environment observations to model-server calls. Use [policy/demo_policy](policy/demo_policy/README.md) as the minimal adapter reference.

`model.py` should define a `Model` class with this shape:

| Method | Contract |
| --- | --- |
| `__init__(model_cfg)` | Load model config, checkpoints, processors, and runtime overrides from `deploy.yml`. |
| `update_obs(obs)` | Update model state from one observation dictionary. |
| `update_obs_batch(obs_list)` | Update model state from a list of observation dictionaries. |
| `get_action()` | Return one action chunk as a list of action dictionaries. |
| `get_action_batch(env_idx_list=None)` | Return batched action chunks aligned with active environment indices. |
| `reset()` | Clear model-side state between evaluation episodes. It takes no arguments — a policy that needs a first observation should `reset()` and then take a normal `update_obs`. |

The policy server decodes camera colors before `update_obs` / `update_obs_batch`, so `obs["vision"][<camera>]["color"]` always arrives as an image array — `model.py` never decodes.

The default policy-server protocol is websocket (`protocol: ws` in `deploy.yml`); `legacy_tcp` exists only for adapters that have not migrated yet. The transport handles reconnects, retries, keepalive, and long model-loading cold starts for you — a normal adapter never touches it.

<details>
<summary>Transport details and timeout tuning (only if evaluation hangs or drops)</summary>

- **Retries are safe**: each request carries a `request_id` that the client reuses across reconnects, and the server answers duplicates from a cache instead of running a non-idempotent call twice. A `timeout` error is the exception — the server may still be running the call, so treat it as fatal for that trial rather than retrying.
- **Server restarts abort the run**: if a reconnect lands on a different server process, the client raises `ServerRestartedError`, because the fresh server lost the model state.
- **Cold start**: the server loads the model before opening its port, so an early client just retries (default budget 15 min). `eval.sh` also gates the client behind `wait_for_policy_server.sh`.
- **Errors**: the client only sees `str(exc)`; the full traceback of a model failure is logged on the *policy server* side, so look there first.
- **Serialization** is msgpack with numpy support (`torch.Tensor` auto-converts). Three quirks: `tuple` arrives as `list`, decoded numpy arrays are read-only views (copy before in-place edits), and int dict keys arrive as strings.

Optional `deploy.yml` keys — omit them to keep the defaults:

| Key | Default | Purpose |
| --- | --- | --- |
| `request_timeout_s` | `120.0` | Timeout for one `update_obs` / `get_action` call — raise it for slow inference. |
| `max_connect_attempts` | `180` | Cold-start retries while the server is still loading. |
| `connect_retry_delay_s` | `5.0` | Delay between those retries. |
| `max_connect_seconds` | `900.0` | Wall-clock cap on the whole retry loop; `0` disables it. |
| `connect_timeout_s` | `30.0` | Timeout for one connect attempt. |
| `handshake_timeout_s` | `60.0` | Timeout for the HELLO round-trip. |
| `ws_ping_interval_s` / `ws_ping_timeout_s` | `20.0` | Keepalive ping/pong; `null` disables. |
| `close_timeout_s` | `10.0` | Cap on the closing handshake. |

</details>

## ⚡ Quick Start

Clone XPolicyLab as a normal Python project for adapter development, offline checks, training from prepared data, or your own environment client:

```bash
mkdir demo_env
cd demo_env
git clone https://github.com/XPolicyLab/XPolicyLab.git
cd XPolicyLab
pip install -e .
```

You do not need a simulator to start model-side development: the bundled downloader fetches prepared RoboDojo data — several simulator export versions plus HDF5 `RoboDojo_real` real-world data — for training and offline debugging. If you use `XPolicyLab/` as a subpackage inside the RoboDojo repository, follow RoboDojo's own data download scripts instead.

Download a small Hugging Face demo bundle and keep the data next to `XPolicyLab/`:

```bash
# From demo_env/XPolicyLab
bash scripts/RoboDojo/download_robodojo_data.sh demo
```

This creates:

```text
demo_env/
├── data/        # demo data, including a small 10-episode HuggingFace bundle
└── XPolicyLab/
```

The same script pulls the full exports — `hdf5`, `lerobot_v3.0`, `lerobot_v2.1`, and `real` (real-world HDF5) — each into its own `../data/` folder.

With this setup, you can test data conversion, model loading, training scripts, and debug-mode evaluation before connecting to a simulator-backed benchmark.

```bash
export EVAL_ENV_TYPE=debug
cd policy/demo_policy
bash install.sh
bash eval.sh RoboDojo stack_bowls demo arx_x5 joint 0 0 0 base base
```

The template for any adapter is the same — swap `demo_policy` and the argument values:

```bash
export EVAL_ENV_TYPE=debug
cd policy/<POLICY>
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> \
  <seed> <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

For RoboDojo simulation, mount `XPolicyLab/` beside the simulator-side `env_cfg/`, `scripts/`, `src/eval_client/`, and `task/` directories.

## 🔄 Common Workflow

Most adapters expose the same top-level shape. Some policies add extra arguments, consume upstream-native datasets, or skip training support. Follow the policy README when it differs from this template.

```bash
cd policy/<POLICY>

# Install the policy runtime.
bash install.sh

# Optional: convert or prepare policy-specific data.
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [extra_args...]

# Optional: train.
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra_args...]

# Evaluate on one machine.
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

### What the arguments mean

When you run `eval.sh`, you are mostly answering: **which benchmark family**, **which task to run now**, **which checkpoint to load**, **which robot setup**, **joint or end-effector actions**, and **which seed**. The same names travel through `process_data.sh`, `train.sh`, and `eval.sh`, so you do not have to rename things at every step.

| Argument | In plain English | Examples |
| --- | --- | --- |
| `bench_name` | Which benchmark or dataset family this run belongs to | `RoboDojo`, `RoboTwin` |
| `task_name` | The task the environment client should run right now | `stack_bowls`, `push_T` — can differ from the tasks seen during training |
| `ckpt_name` | Which weights to load: a short run nickname, the full run folder name, or a path | `cotrain`, `RoboDojo-cotrain-arx_x5-joint-0`, `checkpoints/my_run/` |
| `env_cfg_type` | Robot / camera / scene configuration key | `arx_x5` |
| `action_type` | Action space the policy outputs | usually `joint` or `ee` |
| `seed` | Training or evaluation seed / layout id | `0`, `1`, `2` |
| `policy_gpu_id` / `env_gpu_id` | Which GPU runs the model vs. the simulator/client | `0`, `1` |
| `policy_env_or_uv_path` | Conda env name or uv env path for the policy server | your policy-side env |
| `eval_env_conda_env` | Conda env for the simulator / robot client | your eval-side env |

**How `ckpt_name` resolves.** Usually you pass the short nickname used during training, such as `cotrain`, and XPolicyLab combines it with the other args into `checkpoints/RoboDojo-cotrain-arx_x5-joint-0/`. You can also pass the full folder name, or a path — relative paths resolve from the policy directory, absolute paths work too. Some adapters honor explicit keys in `deploy.yml` (`checkpoint_path`, `model_path`, ...). When in doubt, check the policy README.

**A concrete eval example:**

```bash
cd policy/AHA_WAM
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 joint 0 0 0 aha_wam robodojo
# loads checkpoints/RoboDojo-cotrain-arx_x5-joint-0/ and evaluates on stack_bowls
```

## 🔌 Deployment Flow

During evaluation, the policy server and the environment client talk over websocket. That split is what lets you keep Isaac Sim / robot drivers on one machine and a heavy VLA on another.

For same-machine evaluation, `eval.sh` is enough — it starts the server, runs the client, and cleans up when you are done.

For split-machine deployment, start the policy server on the GPU machine and bind to `0.0.0.0` so other machines can reach it. The client connects to the policy machine's real IP, not `0.0.0.0`.

```bash
cd policy/<POLICY>
bash setup_eval_policy_server.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <policy_env_or_uv_path> <policy_server_port> 0.0.0.0
```

Then start the environment client on the simulator or robot machine:

```bash
cd policy/<POLICY>
bash setup_eval_env_client.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <env_gpu_id> <eval_env_conda_env> <additional_info> \
  <policy_server_port> <policy_server_ip>
```

`<additional_info>` is a comma-separated `key=value` string forwarded to the environment client. `eval.sh` builds it automatically as `ckpt_name=<ckpt_name>,action_type=<action_type>`, which is the right default for most adapters.

`EVAL_ENV_TYPE` selects the environment-side backend:

- unset or `sim`: real simulator-backed evaluation, when the integration is installed.
- `debug`: offline wiring check — no Isaac, no robot, just shapes and IO.
- `real`: real-robot client path, where the hardware integration exists.

## 📐 Standard Data Formats

XPolicyLab standardizes the observation and trajectory dictionaries passed between adapters, converters, and environment clients. Individual policies may convert this standard format into their upstream-native format.

All pose values use `[x, y, z, qw, qx, qy, qz]`. Images are RGB end to end — stored image bits are encoded from RGB frames, and no channel conversion happens anywhere in the pipeline. Note one naming quirk: runtime observations carry camera extrinsics as `extrinsics_matrix`, while trajectory files store `extrinsic_matrix`.

<details>
<summary>Observation Data Format</summary>

```text
Observation Data Format
├── data_format_version                        string, optional
├── instruction / instructions                 string or list[str]
├── env_idx                                    int, optional for batched eval
├── additional_info/
│   └── frequency                              int, optional
├── vision/
│   ├── cam_head/
│   │   ├── color                              (H, W, 3) RGB, decoded by the server
│   │   ├── depth                              (H, W) or (H, W, 1), optional
│   │   ├── intrinsic_matrix                   (3, 3), optional
│   │   ├── extrinsics_matrix                  (4, 4), optional
│   │   └── shape                              (2,) or (3,), optional
│   ├── cam_left_wrist/                        optional
│   ├── cam_right_wrist/                       optional
│   ├── cam_wrist/                             optional for single-arm robots
│   └── cam_third_view/                        optional
└── state/
    ├── left_arm_joint_state                   (DOF,), optional
    ├── left_ee_joint_state                    (EEF_DOF,), optional
    ├── left_ee_pose                           (7,), optional
    ├── left_tcp_pose                          (7,), optional
    ├── left_delta_ee_pose                     (7,), optional
    ├── right_arm_joint_state                  (DOF,), optional
    ├── right_ee_joint_state                   (EEF_DOF,), optional
    ├── right_ee_pose                          (7,), optional
    ├── right_tcp_pose                         (7,), optional
    ├── right_delta_ee_pose                    (7,), optional
    ├── arm_joint_state                        (DOF,), optional for single-arm robots
    ├── ee_joint_state                         (EEF_DOF,), optional for single-arm robots
    ├── ee_pose                                (7,), optional for single-arm robots
    ├── tcp_pose                               (7,), optional for single-arm robots
    ├── delta_ee_pose                          (7,), optional for single-arm robots
    └── mobile/                                optional
        ├── base_pose                          (7,)
        └── base_twist                         (6,), [vx, vy, vz, wx, wy, wz]
```

</details>

<details>
<summary>Trajectory Data Format</summary>

```text
Trajectory Data Format
├── data_format_version                        string, e.g. "v1.0"
├── instruction / instructions                 string, or JSON-serialized list[str]
├── subtasks                                   JSON-serialized annotations, optional
├── additional_info/
│   └── frequency                              int
├── vision/
│   ├── cam_head/
│   │   ├── colors                             (T, H, W, 3), uint8 RGB or encoded stream
│   │   ├── depths                             (T, H, W) or (T, H, W, 1), optional
│   │   ├── intrinsic_matrix                   (3, 3) or (T, 3, 3), optional
│   │   ├── extrinsic_matrix                   (4, 4) or (T, 4, 4), optional
│   │   └── shape                              (2,) or (3,), optional
│   ├── cam_left_wrist/                        optional
│   ├── cam_right_wrist/                       optional
│   ├── cam_wrist/                             optional for single-arm robots
│   └── cam_third_view/                        optional
├── action/                                    action targets, same key naming as state/ below
└── state/
    ├── left_arm_joint_states                  (T, DOF), optional
    ├── left_ee_joint_states                   (T, EEF_DOF), optional
    ├── left_ee_poses                          (T, 7), optional
    ├── left_tcp_poses                         (T, 7), optional
    ├── left_delta_ee_poses                    (T, 7), optional
    ├── right_arm_joint_states                 (T, DOF), optional
    ├── right_ee_joint_states                  (T, EEF_DOF), optional
    ├── right_ee_poses                         (T, 7), optional
    ├── right_tcp_poses                        (T, 7), optional
    ├── right_delta_ee_poses                   (T, 7), optional
    ├── arm_joint_states                       (T, DOF), optional for single-arm robots
    ├── ee_joint_states                        (T, EEF_DOF), optional for single-arm robots
    ├── ee_poses                               (T, 7), optional for single-arm robots
    ├── tcp_poses                              (T, 7), optional for single-arm robots
    ├── delta_ee_poses                         (T, 7), optional for single-arm robots
    └── mobile/                                optional
        ├── base_poses                         (T, 7)
        └── base_twists                        (T, 6), [vx, vy, vz, wx, wy, wz]
```

</details>

Useful converter helpers:

```python
from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info
```

`decode_image_bit` turns encoded image streams into arrays and returns already-decoded values untouched. `get_robot_action_dim_info(env_cfg_type)` returns robot-specific `arm_dim` and `ee_dim` lists, so adapters do not need to hard-code action dimensions.

Offline code — conversion scripts and training dataloaders — must decode through `decode_image_bit` and never through hand-rolled `cv2.imdecode` / `np.frombuffer` / PIL, because RoboTwin and RoboDojo store image bits in legacy layouts that only this function reads correctly. Runtime code does not decode at all; the policy server has already done it, as noted in [Framework Overview](#-framework-overview). Breaking either rule fails silently and is hard to debug.

[CONTRIBUTING.md](CONTRIBUTING.md#modelpy) states both rules in full, along with the two narrow exceptions to the RGB rule and how a new robot gets registered in both `_robot_info.json` files.

## 💾 Data And Checkpoints

Training and data prep usually name things predictably so eval can find them without guesswork:

```text
<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>
<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>
```

So if you trained with `bench_name=RoboDojo`, `ckpt_name=cotrain`, `env_cfg_type=arx_x5`, `action_type=joint`, `seed=0`, the run lands in `checkpoints/RoboDojo-cotrain-arx_x5-joint-0/`. How `ckpt_name` maps back to these folders at eval time is covered in [Common Workflow](#-common-workflow).

Policies may also use upstream-native layouts or explicit paths in `deploy.yml`. Check the policy README before assuming a naming convention. For a small local dataset to play with, see [Quick Start](#-quick-start).

## 🤝 Add Your Own Policy

Community policies are welcome — open a PR that adds `policy/<POLICY>/`. A PR is also **required** to enter the official [RoboDojo](https://robodojo-benchmark.com/LeaderBoard) and [RoboTwin](https://robotwin-platform.github.io/leaderboard) leaderboards, together with the checkpoint that reproduces your results. [CONTRIBUTING.md](CONTRIBUTING.md) is the full standard: required files, the `Model` contract, `deploy.yml` keys, script conventions, and the PR template.

The fastest route is to copy the reference adapter, keep the XPolicyLab boundary small, and debug before touching a simulator:

1. **Read [policy/demo_policy](policy/demo_policy/README.md)** — `model.py`, `deploy.py`, `deploy.yml`, and the `eval.sh` / `setup_eval_policy_server.sh` / `setup_eval_env_client.sh` trio.
2. **Scaffold** with `bash scripts/create_policy.sh <POLICY_NAME>`, then fill in its README.
3. **Implement `model.py` first**, keeping `bench_name`, `task_name`, `ckpt_name`, `env_cfg_type`, `action_type`, and `seed` consistent across data, training, and eval ([Common Workflow](#-common-workflow)).
4. **Put runtime defaults in `deploy.yml`** and keep `deploy.py` aligned with `demo_policy/deploy.py` unless the environment loop truly differs.
5. **Run the checks below**, then move to `EVAL_ENV_TYPE=sim` or a [split-machine deployment](#-deployment-flow).

Eval-only submissions are accepted when training code cannot be open-sourced yet: say so in the PR, notify the maintainers ([Contact](#-contact)), and share a timeline. For leaderboard evaluation, attach a checkpoint download script (Hugging Face or ModelScope preferred).

### Checks before a PR

Static checks from the repo root, then the adapter wiring check from `policy/<POLICY>/` — no simulator required:

```bash
git diff --check
bash -n policy/<POLICY>/*.sh
python -m py_compile policy/<POLICY>/model.py policy/<POLICY>/deploy.py
```

```bash
cd policy/<POLICY>
export EVAL_ENV_TYPE=debug
bash eval.sh RoboDojo stack_bowls demo arx_x5 joint 0 0 0 \
  <policy_env_or_uv_path> <eval_env_conda_env>
```

This verifies imports, server startup, observation serialization, action keys, action dimensions, and batch logic. The debug client sends plain image arrays by default; re-run with `DEBUG_OBS_ENCODED=1` to make it send encoded camera colors instead — a JPEG buffer, raw bytes, and a plain array across the three cameras — which exercises the server-side decode path that real environment clients rely on. For a quick smoke test, `policy/demo_policy` accepts placeholder env names such as `base`.

<details>
<summary>Using a coding agent</summary>

This repo ships two Agent Skills under [.agents/skills](.agents/skills), which `.cursor/skills` and `.claude/skills` symlink to, so Cursor, Claude Code and Codex all pick them up automatically: `xpolicylab-model-integration` builds an adapter (a prompt like "Integrate <POLICY_NAME> into XPolicyLab" is enough), and `xpolicylab-adapter-check` audits one against [CONTRIBUTING.md](CONTRIBUTING.md) before a PR ("Check policy/<POLICY_NAME>"). [AGENTS.md](AGENTS.md) carries the always-on rules every agent must follow. For an agent that supports none of these, paste this checklist:

```text
Integrate <POLICY_NAME> into XPolicyLab.

Use policy/demo_policy as the reference.
1. Inspect the upstream model's inference API and dependencies.
2. Create or update policy/<POLICY_NAME>/README.md with install, checkpoint, train, and eval commands.
3. Implement install.sh and, if needed, process_data.sh and train.sh.
4. Implement model.py with Model.__init__, update_obs, get_action, reset, and batch methods.
5. Keep deploy.py aligned with policy/demo_policy/deploy.py.
6. Put runtime defaults in deploy.yml, keeping the standard key set (protocol: ws, host, port, ...).
7. Run EVAL_ENV_TYPE=debug eval.sh and fix shape/action-key/server errors.
8. Summarize supported action_type, env_cfg_type, checkpoint layout, and remaining limitations.
```

</details>

## 📝 Citation

If XPolicyLab helps your research, please cite:

```bibtex
@misc{community2026xpolicylabunifiedstandardopen,
      title={XPolicyLab: A Unified Standard and Open Ecosystem for Robot Policy Evaluation and Deployment}, 
      author={XPolicyLab Community and Tianxing Chen and Yue Chen and Tian Nian and Zijian Cai and Guangyu Chen and Wenwei Lin and Qiwei Liang and Peicheng Xiang and Kailun Su and Zixuan Li and Junyuan Tang and Yan Qin and Qiangyu Chen and Shaolong Zhu and Xiang Li and Jiahao Zhang and Weijie Wan and Baijun Chen and Honghao Su and Kehe Ye and Shujia Liu and Kaixuan Wang and Haotian Liang and Yunze Liu and Mingleyang Li and Yuran Wang and Boyu Chen and Hongzhe Bi and Shuhe Huang and Hengkai Tan and Jisong Cai and Yao Mu and Jun Guo and Xiaofeng Wang and Zheng Zhu and Weijie Ke and Hengtao Li and Yuhang Tang and Xiaofan Li and Ganlin Yang and Zhangzheng Tu and Shuai Yang and Wenxuan Song and Pengxiang Ding and Kaidong Zhang and Yu Sun and Junliang Guo and Tong Zhang and Yixing Chen and Rongxu Cui and Zongzheng Zhang and Haoxiang Ma and Junhao Cai and Haoyu Zhang and Senqiao Yang and Jinhui Ye and Pengguang Chen and Shu Liu and Xiu Su and Wenhan Fang and Wenhao Li and Yichao Cao and Chengyao Wang and Qiang Chen and Ping Luo and Wenbo Ding},
      year={2026},
      eprint={2608.09892},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2608.09892}, 
}
```

## 📬 Contact

Tianxing Chen (project lead): [chentianxing2002@gmail.com](mailto:chentianxing2002@gmail.com)

A collaborative open-source project led by **MMLab@HKU** and **THU**.

**Core Lead Authors**: Tianxing Chen, Yue Chen, Tian Nian, Zijian Cai, Guangyu Chen, Wenwei Lin, Qiwei Liang.

The full contributor list — spanning every integrated policy — lives on the [project website](https://xpolicylab.github.io/).
