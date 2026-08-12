<div align="center">
  <h1>
  <img src="docs/images/MolmoSpacesLogo.png" alt="MolmoSpaces Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/></br>
  A Large-Scale Open Ecosystem for Robot Manipulation and Navigation
  <div align="center">
    <a href="https://arxiv.org/pdf/2602.11337" target="_blank" rel="noopener noreferrer"><img alt="Paper" src="./docs/images/button_paper.svg"/></a>&nbsp;&nbsp;<a href="https://huggingface.co/datasets/allenai/molmospaces" target="_blank" rel="noopener noreferrer"><img alt="Data" src="./docs/images/button_data.svg"/></a>&nbsp;&nbsp;<a href="https://allenai.github.io/molmospaces/" target="_blank" rel="noopener noreferrer"><img alt="Docs" src="./docs/images/button_docs.svg"/></a>&nbsp;&nbsp;<a href="https://molmospaces.allen.ai/" target="_blank" rel="noopener noreferrer"><img alt="Demo" src="./docs/images/button_demo.svg"/></a>&nbsp;&nbsp;<a href="https://molmospaces.allen.ai/leaderboard" target="_blank" rel="noopener noreferrer"><img alt="Leaderboard" src="./docs/images/button_leaderboard.svg"/></a>
  </div>
  </br>
  &</br>
  <img src="docs/images/MolmoBotLogo.png" alt="MolmoSpaces Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/></br>
  Large-Scale Simulation Enables Zero-Shot Manipulation
  <div align="center">
    <a href="https://allenai.github.io/MolmoBot" target="_blank" rel="noopener noreferrer"><img alt="Paper" src="./docs/images/button_website.svg"/></a>&nbsp;&nbsp;<a href="https://github.com/allenai/MolmoBot" target="_blank" rel="noopener noreferrer"><img alt="Paper" src="./docs/images/button_code_models.svg"/></a>&nbsp;&nbsp;<a href="https://huggingface.co/collections/allenai/molmobot-models" target="_blank" rel="noopener noreferrer"><img alt="Data" src="./docs/images/button_data_models.svg"/></a>&nbsp;&nbsp;<a href="https://huggingface.co/datasets/allenai/MolmoBot-Data" target="_blank" rel="noopener noreferrer"><img alt="Data" src="./docs/images/button_data.svg"/></a>
  </div>
  </h1>
</div>

</br>
<br/>

<div align="center">
  <img src="docs/images/Multi_Simulator_Pan.jpg" alt="Multi-Simulator-Pan" width="1200" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <p>Assets from MolmoSpaces are usable in MuJoCo, Isaac, and ManiSkill.
  <br>
</div>


---
### Updates
- **[2026/06/22]** 🔥 [**awesome-molmospaces-papers**](docs/awesome-molmospaces-papers.md) a list of MolmoSpaces projects.
- **[2026/06/16]** 🔥 [**MolmoSpaces Policy Zoo**](https://github.com/allenai/molmospaces_policy_zoo) is a repository containing standalone third party policy implementations. If you use MolmoSpaces to make a policy (planner-based, learning-based, etc.) please contribute!
- **[2026/06/12]** 🔥 [**MolmoSpaces v0.2.0**](https://github.com/allenai/molmospaces/releases/tag/v0.2.0) is out, with significantly better usability, included tutorials, and more! Check out the changelog for more information.
- **[2026/03/24]** 🔥 [**MolmoBot-Datagen**](https://allenai.org/blog/molmobot-robot-manipulation) Code for scripted planners, data generation, and benchmark creation.
- **[2026/02/27]** 🔥 [**Leaderboards**](https://molmospaces.allen.ai/leaderboard) are out.
- **[2026/02/11]** 🔥 [**Datasets**](docs/assets.md#assets-and-resource-manager) for assets and scenes in MJCF and USDa format.
- **[2026/02/11]** 🔥 [**Benchmark**](molmo_spaces/evaluation/README.md) for 8 tasks, including *pick*, *open*, and *close* tasks in JSONs.
- **[2026/02/11]** 🔥 **MolmoSpaces** Code for scene conversion, grasp generation, teleoperation, and benchmark evaluation.


## Installation

Installing `molmospaces` is easy!

First, clone the project.

```bash
git clone git@github.com:allenai/molmospaces.git
cd molmospaces
```

Then, set up the virtual environment and install.

> Note: If you want to use the debug viewer on macOS you need to use conda or a Homebrew Python. This is because `mjpython` used by the debug viewer requires a shared `libpython3.11.dylib`, which `uv`'s standalone CPython does not ship.


With conda:

```bash
conda create -n mlspaces python=3.11
conda activate mlspaces
pip install -e ".[mujoco]"
```

Or with `uv`:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[mujoco]"
```

One of the following options must be provided:
- `mujoco` to use the classic MuJoCo renderer
- `mujoco-filament` to use the improved Filament renderer for MuJoCo

The optional installation options are:
- `dev` installs dependencies for code development.
- `grasp` installs dependencies for the grasp generation pipeline.
- `housegen` installs dependencies for the house generation pipeline from iTHOR, ProcTHOR, or Holodeck JSONs.
- `curobo` installs CuRobo for GPU-accelerated planning.

You may wish to specify some [environment variables](#environment-variables) to configure behavior.
Currently `molmospaces` supports Linux and Mac.

We provide simulation assets for Mujoco, Isaac, and ManiSkill.
Data generation and benchmarking are only supported for Mujoco.


### Installing the Filament renderer (optional)

If using `uv`, simply run:

```bash
uv pip install -e .[mujoco-filament]
```

Otherwise, first install `mujoco-filament` before installing this project:

```bash
pip install -i https://test.pypi.org/simple/ mujoco-filament
pip install -e .[mujoco-filament]
```

### Installing cuRobo (optional, used only for RB-Y1 tasks)

For cuRobo support, inside your conda environment, install with:

```bash
# Install CUDA toolkit and build tools (conda-forge for toolkit, nvidia channel for headers)
conda install -c conda-forge cuda-toolkit=12.8 ninja evdev cuda-nvcc cuda-cudart-dev -n mlspaces

# Install torch with CUDA 12.8 support BEFORE installing cuRobo (Ignore warnings after this step)
pip install "torch~=2.7.0" "torchvision>=0.22.0,<0.23.0" --index-url https://download.pytorch.org/whl/cu128

# Then compile and install the project against the installed torch
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$(dirname $(find $CONDA_PREFIX -name "cuda_runtime_api.h" | head -1)):$CPATH
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

pip install -e ".[mujoco,curobo]"
```


### Set Environment Variables (Optional)

You may wish to specify some environment variables to configure behavior.
Environment variables beginning with the `MLSPACES` prefix can be used to customize MolmoSpaces behavior.

| Environment Variable | Effect | Default |
|---|---|---|
| `MLSPACES_ASSETS_DIR` | Where to place downloaded assets | `~/.cache/molmospaces/assets/<install-hash>` |
| `MLSPACES_FORCE_INSTALL` | Override existing assets | `True` |
| `MLSPACES_PINNED_ASSETS_FILE` | A `.json` file containing pinned versions for each asset, used to override the versions specified in [molmo_spaces_constants.py](molmo_spaces/molmo_spaces_constants.py). |  |
| `MUJOCO_EGL_DEVICE_ID` | The rendering device; indices do not always match `CUDA_VISIBLE_DEVICES`. See [here](https://github.com/allenai/molmospaces/issues/66) for details. | `0`|


### Quick Test

Run a quick sample of data generation. For machines with a display, use the `--viewer` option to launch the passive debug viewer (push "w" for wire-frame view to see the robot more easily; more details [here](#mujoco-viewer-tips)). Assets should be downloaded automatically for all runs.

```bash
# Linux
python scripts/datagen/run_pipeline.py --viewer --seed 1
# Mac
mjpython scripts/datagen/run_pipeline.py --viewer --seed 1
```

The MolmoSpaces codebase has three entry points for data generation, evaluation, and debugging. The two initial entry points make use of experiment configs to configure runs. The third is more easily modifiable, with some logic for constructing runs on the fly; however, constructing experiments is complicated, and not all permutations have been tested fully.

```bash
molmo_spaces/evaluation/eval_main.py  # evaluation
molmo_spaces/data_generation/main.py  # data generation
scripts/datagen/run_pipeline.py       # debugging
```

This readme contains more information on [experiment configs](#experiment-configs) as well as the other entry points; for those, please see the [evaluation](#benchmarks-and-evaluations) and [data generation](#data-generation) sections of this readme.

## MolmoSpaces Assets

Molmospaces provides scenes, objects, robots, and benchmarks. These can be downloaded using an asset manager to automatically fetch and version-control asset dependencies. A number of assets are provided; this overview explains the naming of the assets in code:

| Type | Code Name            | Paper Name   | Description                                  | Size  |
|---|----------------------|--------------|----------------------------------------------|-------|
| objects| thor                 |              | hand-crafted indoor assets                   | ~2k   |
| objects| objaverse            |              | converted Objaverse assets                   | ~129k |
| scenes | ithor                | MSCrafted    | hand-crafted, many articulated assets        | 120   |
| scenes | procthor-10k         | MSProc       | procedurally generated with THOR assets      | ~120k |
| scenes | procthor-objaverse   | MSProcObja   | procedurally generated with Objaverse assets | ~110k |
| scenes | holodeck             | MSMultiType  | LLM generated with Objaverse assets          | ~110k |
| benchmark| molmospaces_bench_v1 | MS-Bench v1 | base benchmark for atomic tasks              |       |
| benchmark| molmospaces_bench_v2 | MS-Bench v2 | extended benchmark for atomic tasks          |       |


Please refer to [here](./docs/assets.md) for instructions to set up data directories, but you shouldn't need to manually manage any dependencies beyond setting the appropriate environment variables. If you are interested only in data generation and evaluation using MuJoCo, you can skip the rest of this section.

## Documentation

The documentation for MolmoSpaces can be found [here](https://allenai.github.io/molmospaces/).

To see and easily run additional policies in MolmoSpaces, check out the [policy zoo](https://github.com/allenai/molmospaces_policy_zoo/)!

For a list of projects using MolmoSpaces look [here](docs/awesome-molmospaces-papers.md).

Additional documentation for using assets and benchmarks in other simulators are listed below:

| Simulator | Documentation                                                                 |
|---|-------------------------------------------------------------------------------|
| MuJoCo | [MuJoCo Assets Quick Start Instructions](docs/assets.md#mujoco-assets)        |
| ManiSkill | [ManiSkill Assets Quick Start Instructions](molmo_spaces_maniskill/README.md) |
| Isaac-Sim | [Isaac-Sim Assets Quick Start Instructions](molmo_spaces_isaac/README.md)     |
| Isaac Lab-Arena | [Isaac Lab-Arena Support](https://github.com/AravindhShan-nv/molmospaces/tree/codex/isaac-arena-policy-parity-progress/molmo_spaces_isaac#isaac-lab-arena-molmospaces-pick-demo) (by NVIDIA, beta version)|

## Experiment Configs

In MolmoSpaces all runs, whether for data generation or evaluation of policies, are defined by experiment configs.
The base experiment config class is called `MlSpacesExpConfig` and is located in `molmo_spaces/configs/abstract_exp_config.py`, it contains documentation on configuring experiments.

To see a list of all currently defined experiment configs, run this:
```python
from molmo_spaces.data_generation.main import auto_import_configs
from molmo_spaces.data_generation.config_registry import list_available_configs

auto_import_configs()
print(list_available_configs())
```

## Benchmarks and Evaluations

Currently, installing and running the benchmark is only supported in the MuJoCo simulator.

### Installing Benchmarks

```bash
export MLSPACES_ASSETS_DIR=/path/to/symlink/resources
python -m molmo_spaces.molmo_spaces_constants
```

### Running Benchmarks

```bash
python molmo_spaces/evaluation/eval_main.py \
    molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig \
    --benchmark_dir assets/bench/path-to-benchmark.json \
    --checkpoint_path <path/to/checkpoint/pi0_fast_droid_jointpos> \
    --task_horizon_steps 500  # optional (defaults to benchmark value)
```

For more information, please refer to an instruction in the [benchmark](molmo_spaces/evaluation/README.md).


## Data Generation

Our data generation system makes use of predefined experiment configs that specify scenes, robots, tasks, and more.
Example experiment configs can be found in, e.g., `molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py`

```bash
python molmo_spaces/data_generation/main.py FrankaPickOmniCamConfig
```


## Teleop Input

To control a robot via phone-based teleoperation, do the following (only iPhones supported).

1. Install TeleDex from the App Store; see [here](https://apps.apple.com/us/app/teledex/id6612039501).
2. Run the datagen pipeline with the teleop policy
   ```bash
   python molmo_spaces/evaluation/eval_main.py \
    molmo_spaces.evaluation.configs.evaluation_configs:TeleopPolicyEvalConfig \
    --benchmark_dir assets/bench/path-to-benchmark.json \
    --task_horizon_steps 1000
    ```
3. Scan the QR code that shows up using the app (or manually enter the IP port). Example terminal output:
   ```bash
   TeleDex Session Starting on port 8888...
   Session Started. Details:
   IP Address: xxx.xxx.xx.xxx
   Port: 8888
   Waiting for a device to connect...
   ```
4. Start teleoperating!

- Click the toggle to grasp.
- Click the button to go to the next episode.


## Related Repositories:

The repositories related to this project can be found here:

| Repository | Purpose |
|---|---|
| [ai2_robot_infra](https://github.com/allenai/ai2_robot_infra) | Real robot infrastructure and utilities for experiments |
| [MolmoBot](https://github.com/allenai/MolmoBot) | MolmoBot policy code |
| [curobo](https://github.com/allenai/curobo) | Ai2 cuRobo branch |


## Development

### Code Formatting

Before committing, ensure your code is formatted:
```bash
ruff format .
```

### Unit Testing

We use pytest for integration testing.

```bash
PYTHONPATH=. pytest mlspaces_tests/data_generation
PYTHONPATH=. pytest mlspaces_tests/data_generation_curobo  # run tests that require curobo
```

> [!TIP]
> To debug failing tests, use `--log-cli-level DEBUG`

For setting up self-hosted CI runners or building Docker images for Beaker, see **[beaker_scripts/RUNNER_SETUP.md](beaker_scripts/RUNNER_SETUP.md)**.


### Use with Cursor/VSCode

Generating type stubs for mujoco and open3d and saving them in the `typings` folder
```bash
pybind11-stubgen mujoco -o ./typings/
```

### Mujoco Viewer Tips
1. Documentation for the viewer can be found [here](https://mujoco.readthedocs.io/en/stable/programming/samples.html#sasimulate) there are many keyboard shortcuts.
2. If you have red boxes on top of your objects, go to the left panel and toggle `Group Enable > Site groups >  Site 0`
3. Interact with objects by double-clicking > Ctrl + right mouse drag. (only with active viewers, not passive ones)


## Robot Conventions

Robot base conventions: +x=forward, +y=left, +z=up

Robot parallel-jaw gripper conventions: +z=forward, fingers open along the y axis

<img src="docs/images/robot_axis_conventions.png" width="480px">



## License

The codebase is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).
The public MolmoSpaces data endpoint is available [here](https://pub-3555e9bb2d304fab9c6c79819e48aa40.r2.dev). The public MolmoSpaces Isaac data endpoint is available [here](https://pub-96496c3574b24d0c98b235219711d359.r2.dev). Both datasets are also available for download on [HuggingFace](https://huggingface.co/datasets/allenai/molmospaces). The Objaverse subsets in these buckets are licensed under [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/). All other data subsets are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en).
The artifacts are intended for research and educational use in accordance with [Ai2's Responsible Use Guidelines](https://allenai.org/responsible-use).

## Data Attributions

The XML files have been modified from the original versions provided by the following sources:
- [mujoco_menagerie / franka_fr3](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3) - Developed by Franka Robotics
- [mujoco_menagerie / robotiq_2f85_v4](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85_v4) - Copyright (c) 2013, ROS-Industrial
- [Rainbow Robotics / rby1-sdk](https://github.com/RainbowRobotics/rby1-sdk) - Copyright 2024-2025 Rainbow Robotics
- [RUM Gripper](https://github.com/jeffacce/cap-policy) - Copyright (c) 2026 NYU Generalizable Robotics and AI Lab (GRAIL)
- [I2RT Robotics / i2rt Python API](https://github.com/i2rt-robotics/i2rt) - Copyright (c) I2RT Robotics
- [mujoco_menagerie / unitree_g1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) - Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")

## Citing

```
@article{kim2026molmospaces,
  title={MolmoSpaces: A Large-Scale Open Ecosystem for Robot Navigation and Manipulation},
  author={Kim, Yejin and Pumacay, Wilbert and Rayyan, Omar and Argus, Max and Han, Winson and VanderBilt, Eli and Salvador, Jordi and Deshpande, Abhay and Hendrix, Rose and Jauhri, Snehal and others},
  journal={arXiv preprint arXiv:2602.11337},
  year={2026}
}

@misc{deshpande2026molmobot,
      title={MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation},
      author={Abhay Deshpande and Maya Guru and Rose Hendrix and Snehal Jauhri and Ainaz Eftekhar and Rohun Tripathi and Max Argus and Jordi Salvador and Haoquan Fang and Matthew Wallingford and Wilbert Pumacay and Yejin Kim and Quinn Pfeifer and Ying-Chun Lee and Piper Wolters and Omar Rayyan and Mingtong Zhang and Jiafei Duan and Karen Farley and Winson Han and Eli Vanderbilt and Dieter Fox and Ali Farhadi and Georgia Chalvatzaki and Dhruv Shah and Ranjay Krishna},
      year={2026},
      eprint={2603.16861},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.16861},
}
```
