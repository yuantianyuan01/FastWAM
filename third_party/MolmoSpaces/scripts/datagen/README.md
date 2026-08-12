# Data Generation

Data generation system for MolmoBot using MuJoCo. Generates robot manipulation trajectory datasets with configurable tasks, robots, cameras, and scene randomization.

## Quick Start

From the repository root:

```bash
export PYTHONPATH="${PYTHONPATH}:."
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Run with a registered config name
python -m molmo_spaces.data_generation.main DoorOpeningDebugConfig
```

You can also specify the module explicitly to skip auto-importing all configs:

```bash
python -m molmo_spaces.data_generation.main \
  molmo_spaces.data_generation.config.door_opening_configs:DoorOpeningDataGenConfig
```

## Available Configs

Configs are auto-discovered via the `@register_config` decorator. Some commonly used ones:

| Config Name | Description |
|---|---|
| `DoorOpeningDataGenConfig` | Door opening task |
| `DoorOpeningDebugConfig` | Door opening (debug/small scale) |
| `FrankaPickDroidDataGenConfig` | Franka pick with DROID cameras |
| `FrankaPickRandomizedDataGenConfig` | Franka pick with randomized cameras |
| `FrankaPickAndPlaceDroidDataGenConfig` | Franka pick-and-place with DROID cameras |
| `FrankaPickAndPlaceEasyDataGenConfig` | Franka pick-and-place (easier camera setup) |
| `RUMPickDataGenConfig` | RUM robot pick task |

To list all registered configs programmatically:

```python
from molmo_spaces.data_generation.main import auto_import_configs
from molmo_spaces.data_generation.config_registry import list_available_configs

auto_import_configs()
print(list_available_configs())
```

Config files live in `molmo_spaces/data_generation/config/` and can be browsed directly:

- `object_manipulation_datagen_configs.py` — Pick, place, open, close tasks for Franka, RUM, and RB-Y1
- `door_opening_configs.py` — Door opening/debug configs
- `benchmarks_datagen_configs.py` — Re-exports for benchmark pickle compatibility

### Navigation

Whereas we focus on generation of manipulation data, we also provide a basic navigation example (`NavToObjDataGenConfig` in `nav_to_obj_configs.py`) that generates RBY1 navigate-to-object trajectories using an A*-based planner. It can be run in the same way:

```bash
python -m molmo_spaces.data_generation.main NavToObjDataGenConfig
```

## Creating a New Config

1. Create a new file (or add to an existing one) under `molmo_spaces/data_generation/config/`.
2. Decorate your config class with `@register_config`:

```python
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig

@register_config("MyCustomDataGenConfig")
class MyCustomDataGenConfig(MlSpacesExpConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_workers = 4
        self.task_sampler_config = ...
        self.policy_config = ...
```

3. Run it:

```bash
python -m molmo_spaces.data_generation.main MyCustomDataGenConfig
```

## Output Format

Output is organized by house under the config's `output_dir`:

```
<output_dir>/<ConfigName>/<timestamp>/
├── running_log.log
├── config.json
├── house_0/
│   ├── trajectories_batch_1_of_1.h5
│   ├── episode_00000000_exo_camera_1_batch_1_of_1.mp4
│   ├── episode_00000001_exo_camera_2_batch_1_of_1.mp4
│   └── ...
├── house_1/
│   └── ...
└── house_N/
    └── ...
```

Each house directory contains an HDF5 file with trajectory data and MP4 videos from each camera for every episode.

## Utility Scripts

Other useful scripts in `scripts/datagen/`:

| Script | Purpose |
|---|---|
| `print_configs.py` | Print available config details |
| `compare_configs.py` | Diff two config files |
| `combine_trajs_into_h5.py` | Merge multiple trajectory files |
| `upload_videos_to_wandb.py` | Upload episode videos to WandB |
| `fetch_assets.py` | Download required simulation assets |

## Tips

1. **Start small** — use a debug config (e.g. `DoorOpeningDebugConfig`) before scaling up.
2. **Check outputs** — inspect a few houses to confirm trajectories and videos look correct.
4. **WandB logging** — configs can enable WandB tracking; set `use_wandb=True` and configure `wandb_project` in your config.
