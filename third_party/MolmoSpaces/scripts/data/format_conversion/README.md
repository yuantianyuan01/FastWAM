# Conversion from MolmoSpaces → GR00T LeRobot (v2) data

This converts pre-recorded data from the MolmoSpaces h5 data format to the LeRobot formation.

**Note:** This script currently only works with Franka-DROID robots, not RB-Y1.

```bash
pip install "lerobot==0.3.3" h5py decord pillow scipy tqdm zstandard datasets huggingface_hub
```

Output schema:

| Feature | Shape |
|---|---|
| `observation.images.wrist_left` | (180, 320, 3) |
| `observation.images.exterior_1_left` | (180, 320, 3) |
| `observation.images.exterior_2_left` | (180, 320, 3) |
| `observation.state` | (17,) |
| `action` | (17,) |
| `observation.state.eef_9d` | (9,) |
| `observation.state.gripper_position` | (1,) |
| `observation.state.joint_position` | (7,) |
| `action.eef_9d` | (9,) |
| `action.gripper_position` | (1,) |
| `action.joint_position` | (7,) |
| `task` | — |

## 1. Example download a small sample of the pick task

```bash
python scripts/bulk_download_mb_data.py --config FrankaPickOmniCamConfig \
  --split train --part 0 --max_part_shards 1 -y ./mbdata
```

## 2. Postprocess

```bash
DATA=./mbdata/FrankaPickOmniCamConfig/part0/train
python scripts/validate_trajectories.py "$DATA"
```

**3. convert**

```bash
python scripts/format_conversion/mlspaces_to_lerobot.py "$DATA" \
  --repo-id local/mbdroid_franka_pick \
  --root ./lerobot_out \
  --max-episodes 10
```

**4. try running the notebook. also:**

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/mbdroid_franka_pick", root="./lerobot_out")
sample = ds[0]
# sample["observation.state"]             -> torch.float32 (17,)
# sample["action"]                        -> torch.float32 (17,)
# sample["observation.images.wrist_left"] -> torch.float32 (3, 180, 320)
# sample["task"]                          -> str
```
