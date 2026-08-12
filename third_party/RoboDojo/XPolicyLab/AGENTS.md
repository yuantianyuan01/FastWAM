# XPolicyLab Agent Guide

XPolicyLab wraps each robot policy as a self-contained adapter under `policy/<POLICY>/`. The policy
server imports it as `XPolicyLab.policy.<POLICY>.model`, so this checkout is always used as a
package inside a parent workspace — never as the top-level project.

- Submission standard: [CONTRIBUTING.md](CONTRIBUTING.md). Reference adapter: `policy/demo_policy/`.
  Data formats: [README](README.md#-standard-data-formats).
- Two skills carry the end-to-end workflows — `xpolicylab-model-integration` (build an adapter) and
  `xpolicylab-adapter-check` (audit one before a PR). They live in `.agents/skills/`, which
  `.cursor/skills` and `.claude/skills` symlink to, so Cursor, Claude Code and Codex all load them.

The rules below apply to every change in this repo. Their rationale is in CONTRIBUTING.md.

## Images are RGB from end to end

`decode_image_bit` and `decode_obs_images` return RGB, and the policy server hands `update_obs` /
`update_obs_batch` RGB. Treat this as settled: do not re-derive it from the usual "OpenCV returns
BGR" rule, which does not apply, because XPolicyLab buffers are encoded from RGB arrays and
`cv2.imencode` / `cv2.imdecode` carry channels through JPEG in the order they were given. A
`COLOR_BGR2RGB` added to "fix" a decode is always a bug: it trains on BGR and evaluates on RGB.

No channel conversion belongs in conversion, training, or eval code. Two exceptions only:

- Medium adapters: `COLOR_RGB2BGR` immediately before `cv2.VideoWriter.write(...)`, and
  `COLOR_BGR2RGB` immediately after `cv2.VideoCapture.read()`.
- A deliberate RGB→BGR conversion for a checkpoint trained on BGR data, opt-in through a documented
  `deploy.yml` key that defaults to RGB (see `input_color_order` in `policy/Dexora_1B`).

## Decoding goes through the shared helpers

`model.py` never decodes. The server decodes every observation it forwards, including for custom
RPCs, so `obs["vision"][<camera>]["color"]` is already an array; adapters only reshape, cast, resize.

Offline code — conversion scripts and training dataloaders — decodes only with `decode_image_bit`
from `XPolicyLab.utils.process_data`. Never hand-roll `cv2.imdecode` / `np.frombuffer` / PIL
decoding: RoboTwin and RoboDojo legacy image-bit layouts are only handled correctly by that
function. Mechanically, `cv2.imdecode` must not appear outside `utils/process_data.py`.

## Paths and dimensions come from the shared helpers

`env_cfg/` lives in the **parent workspace, outside this repo** — `XPolicyLab/env_cfg` does not
exist, so an adapter must never build that path itself.

- **Importable root** in `policy/<POLICY>/model.py` is `Path(__file__).resolve().parents[2]`, the
  parent of the checkout. `parents[1]` is the checkout itself and `parents[3]` is unrelated; both
  are bugs.
- **Checkpoints** resolve through `XPolicyLab.utils.checkpoint_resolver` — `resolve_checkpoint_root`,
  or `build_run_dir_name` / `candidate_checkpoint_roots` when an adapter adds its own naming layer —
  never by re-deriving `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`.
- **Action dimensions** come from `get_robot_action_dim_info(env_cfg_type)` in
  `XPolicyLab.utils.process_data`, never hard-coded and never through a private re-implementation.

A new robot must be registered in **both** robot-info files, or training and evaluation disagree
about action dimensions:

| File | Read by | Keyed by |
| --- | --- | --- |
| `<parent>/env_cfg/robot/_robot_info.json` | `get_robot_action_dim_info()` and `get_action_dim()` in `XPolicyLab.utils.process_data` — runtime and offline conversion | robot name, from `config.robot` in `<parent>/env_cfg/<env_cfg_type>.yml` |
| `utils/robot/_robot_info.json` | `utils/get_action_dim.sh`, called by `train.sh` | `env_cfg_type` |

The two entry points named `get_action_dim` do **not** read the same file. Runtime and conversion
code imports the Python one; the training path — `train.sh`, or a Python training entry it invokes —
uses `utils/get_action_dim.sh`, which delivers the value as a shell variable, runs on stdlib `json`
alone, and works in a standalone checkout that has no outer `env_cfg/` tree.

## deploy.yml

`policy_name` must equal the directory name. Keep the full key set from
`policy/demo_policy/deploy.yml` — including `protocol: ws`, `host` and `port` — even where the
scripts have a default; per-run fields are overridden at launch.
