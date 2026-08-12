# RoboDojo — Claude Code Guide

## Project Overview

RoboDojo is a unified sim-and-real robotics benchmark built on NVIDIA Isaac Sim / IsaacLab. Policy evaluation runs via XPolicyLab and the shared simulation stack.

Contributors maintain **Tasks** (scene + reward logic), shared **env** / **env_cfg** configs, and eval infrastructure under **scripts/** and **src/eval_client/**.

Human-facing docs: https://robodojo-benchmark.com/doc/

## Repository Layout

```
task/RoboDojo/
  config/<task_name>.yml     task scene/object config
  tasks/<task_name>.py       task logic (inherits TaskEnv)
  task_registry.py           dynamic task loader
  demos/                     non-inventory demos (e.g. dlc)
env/                         TaskEnv backbone and managers
env_cfg/                     robot / scene / sim / camera YAML
src/eval_client/             eval client (main.py, eval_env.py)
utils/                       paths, save/load, pipeline helpers
scripts/                     robodojo.sh, install.sh, init_assets.sh, internal/
docker/                      container eval (see docker/README.md)
XPolicyLab/                  policy server + deploy (submodule)
third_party/                 IsaacLab, curobo (submodules)
Assets/                      robot/object assets (scripts/init_assets.sh; not in git)
eval_result/                 runtime eval output
```

## Submodules & Ownership

| Own in RoboDojo | Own in XPolicyLab |
| :-- | :-- |
| `env/`, `env_cfg/`, `task/RoboDojo/`, `src/eval_client/` | Policy code, checkpoints, `deploy.yml`, policy server |
| `scripts/robodojo.sh`, `scripts/eval_policy.sh` | `XPolicyLab/policy/<POLICY>/eval.sh`, `setup_eval_*` scripts |
| `utils/`, install/assets scripts | Policy-specific dependencies and training |

- Submodule URL: `https://github.com/XPolicyLab/XPolicyLab.git` (branch `main`).
- Update the gitlink pin intentionally; do not edit submodule contents unless updating the pin.
- RoboDojo eval client uses `XPolicyLab/client_server/ws/model_client.py` (WebSocket transport).

## Eval Commands & Flow

**CLI** (`bash scripts/robodojo.sh <command>`):

| Command | Purpose |
| :-- | :-- |
| `doctor` | Check assets, configs, env before eval |
| `eval` | One task: policy server + sim client on localhost |
| `server` | Policy server only (split / multi-machine eval) |
| `client` | Sim client only (connect to running server) |
| `smoke` | Selected/all tasks, `EVAL_NUM=1` by default |
| `benchmark` | Full sequential sweep (`--eval-num native` or numeric) |
| `summarize` | Aggregate `eval_result/` into markdown table |
| `tasks` | List runnable tasks (maintainer inventory) |

**Single-machine eval:**

```
robodojo.sh eval
  → scripts/internal/run_policy_eval.sh
    → (cd policy_dir) setup_eval_policy_server.sh   # policy server
    → setup_eval_env_client.sh                    # sim client → src/eval_client/main.py
```

- Policy `eval.sh` must exist under `--policy-dir` (used for validation and optional `expert_num`); `robodojo eval` does **not** invoke `eval.sh` directly.
- Policy `setup_eval_*` scripts run with **CWD = policy directory** (relative paths in policies).
- Default transport: **WebSocket** (`protocol: ws` in policy `deploy.yml`; see `scripts/eval_policy.sh`).

**Split eval:**

```
robodojo.sh server  →  scripts/internal/run_policy_server.sh
robodojo.sh client  →  scripts/eval_policy.sh  →  src/eval_client/main.py
```

**Docker:** sim client in container, policy server on host — see `docker/README.md`.

Fast validation:

```bash
bash scripts/robodojo.sh doctor --skip-isaac --skip-conda --skip-policy
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/<POLICY> --task stack_bowls --ckpt <CKPT> --policy-env <ENV> --dry-run
bash scripts/robodojo.sh smoke --policy-dir XPolicyLab/policy/<POLICY> --ckpt <CKPT> --policy-env <ENV> --only stack_bowls,push_T --dry-run
```

Full eval-infrastructure acceptance is **sequential**, not parallel:

```bash
bash scripts/robodojo.sh smoke --policy-dir XPolicyLab/policy/<POLICY> --ckpt <CKPT> --policy-env <ENV> --fail-fast
```

New code imports from `env.*`.

## Tasks

- **54 runnable task configs** (42 base + 12 `_random` generalization variants).
- Registration is dynamic: `task_registry.load_task_class(task_name)` imports `task.RoboDojo.tasks.<task_name>` and expects class name == module name.
- Config path: `task/RoboDojo/config/<task_name>.yml`.
- Inventory: `bash scripts/robodojo.sh tasks` or `python scripts/internal/task_inventory.py --format json --check`.
- `task/RoboDojo/demos/` (e.g. `dlc`) is outside the benchmark inventory scan.

## Naming

- **Config / scripts**: lowercase `snake_case` — `env_cfg/`, `utils/`, `scripts/`
- **Python packages/directories**: `snake_case` — `env/robot_manager/`, `env/scene_manager/objects/`
- **Python module files**: `snake_case.py` — `task_env.py`, `obs_manager.py`
- **Python classes**: **PascalCase** — `RewardManager`, `TaskEnv`
- **Tasks**: filename, YAML name, exported env class, and layout/result paths must match. Most tasks use lowercase `snake_case`; **`play_Xylophone`**, **`swap_T`**, **`push_T`**, and **`push_T_random`** match uppercase asset names.
- **Submodules**: keep upstream casing — `XPolicyLab/`, `Assets/`

## Code Conventions

### Task File Structure

Every task Python file must follow this pattern:

```python
class <TaskName>Common:
    def __init__(self, config, app, **kwargs): ...
    def _post_setup_scene(self, sim): ...
    def reset(self, seed=None, options=None): ...
    def soft_reset(self, seed=None, options=None): ...  # when state must reset between episodes
    def run_reward(self): ...                           # reward / success check

class <task_name>(<TaskName>Common, TaskEnv):
    pass
```

- Class names use **PascalCase**; the exported env class uses **snake_case** matching the YAML / Python module basename.
- `run_reward()` must call `self.reward_manager.check(...)` (or equivalent) meaningfully — never leave success checks trivially always `True`.
- No print statements in task logic; use the shared logger if needed.

### Config File (YAML)

Typical rigid-body tasks define:
- `Rigid` section with at least one object category and a `label` list.
- `common` with `xlim`, `ylim`, `rotate_rand` (and `rotate_deg` when rotation is enabled).
- `select_mode` with `nums` and `mode`.

Garment tasks (e.g. `fold_clothes*`) use a `Garment` section instead of `Rigid`. Many configs add `Geometry`, `Clutter`, etc.

### Commit Message Format

```
[Scope] type: short description

Scopes: Task | Env | Config | scripts | utils | fix | refactor
Types : feat | fix | update | refactor | docs | chore
```

Examples:
- `[Task] update: fix stack_bowls run_reward bowl_color check`
- `[Env] fix: correct gripper warm-up timing`
- `[Config] update: adjust xlim for insert_test_tubes`

### Branch Naming

`<contributor>/<scope>-<short-description>`  e.g. `zixuan/task-hang-mugs`

## Validation Loops

Use the **smallest loop** that proves the current change.

### Docs / scripts / config only

```bash
bash -n scripts/robodojo.sh scripts/eval_policy.sh
python scripts/internal/task_inventory.py --format json --check
bash scripts/robodojo.sh doctor --skip-isaac --skip-conda --skip-policy
ruff check .
git diff --check
```

### Dry-run eval path (no Isaac, no policy server)

```bash
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/<POLICY_NAME> \
  --task stack_bowls \
  --ckpt <CKPT_NAME> \
  --policy-env <POLICY_ENV> \
  --dry-run
bash scripts/robodojo.sh smoke \
  --policy-dir XPolicyLab/policy/<POLICY_NAME> \
  --ckpt <CKPT_NAME> \
  --policy-env <POLICY_ENV> \
  --only stack_bowls,push_T \
  --dry-run
```

### Runtime (Isaac Sim + policy ready)

```bash
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/<POLICY_NAME> \
  --task stack_bowls \
  --ckpt <CKPT_NAME> \
  --policy-env <POLICY_ENV> \
  --eval-num 1
```

Smoke/benchmark acceptance: each task must exit `0` **and** write `_result.json` with `eval_time >= 1`. Exit code alone is not enough.

## PR Checklist

### Task PR
- [ ] `config/<task>.yml` + `tasks/<task>.py` both present, names match
- [ ] `run_reward()` calls `self.reward_manager.check(...)` meaningfully
- [ ] YAML labels match Python labels exactly
- [ ] `step_lim` is appropriate for task length
- [ ] Task inherits `TaskEnv` (not `SyncCollectEnv` / `SyncRobotEnv`)
- [ ] New task filenames are lowercase `snake_case` (unless uppercase asset exception)

### Framework / Env PR
- [ ] Backward-compatible; no new mandatory constructor args
- [ ] `reset()` and `soft_reset()` both cover new state when applicable

### Config PR
- [ ] YAML valid; defaults safe; no silent breakage to shared configs

### General
- [ ] Commit messages: `[Scope] type: desc`
- [ ] No `print`, `breakpoint`, or commented-out debug code
- [ ] New files follow Naming rules above
- [ ] For scripts/eval infrastructure, run `robodojo.sh doctor` and at least dry-run `robodojo.sh smoke` before requesting review

---

## Review Instructions for Claude

When asked to review a PR (`/review`):

1. **1-sentence summary** + PR type (Task / Framework / Config / Scripts / Mixed).
2. **Run the relevant checklist** from above. Skip inapplicable sections.
3. **Report only real issues** — skip items that pass. For each issue: file:line, what's wrong, one-line fix.
4. **Score**: ✅ Approve / ⚠️ Approve with comments / ❌ Request changes.

Type-specific red flags to check first:
- *Task*: YAML labels ≠ Python labels? Still using `SkillManager` / `SyncCollectEnv` / `pre_*`?
- *Framework*: breaks existing tasks or eval client? `reset`/`soft_reset` miss new state?
- *Config*: YAML valid? shared config silently breaks other tasks?

When creating a PR (`gh pr create`):
- Title: `[Scope] type: description (≤70 chars)`
- Body: summary bullets, affected tasks, test plan (`bash scripts/robodojo.sh eval ...` or task-specific eval).

---

## Common Pitfalls

- `check_success` always `True` → garbage eval metrics
- YAML label `["target"]` but code uses `label="obj"` → silent mismatch
- Missing `soft_reset` override → state leak between episodes
- `step_lim = 200` on long-horizon tasks → truncation artifacts
- Camera config changes affecting all tasks without announcement
- Treating an `eval.sh` exit code alone as smoke success → also require `_result.json` with `eval_time >= 1`
- Editing policy logic in RoboDojo instead of XPolicyLab
- Running policy `setup_eval_*` without CWD in the policy directory
- Treating submodule content as owned by this repo unless explicitly updating the gitlink pin