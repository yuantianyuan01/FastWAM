# MolmoBot Benchmarks

## Usage

We first run an evaluation like
```bash
python molmo_spaces/evaluation/eval_main.py \
  <YOUR_POLICY_CONFIG> \
  [OPTIONS] \
  --benchmark_dir <BENCHMARK_DIR> \
  --output_dir <eval_output_dir>
```
Please see detailed commands for each task type below, and replace `<YOUR_POLICY_CONFIG>` with your evaluation config (e.g. `molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig`).

Finally, run the evaluation output script that aggregates results as csv files:
```bash 
python scripts/benchmarks/eval_to_csv.py \
  <eval_output_dir>/<date_str> \
  <policy_name> \
  --success-condition both \
  --output-csv /eg/path/to/<task_type>/<policy_name>.csv
```

## Benchmarks with classic renderer

For benchmarks using classic renderer we need to install the `mujoco` version from [our dependencies](../../pyproject.toml), e.g., by calling
```bash
pip install -e ".[mujoco]"
```
from the project root directory.

### Pick-MSProc (Pick-v1.5)

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231
```

### Pick-Classic (Pick-v2-classic)

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark
```

## Benchmarks with filament renderer

For benchmarks using filament we should install `mujoco-filament` from [our dependencies](../../pyproject.toml), e.g., by calling
```bash
pip install -e ".[mujoco-filament]"
```
from the project root directory and pass the `--use-filament` option to the evaluation script.

### Pick-Filament (Pick-v2-filament)

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> \
  --use-filament \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark
```

### Pick-RandCam (Pick-v2-rand-cam)

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> \
  --use-filament \
  --camera_names randomized_zed2_analogue_1 wrist_camera_zed_mini \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark
```

### Pick & Place (PnP-v2)

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> \
  --use-filament \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_benchmark
```

### Pick & Place-NextTo (PnP-next-to-v2)

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> \
  --use-filament \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceNextToHardBench/FrankaPickandPlaceNextToHardBench_20260305_json_benchmark
```

### Pick & Place-Color (PnP-color-v2)

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> \
  --use-filament \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceColorHardBench/FrankaPickandPlaceColorHardBench_20260304_json_benchmark
```
