# MolmoSpaces Benchmarks

## Usage

We first run an evaluation like
```bash
python molmo_spaces/evaluation/eval_main.py \
  <YOUR_POLICY_CONFIG> \
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

## Benchmarks

### Close (Close-v1)

```bash
python molmo_spaces/evaluation/eval_main.py YOUR_POLICY_CONFIG \
  --benchmark_dir assets/benchmarks/molmospaces-bench-v1/ithor/FrankaCloseDataGenConfig/FrankaCloseDataGenConfig_20260123_json_benchmark
```

### Open (Open-v1)

```bash
python molmo_spaces/evaluation/eval_main.py YOUR_POLICY_CONFIG \
  --benchmark_dir assets/benchmarks/molmospaces-bench-v1/ithor/FrankaOpenDataGenConfig/FrankaOpenDataGenConfig_20260123_json_benchmark
```

### Pick (Pick-v1.1)

```bash
python molmo_spaces/evaluation/eval_main.py YOUR_POLICY_CONFIG \
  --benchmark_dir assets/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231
```

### Pick and Place (PnP-v1)

```bash
python molmo_spaces/evaluation/eval_main.py YOUR_POLICY_CONFIG \
  --benchmark_dir assets/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark
```
