# Vendored RoboDojo

This directory vendors code from the upstream RoboDojo repository:

- Upstream project: https://github.com/RoboDojo-Benchmark/RoboDojo
- Upstream commit: `36bfcb7c580b149c6e39ed2eb77d60689152e570`
- Upstream license: Apache License 2.0

License compliance notes:

- The original upstream license is preserved in [`LICENSE`](./LICENSE).
- Files copied from RoboDojo remain subject to the upstream license in this directory.
- RoboDojo assets, generated datasets, evaluation outputs, and checkpoints are not vendored in this repository.

Local integration notes:

- RoboDojo is vendored under `third_party/RoboDojo`, matching the existing `third_party/RoboTwin` layout.
- The FastWAM-specific policy adapter lives in [`experiments/robodojo/fastwam_policy`](../../experiments/robodojo/fastwam_policy).
- Link the adapter into the vendored XPolicyLab tree before evaluation:

```bash
ln -sfn "$(pwd)/experiments/robodojo/fastwam_policy" \
  "$(pwd)/third_party/RoboDojo/XPolicyLab/policy/FastWAM"
```

