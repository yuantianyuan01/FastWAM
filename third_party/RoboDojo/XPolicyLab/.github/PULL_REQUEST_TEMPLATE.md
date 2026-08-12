<!-- Policy submission PR — see CONTRIBUTING.md for the full standard.
     For non-policy changes, delete the sections that do not apply. -->

## Policy
- Name / paper / upstream repo:
- Supported: bench_name=..., env_cfg_type=..., action_type=...
- Training support: full | eval-only (training release ETA: ...)

## Components
- [ ] install.sh
- [ ] model.py (+ __init__.py)
- [ ] images: RGB end to end, decoding only via decode_image_bit, no channel swaps (see CONTRIBUTING.md)
- [ ] deploy.yml (standard key set incl. protocol: ws / host / port, policy_name matches the directory)
- [ ] deploy.py aligned with demo_policy (or divergence explained)
- [ ] eval.sh + setup_eval_policy_server.sh + setup_eval_env_client.sh
- [ ] process_data.sh / train.sh (or eval-only, declared above)
- [ ] policy README with install / data / train / eval commands

## Testing
- [ ] bash -n + py_compile pass
- [ ] EVAL_ENV_TYPE=debug closed loop passes (paste the log tail)
- [ ] Simulator eval: task=..., success=... (if available)

## Checkpoint (required for leaderboard evaluation)
<download script, Hugging Face or ModelScope preferred>

## Limitations / notes
...
