# Note

# TODOs
+ [] fix prefix padding: currently add a flag to filter out too short data, about 1/3. But we should actually support right padding. 
+ [] resize_and_crop lots of log: to disable, use os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all,1=info,2=warn,3=error
+ [x] support context caching
+ [x] separate out a training dataset and a validation dataset
+ [x] support causal training
+ [ ] support droid video dataset
+ [ ] causal forcing for multiple frames generation
+ [ ] make sure shuffle is working


### Run pretraining
```bash
torch run --standalone --nproc_per_node=8 scripts/precompute_text_embeds_oxe.py \
--dataset_dirs /scratch/cgao304/dev/datasets/ \
--data_mix bridge \
--text_embedding_cache_dir ./data/cache/bridge/ \
--no-overwrite
```

common invocation:
```bash
bash scripts/train_causal_wan22.sh 8 task=causal_wan22_pretrain

bash scripts/train_causal_wan22.sh 8 task=causal_wan22_pretrain wandb.enabled=true wandb.project=fast-wam wandb.workspace=gaochenxiao

bash scripts/train_causal_wan22.sh 8 task=causal_wan22_pretrain \
resume=./runs/causal_wan22_pretrain/RUN_ID/checkpoints/state/step_00010000
```
