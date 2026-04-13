# Note

# TODOs
+ [] fix prefix padding: currently add a flag to filter out too short data, about 1/3. But we should actually support right padding. 
+ [] resize_and_crop lots of log: to disable, use os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all,1=info,2=warn,3=error
+ [x] support context caching
+ [x] separate out a training dataset and a validation dataset
+ [x] support causal training
+ [x] support droid video dataset
+ [ ] diffusion forcing for multiple frames generation + multi-horizon prediction
+ [x] make sure shuffle is working

# note
+ support resume training
+ support teacher forcing evaluation
+ check the conformity to bidirectional wan22 in evaluation
+ check what is this

```
Building OXE train dataset (rank=0, seed=42)
Using LIBERO constants:
  NUM_ACTIONS_CHUNK = 8
  ACTION_DIM = 7
  PROPRIO_DIM = 8
  ACTION_PROPRIO_NORMALIZATION_TYPE = bounds_q99
If needed, manually set the correct constants in `oxe_dataset/constants.py`!
Load dataset info from data/datasets/droid/1.0.1
Creating a tf.data.Dataset reading 2048 files located in folders: data/datasets/droid/1.0.1.
Constructing tf.data.Dataset droid_101 for split all, from data/datasets/droid/1.0.1
Computing dataset statistics. This may take a bit, but should only need to happen once.
```



### Run pretraining
```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds_oxe.py \
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
