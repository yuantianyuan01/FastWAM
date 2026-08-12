# Quick start to openpi evaluation


In your host workstation, run the following commands to set up the docker environment. And then a simple `make` will take you to the docker environment.
```bash
#in host
bash setup_docker_for_host.sh
```


```bash
#in host
make
```
>[!NOTE]
>First time running `make` should take around 11 minutes, not including pulling public image. Basically it builds a private image on top of the public image, changing the ownership of the installed packages to the current user. 

>[!NOTE]
>Check the `DISPLAY` environment variable with `echo $DISPLAY`, and make sure it's not empty. 



Download data assets

```bash
#in docker
deactivate #deactivate uv env
conda activate #activate base env
cd /app/VLABench
python scripts/download_assets.py
```

Generate dataset
```bash
#in docker
deactivate #deactivate uv env
conda activate #activate base env
cd /app/VLABench
./dataset_generation.sh
```

```bash
#in docker
# convert dataset to lerobot format
deactivate #deactivate uv env
conda activate #activate base env
cd /app/VLABench
python scripts/convert_to_lerobot.py --dataset-name select_toy --task-list select_toy --dataset-path ~/data/vlabench/trajectory/dataset
python scripts/convert_to_lerobot.py --dataset-name select_fruit  --task-list select_fruit --dataset-path ~/data/vlabench/trajectory/dataset
```

Visualize one episode of the dataset. 

```bash
#in docker
#need to setup DISPLAY env variable
cd /app/lerobot
conda deactivate
deactivate
conda activate lerobot
python lerobot/scripts/visualize_dataset.py --repo-id select_toy  --episode-index 0 
# try choose another --episode-index to visualize different episode
# --repo-id canbe `select_toy` or `select_fruit` 
```

>[!NOTE]
>Note that, you need a display connected to your host gpu, and set up the DISPLAY env variable. It could pose problem if you are using a remote server through ssh.


launch the server and client in two different terminals.

running server:

>[!NOTE]
>You will need to download the checkpoint and put it in the right place.

```bash
#in docker
#server
cd /app/VLABench/third_party/openpi
conda deactivate
source examples/vlabench/.venv/bin/activate
/app/.local/bin/uv run scripts/serve_policy.py --env VLABENCH policy:checkpoint --policy.config=pi0_vlabench_primitive_lora --policy.dir=${HOME}/data/vlabench_checkpoints/pi0_base_vlabench_lora/99999

```


running client should start evaluation:

```bash
#in docker
#client
cd /app/VLABench/third_party/openpi
conda deactivate
source examples/vlabench/.venv/bin/activate
python examples/vlabench/eval.py --args.episode-config-path /app/VLABench/VLABench/configs/evaluation/tracks/track_1_in_distribution.json --args.save_dir ${HOME}/data/vlabench_results/pi0_base_vlabench_lora/track_1
```