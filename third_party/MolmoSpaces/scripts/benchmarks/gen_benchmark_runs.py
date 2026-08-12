from pathlib import Path
# exp_config_names = "FrankaOpenHardBench", "FrankaCloseHardBench", "FrankaPickHardBench", "FrankaPickandPlaceHardBench", "FrankaPickandPlaceNextToHardBench"
# scene_datasets = "ithor", "procthor-10k", "procthor-objaverse", "holodeck-objaverse"

exp_config_names = "FrankaPickandPlaceNextToHardBench", "FrankaPickandPlaceColorHardBench"
scene_datasets = ("procthor-objaverse",)

workspace = "ai2/vida"
datagen_template = """python molmo_spaces/data_generation/distributed/manager_multi_machine_sqs_beaker.py \\
    --exp_config_path molmo_spaces.data_generation.config.object_manipulation_datagen_configs \\
    --exp_config_name {exp_config_name} \\
    --tag {tag} \\
    --scene_dataset {scene_dataset} \\
    --split val \\
    --num_houses {num_houses} \\
    --house_repeats {house_repeats} \\
    --output_dir {output_dir} \\
    --exp_count 20 \\
    --beaker_codebase $BEAKER_CODEBASE \\
    --wandb_project_name robomolmo-datagen \\
    --beaker_image_name $BEAKER_IMAGE \\
    --workspace {workspace} \\
"""

benchmark_template = (
    "python scripts/datagen/create_json_benchmark.py --base_path {output_dir} --num_episodes 2000"
)

skiplog = []
for exp_config_name in exp_config_names:
    for scene_dataset in scene_datasets:
        if scene_dataset == "ithor":
            num_houses = 120  # there will be 48 in val
            house_repeats = 100
        if scene_dataset == "procthor-10k":
            num_houses = 1000  # there will be 48 in val
            house_repeats = 10
        else:
            num_houses = 5000
            house_repeats = 2

        tag = f"{exp_config_name.lower()}_{scene_dataset}_bench_run"

        output_dir = f"/weka/prior/datasets/robomolmo/bench_new/{scene_dataset}/"

        if (Path(output_dir) / exp_config_name).exists():
            skiplog.append(Path(output_dir) / exp_config_name)
            continue

        command = datagen_template.format(
            exp_config_name=exp_config_name,
            tag=tag,
            scene_dataset=scene_dataset,
            num_houses=num_houses,
            house_repeats=house_repeats,
            output_dir=output_dir,
            workspace=workspace,
        )
        print(command)

for entry in skiplog:
    print(f"# skipping existing output dir: {entry}")

for exp_config_name in exp_config_names:
    for scene_dataset in scene_datasets:
        output_dir = (
            f"/weka/prior/datasets/robomolmo/bench_new/{scene_dataset}/{exp_config_name}/val"
        )
        command = benchmark_template.format(
            output_dir=output_dir,
        )
        print(command)
