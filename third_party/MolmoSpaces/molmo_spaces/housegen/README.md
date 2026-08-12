# House Generation module for iTHOR and ProcTHOR houses

## Setup

First, make sure you have installed `bpy` from a custom index, as we're using an old version that
is no longer available in PyPI for `Python3.10`

```bash
# Install bpy, the version we use for python 3.10
pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/
```

After this, just make sure you install the `molmo_spaces` package in editable mode, like this,
and include the dependencies from the `housegen` group:

```bash
# From the root of the project
pip install -e .[housegen]
```

## Usage

### Generate houses from a specific dataset

To generate a dataset we should just use the `exporter.py` script. There's a helper shortcut that
we can use named `generate-houses` which basically calls this script. The most basic call we can use
is the following:

```bash
generate-houses --dataset DATASET_NAME --split SPLIT_NAME --max-workers NUM_WORKERS
```

This will generate the dataset named `DATASET_NAME`, with the given split `SPLIT_NAME` using the
provided number of parallel workers `NUM_WORKERS`. For example, below are some concrete examples:

```bash
# Generate procthor-10k train set
generate-houses --dataset procthor-10k --split train --max-workers 10
# Generate ithor set (doesn't need split, bc we only have one set for ithor)
generate-houses --dataset ithor --max-workers 10
```

An alternative is to use the `run_houses_generation.sh` bash script, like this:

```bash
./scripts/housegen/run_houses_generation.sh --dataset DATASET_NAME --split SPLIT_NAME --max-workers NUM_WORKERS
```

If you need to only export one single house, use the `--start` and `--end` arguments, like this:

```bash
./scripts/housegen/run_houses_generation.sh --dataset procthor-10k --split train --start 42 --end 43
```

### Run house generation and scene processing using the current scene tests available

To export and run the scenes tests|processing steps you can use the `run_houses_processing.sh` script, like this:

```bash
./scripts/housegen/run_houses_processing.sh --dataset procthor-10k --split train --max-workers 10
```

### Generate on weka using experiments

To launch experiments on weka that will generate a whole dataset we can make use of the `beaker_launch_houses_generation.py`
script, which takes advantange of beaker-py to launch the jobs required for generating and processing
a given dataset. For example, to use it to generate procthor-objaverse-train on weka we can do the following:

```bash
python scripts/housegen/beaker_launch_houses_generation.py --action generate \
    --dataset procthor-objaverse --split train --njobs 10 --workers-per-job 50
```

This will launch 10 experiment jobs on beaker, each running with 50 parallel workers to generate the
houses for `procthor-objaverse-train`. The generated houses will be stored on weka at the location
`/weka/prior-default/datasets/molmo-spaces/assets/scenes/procthor-objaverse-train`

If after generation you want to run tests+filtering of the houses, you can run the following:

```bash
python scripts/housegen/beaker_launch_houses_generation.py --action process \
    --dataset procthor-objaverse --split train --njobs 10 --workers-per-job 50
```

And if you want to do both generation and processing in the same jobs, then just run the following:

```bash
python scripts/housegen/beaker_launch_houses_generation.py --action generate-and-process \
    --dataset procthor-objaverse --split train --njobs 10 --workers-per-job 50
```

We're using the bekaer image `wilbertp/mjt-houses-generation`, with the workspace `ai2/mujoco-thor-objaverse`,
so we can run the jobs with priority `urgent`.
