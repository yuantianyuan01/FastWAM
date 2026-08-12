## Install Isaac Sim and Isaac Lab to use Omniverse RTX Renderer

## Prerequisites

Reference: [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-lab)

Installing Isaac Sim with pip requires GLIBC 2.34+ version compatibility. Check your GLIBC version:

```bash
ldd --version
```


**Note:** Some Linux distributions may have compatibility issues. For example, Ubuntu 20.04 LTS has GLIBC 2.31 by default. In such cases, follow the Isaac Sim Binaries Installation approach below.

## Install Isaac Sim Binaries

Reference: [Isaac Lab Binaries Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html#isaaclab-binaries-installation)

### 1. Install Omniverse Launcher
Download link (Linux): https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage


```bash
chmod a+x omniverse-launcher-linux.AppImage
./omniverse-launcher-linux.AppImage
```


### 2. Install Required Components
In the Exchange Tab of the Launcher, install:
- Cache
- Nucleus
- Isaac Sim (found in APPS left tab)

### 3. Verify Installation
```bash
# Isaac Sim root directory
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.2.0"
# Isaac Sim python executable
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

# Test that the simulator runs as expected
# note: you can pass the argument "--help" to see all arguments possible.
${ISAACSIM_PATH}/isaac-sim.sh

# checks that python path is set correctly
${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"

# checks that Isaac Sim can be launched from python
${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/omni.isaac.core/add_cubes.py
```

### 4. Create Virtual Environment
Python path for Isaac Sim: `~/.local/share/ov/pkg/isaac-sim-4.2.0/kit/python/bin/python3`

```bash
~/.local/share/ov/pkg/isaac-sim-4.2.0/kit/python/bin/python3 -m venv issac
```



## Install Isaac Lab
```bash
git clone git@github.com:isaac-sim/IsaacLab.git

# create symlink
ln -s ~/.local/share/ov/pkg/isaac-sim-4.2.0 _isaac_sim

# create and activate venv
./isaaclab.sh --conda isaac
conda activate isaac

# install issac lab
./isaaclab.sh -i
```

### Available Commands

usage: isaaclab.sh [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-c] -- Utility to manage Isaac Lab.

optional arguments:
   -h, --help           Display the help content.
   -i, --install [LIB]  Install the extensions inside Isaac Lab and learning frameworks (rl-games, rsl-rl, sb3, skrl) as extra dependencies. Default is 'all'.
   -f, --format         Run pre-commit to format the code and check lints.
   -p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active).
   -s, --sim            Run the simulator executable (isaac-sim.sh) provided by Isaac Sim.
   -t, --test           Run all python unittest tests.
   -o, --docker         Run the docker container helper script (docker/container.sh).
   -v, --vscode         Generate the VSCode settings file from template.
   -d, --docs           Build the documentation from source using sphinx.
   -c, --conda [NAME]   Create the conda environment for Isaac Lab. Default name is 'isaaclab'.



### Verify Installation

```bash
conda activate isaac
python source/standalone/tutorials/00_sim/create_empty.py
```



## Additional Dependencies

```bash
pip install usd-core # (this install pxr library)
```
