#!/bin/bash

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
export EMBODIED_BENCH_ROOT=$(pwd)


# # Environment for ```Habitat and Alfred```
conda env create -f conda_envs/environment.yaml 
conda activate embench
pip install -e .

# Environment for ```EB-Navigation```
conda env create -f conda_envs/environment_eb-nav.yaml 
conda activate embench_nav
pip install -e .

# Environment for ```EB-Manipulation```
conda env create -f conda_envs/environment_eb-man.yaml 
conda activate embench_man
pip install -e .

# Install Git LFS
git lfs install
git lfs pull

# Install EB-ALFRED
conda activate embench
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
mv EB-ALFRED embodiedbench/envs/eb_alfred/data/json_2.1.0

# Install EB-Habitat
conda activate embench
conda install -y habitat-sim==0.3.0 withbullet  headless -c conda-forge -c aihabitat
git clone -b 'v0.3.0' --depth 1 https://github.com/facebookresearch/habitat-lab.git ./habitat-lab
cd ./habitat-lab
pip install -e habitat-lab
cd ..
conda install -y -c conda-forge git-lfs
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets
mv data embodiedbench/envs/eb_habitat

# Install EB-Manipulation (idempotent & aligned with evaluation script)
conda activate embench_man
cd embodiedbench/envs/eb_manipulation

echo "[EB-Man] Setting up CoppeliaSim + PyRep..."

# Download CoppeliaSim only if not already present
if [ ! -d CoppeliaSim_Pro_V4_1_0_Ubuntu20_04 ]; then
	if [ ! -f CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz ]; then
		echo "[EB-Man] Downloading CoppeliaSim archive..."
		wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
	fi
	echo "[EB-Man] Extracting CoppeliaSim..."
	tar -xf CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
	rm -f CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
else
	echo "[EB-Man] CoppeliaSim directory already exists. Skipping download/extract."
fi

# Prefer keeping CoppeliaSim local; evaluation script now falls back to this path.
export COPPELIASIM_ROOT="$(pwd)/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"

# Ensure usrset.txt exists for PyRep setup.py (some versions expect it)
if [ ! -f "$COPPELIASIM_ROOT/system/usrset.txt" ]; then
	echo "[EB-Man] Creating missing usrset.txt"
	mkdir -p "$COPPELIASIM_ROOT/system" && touch "$COPPELIASIM_ROOT/system/usrset.txt"
fi

# Clone PyRep only if absent
if [ ! -d PyRep ]; then
	echo "[EB-Man] Cloning PyRep repository..."
	git clone https://github.com/stepjam/PyRep.git
else
	echo "[EB-Man] PyRep already cloned. Pulling latest..."
	(cd PyRep && git pull --ff-only || echo "[EB-Man] PyRep pull failed; keeping existing state")
fi

cd PyRep
pip install -r requirements.txt
pip install -e .
cd ..

# Install any local environment requirements (if present)
if [ -f requirements.txt ]; then
	pip install -r requirements.txt || echo "[EB-Man] Warning: local env requirements install failed"
fi
pip install -e .

# Copy addon script
if [ -f simAddOnScript_PyRep.lua ]; then
	cp ./simAddOnScript_PyRep.lua "$COPPELIASIM_ROOT" || echo "[EB-Man] Warning: failed to copy simAddOnScript_PyRep.lua"
fi

# Fetch EB-Manipulation dataset if missing
if [ ! -d data ]; then
	echo "[EB-Man] Downloading EB-Manipulation dataset..."
	git clone https://huggingface.co/datasets/EmbodiedBench/EB-Manipulation tmp_eb_man_dataset
	mv tmp_eb_man_dataset/data ./
	rm -rf tmp_eb_man_dataset
else
	echo "[EB-Man] Dataset directory already present. Skipping clone."
fi

# Quick import test for PyRep
python -c "import pyrep; print('[EB-Man] PyRep import OK')" || { echo '[EB-Man] PyRep import failed.' >&2; exit 1; }

echo "[EB-Man] Setup complete. COPPELIASIM_ROOT=$COPPELIASIM_ROOT"
cd ../../..
