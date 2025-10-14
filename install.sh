#!/bin/bash
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

# Install NVIDIA GL libraries required for EGL rendering (fix for multi-GPU systems)
# This resolves the "unable to find CUDA device" error in habitat-sim
# Reference: https://github.com/facebookresearch/habitat-sim/issues/2099
NVIDIA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d'.' -f1)
echo "Installing NVIDIA GL libraries for driver version $NVIDIA_DRIVER_VERSION..."
sudo apt-get update
sudo apt-get install -y libnvidia-gl-$NVIDIA_DRIVER_VERSION libglvnd-dev

# Install habitat-sim with display variant (not headless) to avoid EGL issues
conda install -y habitat-sim==0.3.0 withbullet -c conda-forge -c aihabitat

# Clone and install habitat-lab
git clone -b 'v0.3.0' --depth 1 https://github.com/facebookresearch/habitat-lab.git ./habitat-lab
cd ./habitat-lab
pip install -e habitat-lab
cd ..

# Download Habitat datasets
conda install -y -c conda-forge git-lfs
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets
mv data embodiedbench/envs/eb_habitat

# Reload NVIDIA kernel modules to apply GL library changes (avoids need for reboot)
echo "Reloading NVIDIA kernel modules..."
sudo modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia || true
sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm

# Install EB-Manipulation
conda activate embench_man
cd embodiedbench/envs/eb_manipulation
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
rm CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
mv CoppeliaSim_Pro_V4_1_0_Ubuntu20_04/ $EMBODIED_BENCH_ROOT
export COPPELIASIM_ROOT=$EMBODIED_BENCH_ROOT/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install -e .
cd ..
pip install -r requirements.txt
pip install -e .
cp ./simAddOnScript_PyRep.lua $COPPELIASIM_ROOT
git clone https://huggingface.co/datasets/EmbodiedBench/EB-Manipulation
mv EB-Manipulation/data/ ./
rm -rf EB-Manipulation/
cd ../../..
