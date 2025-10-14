#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
export EMBODIED_BENCH_ROOT=$(pwd)

echo "Starting EmbodiedBench uninstallation..."
echo "This will remove all conda environments, data, and installations created by install.sh"
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstallation cancelled."
    exit 1
fi

echo "Removing conda environments..."

# Remove conda environments
echo "Removing embench environment..."
conda env remove -n embench -y 2>/dev/null || echo "embench environment not found or already removed"

echo "Removing embench_nav environment..."
conda env remove -n embench_nav -y 2>/dev/null || echo "embench_nav environment not found or already removed"

echo "Removing embench_man environment..."
conda env remove -n embench_man -y 2>/dev/null || echo "embench_man environment not found or already removed"

echo "Removing installed data and directories..."

# Remove EB-ALFRED data
if [ -d "embodiedbench/envs/eb_alfred/data/json_2.1.0" ]; then
    echo "Removing EB-ALFRED data..."
    rm -rf embodiedbench/envs/eb_alfred/data/json_2.1.0
fi

# Remove EB-Habitat data and habitat-lab
if [ -d "embodiedbench/envs/eb_habitat" ]; then
    echo "Removing EB-Habitat data..."
    rm -rf embodiedbench/envs/eb_habitat
fi

if [ -d "habitat-lab" ]; then
    echo "Removing habitat-lab directory..."
    rm -rf habitat-lab
fi

# Remove EB-Manipulation components
if [ -d "embodiedbench/envs/eb_manipulation/PyRep" ]; then
    echo "Removing PyRep from eb_manipulation..."
    rm -rf embodiedbench/envs/eb_manipulation/PyRep
fi

if [ -d "embodiedbench/envs/eb_manipulation/data" ]; then
    echo "Removing EB-Manipulation data..."
    rm -rf embodiedbench/envs/eb_manipulation/data
fi

# Remove CoppeliaSim installation
if [ -d "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04" ]; then
    echo "Removing CoppeliaSim installation..."
    rm -rf CoppeliaSim_Pro_V4_1_0_Ubuntu20_04
fi

# Remove the copied simAddOnScript_PyRep.lua from CoppeliaSim root if it existed
# (This file would have been copied to the CoppeliaSim directory, but since we're removing the whole directory, it's already handled)

# Clean up any egg-info directories that might have been created by pip install -e .
if [ -d "embodiedbench.egg-info" ]; then
    echo "Removing embodiedbench.egg-info..."
    rm -rf embodiedbench.egg-info
fi

# Note: Git LFS is a global installation, so we don't uninstall it automatically
# Users can run 'git lfs uninstall' manually if they want to remove it completely

echo ""
echo "Uninstallation completed!"
echo ""
echo "Note: Git LFS was not uninstalled as it may be used by other projects."
echo "If you want to remove Git LFS completely, run: git lfs uninstall"
echo ""
echo "The following were removed:"
echo "- Conda environments: embench, embench_nav, embench_man"
echo "- EB-ALFRED data directory"
echo "- EB-Habitat data directory and habitat-lab"
echo "- EB-Manipulation data and PyRep"
echo "- CoppeliaSim installation"
echo "- Local package installations (embodiedbench.egg-info)"