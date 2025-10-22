#!/bin/bash
# Start EmbodiedBench VM using VirtualBox from WSL
# VirtualBox is GPL licensed - no licensing issues!

echo "========================================"
echo "EmbodiedBench VM Starter"
echo "========================================"
echo ""

# Enable Windows access for Vagrant (required for VirtualBox in WSL)
export VAGRANT_WSL_ENABLE_WINDOWS_ACCESS="1"

# Add VirtualBox to PATH
export PATH="$PATH:/mnt/c/Program Files/Oracle/VirtualBox"

# Verify VirtualBox is accessible
if ! command -v VBoxManage.exe &> /dev/null; then
    echo "❌ Error: VirtualBox not found!"
    echo ""
    echo "Please install VirtualBox on Windows from:"
    echo "  https://www.virtualbox.org/wiki/Downloads"
    echo ""
    echo "Note: Only install VirtualBox itself (GPL license - FREE)"
    echo "      Do NOT install the Extension Pack (not needed)"
    exit 1
fi

echo "✅ VirtualBox: $(VBoxManage.exe --version)"
echo ""

# Start the VM
echo "🚀 Starting VM (this may take a while on first run)..."
echo ""

vagrant up --provider=virtualbox

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ VM is running!"
    echo ""
    echo "To connect: vagrant ssh"
    echo "To stop: vagrant halt"
    echo "To destroy: vagrant destroy"
fi
