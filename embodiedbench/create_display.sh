#!/bin/bash
set -e

echo "--= Create Display =--"

display="1"

echo "Starting Xvfb :$display..."
Xvfb :$display -screen 0 1024x1024x24 -ac +extension GLX +render -noreset &

export DISPLAY=":$display"
echo "Set DISPLAY=$DISPLAY"

# Wait for Xvfb to start
sleep 3

# Run a tiny initialization to force Unity to start and create Player.log
echo "--= Running AI2Thor initialization... =--"
echo "AI2-THOR version check:"
python -c "import ai2thor; print(f'AI2-THOR Version: {ai2thor.__version__}')" || echo "Could not get AI2-THOR version"
python embodiedbench/force_ai2thor_initialization.py || true
echo "--= Create Display Done =--"
