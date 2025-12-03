#!/bin/bash

# Cleanup script for stuck AI2-THOR processes
# These processes often get stuck running at 100% CPU when experiments
# are interrupted or crash without proper cleanup.

echo "=================================================="
echo "AI2-THOR Process Cleanup Script"
echo "=================================================="
echo ""

# Count current thor processes (exclude grep and defunct/zombie processes)
THOR_COUNT=$(ps aux | grep -i "thor-Linux" | grep -v grep | grep -v defunct | wc -l)

echo "Found $THOR_COUNT active AI2-THOR processes"
echo ""

if [ $THOR_COUNT -eq 0 ]; then
    echo "No AI2-THOR processes found. Nothing to clean up."
    exit 0
fi

echo "Active AI2-THOR processes:"
echo "------------------------"
ps aux | grep -i "thor-Linux" | grep -v grep | grep -v defunct | head -20
echo ""

# Ask for confirmation
read -p "Do you want to kill all these AI2-THOR processes? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Killing all AI2-THOR processes..."
    pkill -9 -f "thor-Linux"
    sleep 2
    
    # Check remaining
    REMAINING=$(ps aux | grep -i "thor-Linux" | grep -v grep | grep -v defunct | wc -l)
    echo ""
    echo "Cleanup complete!"
    echo "Remaining processes: $REMAINING"
    
    if [ $REMAINING -gt 0 ]; then
        echo ""
        echo "Warning: Some processes may still be running. You can:"
        echo "  1. Wait a moment and run this script again"
        echo "  2. Manually kill specific PIDs if needed"
    fi
else
    echo "Cleanup cancelled."
fi

echo ""
echo "To prevent this in the future, ensure experiments cleanup properly"
echo "when interrupted (Ctrl+C), and consider using the test_thor_cleanup.py"
echo "script to verify proper AI2-THOR shutdown."
