#!/usr/bin/env python
"""Test script to verify AI2THOR cleanup works correctly."""

import sys
import os
import time
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv

def get_thor_processes():
    """Find all thor processes on the system."""
    thor_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'thor-' in cmdline.lower() or 'ai2thor' in cmdline.lower():
                thor_procs.append((proc.info['pid'], proc.info['name']))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return thor_procs

def main():
    print(f"[TEST] Parent PID: {os.getpid()}")
    
    initial_procs = get_thor_processes()
    print(f"[TEST] Initial THOR processes: {len(initial_procs)}")
    
    print("\n[TEST] Creating EBNavigationEnv...")
    env = EBNavigationEnv(eval_set='base', exp_name='cleanup_test', selected_indexes=[0])
    
    print("[TEST] Resetting environment...")
    env.reset()
    
    time.sleep(3)
    
    after_init_procs = get_thor_processes()
    print(f"\n[TEST] THOR processes after init: {len(after_init_procs)}")
    
    new_procs = set(after_init_procs) - set(initial_procs)
    if new_procs:
        print(f"[TEST] New THOR processes: {len(new_procs)}")
        for pid, name in new_procs:
            print(f"  - PID={pid}, name={name}")
    
    print("\n[TEST] Exiting... cleanup should trigger")
    sys.exit(0)

if __name__ == "__main__":
    main()
