# AI2THOR Fix for Semaphore Error

## Problem
The original ai2thor version 2.1.0 (from 2019) had a critical bug where the Mono runtime failed to create semaphores on modern Linux systems, causing the following error:

```
sem_create: error creating semaphore handle
* Assertion at threadpool.c:1108, condition `job_added != NULL' not met
```

## Solution
Upgraded ai2thor from 2.1.0 to 2.7.4, which uses a newer Unity/Mono build that fixes the semaphore issue.

## Changes Made

### 1. Updated requirements.txt
Changed `ai2thor==2.1.0` to `ai2thor==2.7.4` in:
- `embodiedbench/envs/eb_alfred/requirements.txt`

### 2. Modified thor_env.py
Added compatibility for the new ai2thor API by making the `step()` method accept keyword arguments:

**File**: `embodiedbench/envs/eb_alfred/env/thor_env.py`

```python
# Changed from:
def step(self, action, smooth_nav=False):

# To:
def step(self, action, smooth_nav=False, **kwargs):
```

And passed kwargs to parent class:
```python
super().step(action, **kwargs)
```

Also added `headless=True` to the Controller initialization to properly run in server environments.

### 3. Prerequisites
Make sure you have Xvfb running for headless rendering:
```bash
# Start Xvfb on display :1
python embodiedbench/envs/eb_alfred/scripts/startx.py 1 &

# Set DISPLAY environment variable
export DISPLAY=:1
```

## Installation
```bash
conda activate embench
pip install --upgrade 'ai2thor==2.7.4'
```

## Testing
```bash
export DISPLAY=:1
python -m embodiedbench.envs.eb_alfred.EBAlfEnv
```

## Notes
- The 2.7.4 version is backward compatible with the 2.1.0 scene format used by ALFRED
- The semaphore bug was a kernel-level issue in the old Mono runtime (from 2019)
- The new version uses Unity 2019.4.20f1 with MonoBleedingEdge which resolves the threading issues
