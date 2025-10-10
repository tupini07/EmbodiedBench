import os

import numpy as np
from ai2thor.controller import Controller

print("=" * 60)
print("TEST 1: headless=False")
print("=" * 60)
c = Controller(headless=False, x_display="1")
print(
    f"Controller initialized (headless=False). last_event.frame is None: {c.last_event.frame is None}"
)
event = c.step(dict(action="Initialize", renderImage=True))
print(f"After Initialize: frame is None: {event.frame is None}")
if event.frame is not None:
    print(f"Frame shape: {event.frame.shape}")
    print("✓ headless=False works!")
c.stop()

print("\n" + "=" * 60)
print("TEST 2: headless=True")
print("=" * 60)
c2 = Controller(headless=True, x_display="1")
print(
    f"Controller initialized (headless=True). last_event.frame is None: {c2.last_event.frame is None}"
)
event2 = c2.step(dict(action="Initialize", renderImage=True))
print(f"After Initialize: frame is None: {event2.frame is None}")
if event2.frame is not None:
    print(f"Frame shape: {event2.frame.shape}")
    print("✓ headless=True works!")
else:
    print("✗ headless=True does NOT work - frame is None!")
c2.stop()

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print("The fix: Change headless=True to headless=False in thor_env.py")
