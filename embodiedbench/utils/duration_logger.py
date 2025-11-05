
import os
import time
from typing import Optional

class DurationLogger:
    def __init__(self, name: str):
        self.name = name
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def start(self):
        self.start_time = time.time()
        return self

    def stop(self):
        assert self.start_time is not None, "DurationLogger was not started properly."
        end = time.time()

        duration = end - self.start_time
        
        if os.getenv("EB_SUPRESS_DURATION_LOGS", "0") != "1":
            print(f"[DurationLogger] {self.name}: {duration:.3f} seconds")
            
        self.start_time = None

    def extend(self, additional_name: str):
        return DurationLogger(f"{self.name} => {additional_name}")