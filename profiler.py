# profiler.py
# Measures speed of every part of the pipeline

import time
import numpy as np
import collections


class Profiler:
    """
    Tracks how long each part of the pipeline takes.
    Call start("name") before a section, stop("name") after.
    """

    def __init__(self, history=100):
        # Stores last N timings for each section
        self._times   = collections.defaultdict(
                            lambda: collections.deque(maxlen=history))
        self._starts  = {}
        self._history = history

    def start(self, name):
        self._starts[name] = time.perf_counter()

    def stop(self, name):
        if name in self._starts:
            elapsed = time.perf_counter() - self._starts[name]
            self._times[name].append(elapsed * 1000)   # Convert to ms

    def avg(self, name):
        t = self._times.get(name)
        if not t:
            return 0.0
        return sum(t) / len(t)

    def report(self):
        """Print a formatted timing report."""
        print(f"\n{'─'*45}")
        print(f"    Performance Report")
        print(f"{'─'*45}")
        print(f"  {'Section':<22} {'Avg(ms)':>8}  {'Est FPS':>8}")
        print(f"{'─'*45}")

        total_ms = 0
        for name, times in self._times.items():
            avg_ms  = sum(times) / len(times)
            est_fps = 1000 / avg_ms if avg_ms > 0 else 0
            total_ms += avg_ms
            print(f"  {name:<22} {avg_ms:>8.1f}  {est_fps:>8.1f}")

        print(f"{'─'*45}")
        overall_fps = 1000 / total_ms if total_ms > 0 else 0
        print(f"  {'TOTAL':<22} {total_ms:>8.1f}  {overall_fps:>8.1f}")
        print(f"{'─'*45}\n")
        return total_ms
