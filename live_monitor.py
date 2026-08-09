# live_monitor.py
# Live performance graph — run alongside run.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os
import time
import collections
from datetime import datetime

# ── Data stores ──
fps_data     = collections.deque(maxlen=60)
alert_times  = []
times        = collections.deque(maxlen=60)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle("Robust PCA — Live Monitor", fontsize=13)
fig.patch.set_facecolor("#1e1e2e")

for ax in axes:
    ax.set_facecolor("#2a2a3e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")


def update(frame_num):
    # ── Read detection log for alert count ──
    alert_count = 0
    if os.path.exists("detection_log.csv"):
        with open("detection_log.csv") as f:
            lines = f.readlines()
            alert_count = max(0, len(lines) - 1)   # Subtract header

    # Simulated FPS (replace with shared memory in advanced version)
    import random
    fps_data.append(random.uniform(4, 12))
    times.append(datetime.now().strftime("%H:%M:%S"))

    # ── Plot 1: FPS over time ──
    axes[0].clear()
    axes[0].set_facecolor("#2a2a3e")
    axes[0].plot(list(fps_data), color="#00d4aa",
                 linewidth=2, label="FPS")
    axes[0].fill_between(range(len(fps_data)),
                         list(fps_data), alpha=0.2,
                         color="#00d4aa")
    axes[0].axhline(y=5, color="#ff6b6b",
                    linestyle="--", linewidth=1,
                    label="Min acceptable (5 FPS)")
    axes[0].set_ylabel("FPS", color="white")
    axes[0].set_title("Live FPS", color="white")
    axes[0].legend(facecolor="#2a2a3e",
                   labelcolor="white", fontsize=8)
    axes[0].set_ylim(0, 20)
    axes[0].tick_params(colors="white")

    # ── Plot 2: Alert count over time ──
    axes[1].clear()
    axes[1].set_facecolor("#2a2a3e")
    axes[1].bar(["Total Alerts"], [alert_count],
                color="#ff6b6b", width=0.4)
    axes[1].set_ylabel("Count", color="white")
    axes[1].set_title("Detection Events (from log)",
                      color="white")
    axes[1].tick_params(colors="white")
    axes[1].set_ylim(0, max(10, alert_count + 2))

    # Show current time
    axes[1].text(0.98, 0.95,
                 datetime.now().strftime("%H:%M:%S"),
                 transform=axes[1].transAxes,
                 color="#aaaaaa", fontsize=9,
                 ha="right", va="top")

    plt.tight_layout()


ani = animation.FuncAnimation(
    fig, update, interval=1000, cache_frame_data=False)

plt.show()