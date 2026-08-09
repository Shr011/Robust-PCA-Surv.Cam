# auto_optimizer.py
# Finds the best config settings for your laptop automatically

import numpy as np
import cv2
import time
from separator import Separator
from rpca import RobustPCA

print("=" * 50)
print("   Auto Optimizer")
print("=" * 50)
print("\nTesting your laptop's performance...")
print("This will take about 1-2 minutes.\n")

results = []

# Test different frame sizes
configs = [
    {"w": 160, "h": 120, "label": "Tiny   160x120"},
    {"w": 320, "h": 240, "label": "Small  320x240"},
    {"w": 480, "h": 360, "label": "Medium 480x360"},
    {"w": 640, "h": 480, "label": "Large  640x480"},
]

print(f"  {'Size':<20} {'RPCA Time':>10} {'FPS':>8}  {'Verdict'}")
print(f"  {'─'*52}")

for cfg in configs:
    w, h    = cfg["w"], cfg["h"]
    pixels  = w * h
    frames  = 50

    np.random.seed(0)
    M = np.random.rand(pixels, frames)

    t_start = time.time()
    rpca    = RobustPCA(max_iter=30, tol=1e-4,
                        n_components=min(5, frames-1))
    L, S    = rpca.fit(M)
    t_rpca  = time.time() - t_start

    # Estimate per-frame detection time
    bg   = np.mean(L, axis=1)
    t_det_start = time.time()
    for _ in range(100):
        frame = np.random.rand(h, w)
        flat  = frame.flatten()
        S_f   = flat - bg
        sep   = Separator(threshold=0.08, min_area=300)
        mask  = sep.process_foreground(S_f, h, w)
        boxes = sep.find_objects(mask)
    t_det = (time.time() - t_det_start) / 100 * 1000

    est_fps = 1000 / t_det if t_det > 0 else 0

    if t_rpca < 20 and est_fps >= 8:
        verdict = " Excellent"
    elif t_rpca < 45 and est_fps >= 4:
        verdict = " Good"
    elif est_fps >= 2:
        verdict = " Acceptable"
    else:
        verdict = " Too slow"

    print(f"  {cfg['label']:<20} {t_rpca:>9.1f}s "
          f"{est_fps:>7.1f}  {verdict}")

    results.append({
        "label"    : cfg["label"],
        "w"        : w,
        "h"        : h,
        "rpca_time": t_rpca,
        "fps"      : est_fps,
        "verdict"  : verdict
    })

# Pick best config
good = [r for r in results if "ok" in r["verdict"]]
best = max(good, key=lambda x: x["fps"]) if good else results[0]

print(f"\n{'─'*52}")
print(f"   Recommended settings for your laptop:")
print(f"{'─'*52}")
print(f"\n  Update your config.py with these values:\n")
print(f"  FRAME_WIDTH  = {best['w']}")
print(f"  FRAME_HEIGHT = {best['h']}")

if best["fps"] >= 10:
    print(f"  LEARN_FRAMES = 100")
elif best["fps"] >= 5:
    print(f"  LEARN_FRAMES = 80")
else:
    print(f"  LEARN_FRAMES = 50")

print(f"\n  Expected performance:")
print(f"  → RPCA learning : ~{best['rpca_time']:.0f}s")
print(f"  → Detection FPS : ~{best['fps']:.1f} FPS")
print(f"\n{'='*50}")
