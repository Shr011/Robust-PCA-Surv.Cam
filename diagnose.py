# diagnose.py
# Run this anytime to test your phone camera connection quality

import time
import urllib.request
import numpy as np
import cv2
from config import SNAPSHOT_URL, PHONE_STREAM_URL

print("=" * 50)
print("    Phone Camera Diagnostics")
print("=" * 50)

# ── Test 1: Snapshot Speed ──
print(f"\n Test 1: Snapshot Speed (10 fetches)")
print(f"   URL: {SNAPSHOT_URL}\n")

times  = []
passed = 0

for i in range(10):
    try:
        t_start   = time.time()
        response  = urllib.request.urlopen(SNAPSHOT_URL, timeout=5)
        url_bytes = response.read()
        t_end     = time.time()

        arr   = np.frombuffer(url_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is not None:
            ms = (t_end - t_start) * 1000
            times.append(ms)
            passed += 1
            print(f"   Fetch {i+1:>2}: {ms:>6.0f}ms    "
                  f"({frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"   Fetch {i+1:>2}: Empty frame  ")

    except Exception as e:
        print(f"   Fetch {i+1:>2}: Failed   ({e})")

# ── Summary ──
print(f"\n Results:")
print(f"   Success Rate : {passed}/10  ({passed*10}%)")

if times:
    avg_ms  = sum(times) / len(times)
    min_ms  = min(times)
    max_ms  = max(times)
    est_fps = 1000 / avg_ms

    print(f"   Avg Fetch    : {avg_ms:.0f}ms")
    print(f"   Fastest      : {min_ms:.0f}ms")
    print(f"   Slowest      : {max_ms:.0f}ms")
    print(f"   Est. FPS     : {est_fps:.1f}")

    print(f"\n Recommendation:")
    if est_fps >= 10:
        print(f"    Excellent! ({est_fps:.0f} FPS) — "
              f"You can increase FRAME_WIDTH to 640 in config.py")
    elif est_fps >= 5:
        print(f"    Good ({est_fps:.0f} FPS) — "
              f"Current settings are ideal")
    elif est_fps >= 2:
        print(f"    Slow ({est_fps:.0f} FPS) — "
              f"Move phone closer to Wi-Fi router")
    else:
        print(f"    Very slow ({est_fps:.1f} FPS) — "
              f"Check Wi-Fi signal strength")
else:
    print(f"\n    All fetches failed. Check:")
    print(f"      → IP Webcam running on phone?")
    print(f"      → Phone and laptop same Wi-Fi?")
    print(f"      → Correct IP in config.py?")
    print(f"      → Try opening in browser: {SNAPSHOT_URL}")
