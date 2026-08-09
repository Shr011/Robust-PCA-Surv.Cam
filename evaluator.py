# evaluator.py
# Tests RPCA accuracy and system performance

import numpy as np
import cv2
import time
import os
import json
from datetime import datetime
from rpca import RobustPCA
from separator import Separator
from profiler import Profiler
from config import (FRAME_WIDTH, FRAME_HEIGHT, LEARN_FRAMES,
                    MAX_ITER, TOLERANCE, N_COMPONENTS,
                    THRESHOLD, MIN_AREA)


class Evaluator:
    """
    Runs a full evaluation of the RPCA surveillance system.
    Tests speed, accuracy, and stability.
    Saves results to a JSON report.
    """

    def __init__(self):
        self.profiler  = Profiler()
        self.sep       = Separator(threshold=THRESHOLD,
                                   min_area=MIN_AREA)
        self.results   = {}

    # -------------------------------------------------------
    # TEST 1: RPCA SPEED TEST
    # -------------------------------------------------------
    def test_rpca_speed(self):
        """
        Tests how fast RPCA runs at different matrix sizes.
        """
        print(f"\n{'='*50}")
        print(f"    Test 1: RPCA Speed")
        print(f"{'='*50}")

        configs = [
            {"name": "Small  (160x120, 50 frames)",
             "w": 160,  "h": 120,  "f": 50},
            {"name": "Medium (320x240, 80 frames)",
             "w": 320,  "h": 240,  "f": 80},
            {"name": "Large  (640x480, 80 frames)",
             "w": 640,  "h": 480,  "f": 80},
        ]

        speed_results = []

        for cfg in configs:
            w, h, f = cfg["w"], cfg["h"], cfg["f"]
            pixels  = w * h

            print(f"\n  Testing: {cfg['name']}")
            print(f"  Matrix size: ({pixels} × {f})")

            # Create synthetic test matrix
            # Simulate background (low-rank) + moving object (sparse)
            np.random.seed(42)
            bg     = np.random.randn(pixels, 3)              # Low-rank background
            sparse = np.zeros((pixels, f))
            # Add sparse foreground (person in a small region)
            region = slice(pixels//4, pixels//4 + pixels//8)
            sparse[region, f//3:f//2] = np.random.randn(
                                            pixels//8, f//6) * 0.5
            L_true = bg @ np.random.randn(3, f) * 0.3
            M      = L_true + sparse

            t_start = time.time()
            rpca    = RobustPCA(max_iter=50, tol=1e-4,
                                n_components=min(10, f-1))
            L, S    = rpca.fit(M)
            t_end   = time.time()

            elapsed  = t_end - t_start
            recon    = np.linalg.norm(M - L - S, 'fro')
            norm_M   = np.linalg.norm(M, 'fro')
            accuracy = max(0, (1 - recon/norm_M)) * 100

            result = {
                "config"       : cfg["name"],
                "time_seconds" : round(elapsed, 2),
                "accuracy_pct" : round(accuracy, 2),
                "pixels"       : pixels,
                "frames"       : f
            }
            speed_results.append(result)

            status = "ok" if elapsed < 30 else "noo"
            print(f"  {status} Time     : {elapsed:.1f}s")
            print(f"   Accuracy : {accuracy:.1f}%")

        self.results["rpca_speed"] = speed_results
        return speed_results

    # -------------------------------------------------------
    # TEST 2: SEPARATOR ACCURACY
    # -------------------------------------------------------
    def test_separator(self):
        """
        Tests how well the separator finds moving objects.
        Creates known foreground and checks detection.
        """
        print(f"\n{'='*50}")
        print(f"    Test 2: Separator Accuracy")
        print(f"{'='*50}")

        pixels    = FRAME_WIDTH * FRAME_HEIGHT
        sep_results = []

        # Test at different threshold values
        thresholds = [0.03, 0.05, 0.08, 0.12, 0.15]

        print(f"\n  {'Threshold':>10} {'Sensitivity':>13}"
              f" {'Precision':>11} {'F1 Score':>10}")
        print(f"  {'─'*50}")

        for thresh in thresholds:

            # Create synthetic foreground signal
            S_frame  = np.zeros(pixels)
            true_fg  = np.zeros(pixels, dtype=bool)

            # Place a "person" in the center region
            cy, cx  = FRAME_HEIGHT // 2, FRAME_WIDTH // 2
            for y in range(cy - 40, cy + 40):
                for x in range(cx - 20, cx + 20):
                    if 0 <= y < FRAME_HEIGHT and 0 <= x < FRAME_WIDTH:
                        idx           = y * FRAME_WIDTH + x
                        S_frame[idx]  = 0.2    # Strong signal
                        true_fg[idx]  = True

            # Add background noise
            noise             = np.random.randn(pixels) * 0.02
            S_frame_noisy     = S_frame + noise

            # Run separator
            sep         = Separator(threshold=thresh, min_area=100)
            fg_mask     = sep.process_foreground(
                            S_frame_noisy, FRAME_HEIGHT, FRAME_WIDTH)

            pred_fg     = fg_mask > 0

            # Calculate metrics
            tp = np.sum( true_fg &  pred_fg)   # True positives
            fp = np.sum(~true_fg &  pred_fg)   # False positives
            fn = np.sum( true_fg & ~pred_fg)   # False negatives

            sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0
            precision   = tp / (tp + fp) if (tp+fp) > 0 else 0
            f1          = (2 * sensitivity * precision /
                          (sensitivity + precision)
                          if (sensitivity + precision) > 0 else 0)

            rating = " Best" if f1 > 0.7 else (" OK" if f1 > 0.4 else " Poor")

            print(f"  {thresh:>10.2f} {sensitivity:>12.1%}"
                  f" {precision:>11.1%} {f1:>9.1%}  {rating}")

            sep_results.append({
                "threshold"   : thresh,
                "sensitivity" : round(sensitivity, 3),
                "precision"   : round(precision, 3),
                "f1_score"    : round(f1, 3)
            })

        # Find best threshold
        best = max(sep_results, key=lambda x: x["f1_score"])
        print(f"\n   Best threshold for this scene: {best['threshold']}"
              f"  (F1={best['f1_score']:.1%})")
        print(f"     Update THRESHOLD = {best['threshold']} in config.py")

        self.results["separator"] = sep_results
        self.results["recommended_threshold"] = best["threshold"]
        return sep_results

    # -------------------------------------------------------
    # TEST 3: PIPELINE TIMING
    # -------------------------------------------------------
    def test_pipeline_timing(self):
        """
        Times each stage of the detection pipeline.
        """
        print(f"\n{'='*50}")
        print(f"    Test 3: Pipeline Timing")
        print(f"{'='*50}\n")

        pixels  = FRAME_WIDTH * FRAME_HEIGHT
        profiler = Profiler()

        # Simulate background
        np.random.seed(42)
        background = np.random.rand(pixels) * 0.5

        # Run 50 simulated frames
        for i in range(50):
            # Simulate frame capture
            profiler.start("1. Frame Resize")
            frame = np.random.randint(0, 255,
                        (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            profiler.stop("1. Frame Resize")

            # Simulate grayscale + normalize
            profiler.start("2. Preprocess")
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            norm = gray.astype(np.float64) / 255.0
            flat = norm.flatten()
            profiler.stop("2. Preprocess")

            # Background subtraction
            profiler.start("3. BG Subtraction")
            S_frame = flat - background
            profiler.stop("3. BG Subtraction")

            # Separator
            profiler.start("4. Separator")
            sep     = Separator(threshold=THRESHOLD, min_area=MIN_AREA)
            fg_mask = sep.process_foreground(
                          S_frame, FRAME_HEIGHT, FRAME_WIDTH)
            boxes   = sep.find_objects(fg_mask)
            profiler.stop("4. Separator")

            # Draw results
            profiler.start("5. Draw Results")
            annotated = sep.draw_results(gray, fg_mask, boxes)
            profiler.stop("5. Draw Results")

        total_ms = profiler.report()
        est_fps  = 1000 / total_ms if total_ms > 0 else 0

        print(f"   Estimated real-time FPS: {est_fps:.1f}")

        if est_fps >= 10:
            print(f"   Excellent! System runs smoothly.")
        elif est_fps >= 5:
            print(f"   Good. Acceptable for surveillance.")
        elif est_fps >= 2:
            print(f"   Slow. Consider reducing FRAME_WIDTH to 160.")
        else:
            print(f"   Too slow. Reduce frame size and LEARN_FRAMES.")

        self.results["pipeline_timing"] = {
            "total_ms_per_frame" : round(total_ms, 2),
            "estimated_fps"      : round(est_fps, 1)
        }
        return est_fps

    # -------------------------------------------------------
    # TEST 4: SYSTEM INFO
    # -------------------------------------------------------
    def collect_system_info(self):
        """Collects basic system information."""
        print(f"\n{'='*50}")
        print(f"    Test 4: System Info")
        print(f"{'='*50}\n")

        import platform
        import sys

        info = {
            "os"          : platform.system() + " " + platform.release(),
            "python"      : sys.version.split()[0],
            "numpy"       : np.__version__,
            "opencv"      : cv2.__version__,
            "frame_size"  : f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "learn_frames": LEARN_FRAMES,
            "threshold"   : THRESHOLD,
            "min_area"    : MIN_AREA
        }

        for k, v in info.items():
            print(f"  {k:<16}: {v}")

        self.results["system_info"] = info
        return info

    # -------------------------------------------------------
    # SAVE REPORT
    # -------------------------------------------------------
    def save_report(self):
        """Saves all results to a JSON file."""
        self.results["generated_at"] = datetime.now().strftime(
                                            "%Y-%m-%d %H:%M:%S")
        path = "evaluation_report.json"
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n Full report saved → {path}")
        return path

    # -------------------------------------------------------
    # RUN ALL TESTS
    # -------------------------------------------------------
    def run_all(self):
        print(f"\n{'#'*50}")
        print(f"    Robust PCA — Full Evaluation")
        print(f"{'#'*50}")

        self.collect_system_info()
        self.test_pipeline_timing()
        self.test_separator()
        self.test_rpca_speed()
        self.save_report()
        self.print_final_summary()

    # -------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------
    def print_final_summary(self):
        print(f"\n{'#'*50}")
        print(f"    Final Summary")
        print(f"{'#'*50}")

        # Pipeline FPS
        pt  = self.results.get("pipeline_timing", {})
        fps = pt.get("estimated_fps", 0)
        print(f"\n   Pipeline Speed  : {fps:.1f} FPS  "
              + ("ok" if fps >= 5 else "prob"))

        # Best threshold
        rt = self.results.get("recommended_threshold", THRESHOLD)
        print(f"   Best Threshold  : {rt}  "
              + ("ok" if rt == THRESHOLD
                 else f" (update config.py: THRESHOLD = {rt})"))

        # RPCA speed
        rs  = self.results.get("rpca_speed", [])
        med = next((r for r in rs
                    if "320x240" in r["config"]), None)
        if med:
            t = med["time_seconds"]
            print(f"   RPCA (320x240)  : {t:.1f}s  "
                  + ("ok" if t < 30 else " slow"))

        print(f"\n   Files generated:")
        print(f"     evaluation_report.json  ← detailed results")
        print(f"\n{'#'*50}\n")
