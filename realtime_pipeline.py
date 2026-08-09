# realtime_pipeline.py

import cv2
import numpy as np
import threading
import time
from rpca import RobustPCA
from separator import Separator
from anomaly_detector import AnomalyDetector
from camera import PhoneCamera
from config import (FRAME_WIDTH, FRAME_HEIGHT,
                    LEARN_FRAMES, MAX_ITER, TOLERANCE,
                    N_COMPONENTS, THRESHOLD, MIN_AREA, REFRESH_MINS)


class RealTimePipeline:

    def __init__(self, learn_frames=None, threshold=None,
                 min_area=None, refresh_mins=None):

        self.learn_frames   = learn_frames or LEARN_FRAMES
        self.threshold      = threshold    or THRESHOLD
        self.min_area       = min_area     or MIN_AREA
        self.refresh_secs   = (refresh_mins or REFRESH_MINS) * 60

        self.background     = None
        self.is_learning    = False
        self.learn_progress = 0
        self.last_refresh   = None
        self.status_msg     = "Starting..."

        self.sep = Separator(threshold=self.threshold,
                             min_area=self.min_area)

        # ── Anomaly Detector ──
        self.detector = AnomalyDetector(
            loiter_seconds   = 10,
            cooldown_seconds = 20,
            save_screenshots = True,
            log_file         = "detection_log.csv",
            screenshot_dir   = "detections"
        )

        # Add detection zones
        #  Full frame zone — change these to monitor specific areas
        self.detector.add_zone(
     "Full Frame", 0, 0, FRAME_WIDTH, FRAME_HEIGHT,
    color=(0, 200, 255)
)

        self.frame_count    = 0
        self.fps_start      = time.time()
        self.current_fps    = 0.0

    # -------------------------------------------------------
    # LOADING SCREEN
    # -------------------------------------------------------
    def _show_loading(self, cam):
        while self.is_learning:
            ret, frame = cam.read()
            display = (cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                       if ret and frame is not None
                       else np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3),
                                     dtype=np.uint8))
            overlay  = (display * 0.4).astype(np.uint8)

            bx, by   = 40, FRAME_HEIGHT // 2
            bw, bh   = FRAME_WIDTH - 80, 30
            fill     = int(bw * self.learn_progress / 100)

            cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh), (50,50,50), -1)
            cv2.rectangle(overlay, (bx, by), (bx+fill, by+bh), (0,200,0), -1)
            cv2.putText(overlay,
                        f"Learning Background: {self.learn_progress:.0f}%",
                        (bx, by-12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255,255,255), 2)
            cv2.putText(overlay, self.status_msg,
                        (bx, by+bh+28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180,180,180), 1)
            cv2.putText(overlay, "Please wait — do not close this window",
                        (bx, by+bh+55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (120,120,120), 1)
            cv2.imshow("Robust PCA Surveillance", overlay)
            cv2.waitKey(30)

    # -------------------------------------------------------
    # LEARNING THREAD
    # -------------------------------------------------------
    def _learn_thread(self, cam):
        self.is_learning    = True
        self.learn_progress = 0
        self.status_msg     = "Capturing frames..."
        frames              = []
        fail_streak         = 0

        while len(frames) < self.learn_frames:
            try:
                ret, frame = cam.read()
            except Exception:
                ret, frame = False, None

            if not ret or frame is None:
                fail_streak += 1
                if fail_streak > 30:
                    print(" Too many read failures during learning.")
                    fail_streak = 0
                time.sleep(0.05)
                continue

            fail_streak   = 0
            fr            = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            fg            = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            fn            = fg.astype(np.float64) / 255.0
            frames.append(fn.flatten())
            self.learn_progress = (len(frames) / self.learn_frames) * 50

        self.status_msg = "Running RPCA..."
        M    = np.column_stack(frames)
        rpca = RobustPCA(max_iter=MAX_ITER, tol=TOLERANCE,
                         n_components=N_COMPONENTS)

        def cb(it, err):
            self.learn_progress = 50 + min(it/MAX_ITER, 1.0) * 50
            self.status_msg     = f"RPCA iter {it} | err {err:.5f}"

        L, _              = rpca.fit(M, progress_callback=cb)
        self.background   = np.mean(L, axis=1)
        self.last_refresh = time.time()
        self.learn_progress = 100
        self.status_msg   = "Done!"
        self.is_learning  = False
        print("\n Background learned! Live detection starting...\n")

    # -------------------------------------------------------
    # DETECT
    # -------------------------------------------------------
    def detect(self, frame):
        if self.background is None:
            return frame, None, []

        fr  = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        fg  = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        fn  = fg.astype(np.float64) / 255.0
        S   = fn.flatten() - self.background

        fg_mask   = self.sep.process_foreground(S, FRAME_HEIGHT, FRAME_WIDTH)
        boxes     = self.sep.find_objects(fg_mask)
        annotated = self.sep.draw_results(fg, fg_mask, boxes)

        return annotated, fg_mask, boxes

    # -------------------------------------------------------
    # FPS
    # -------------------------------------------------------
    def _update_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.fps_start
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start   = time.time()

    # -------------------------------------------------------
    # MAIN RUN
    # -------------------------------------------------------
    def run(self):
        print("=" * 50)
        print("    Robust PCA Surveillance System")
        print("=" * 50)

        cam = PhoneCamera()
        if not cam.connect():
            return

        cv2.namedWindow("Robust PCA Surveillance", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Robust PCA Surveillance", 960, 720)

        # Start learning
        t = threading.Thread(target=self._learn_thread,
                             args=(cam,), daemon=True)
        t.start()
        self._show_loading(cam)
        t.join()

        print("  Live detection started")
        print("   Q = Quit  |  R = Refresh  |  S = Save  |  Z = Zone editor\n")

        saved = 0

        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            # Detect
            annotated, fg_mask, boxes = self.detect(frame)
            self._update_fps()

            # Convert to BGR for display
            if len(annotated.shape) == 2:
                display = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
            else:
                display = annotated.copy()

            # Run anomaly detection
            display, alert = self.detector.process(display, boxes)

            # FPS overlay
            cv2.putText(display, f"FPS: {self.current_fps:.1f}",
                        (FRAME_WIDTH - 110, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            cv2.imshow("Robust PCA Surveillance", display)

            # Foreground mask window
            if fg_mask is not None:
                fg_col = cv2.applyColorMap(fg_mask, cv2.COLORMAP_HOT)
                fg_big = cv2.resize(fg_col, (640, 480))   # ← Resize mask window
                cv2.imshow("Foreground Mask", fg_big)
            # Auto refresh background
            if (not self.is_learning and self.last_refresh and
                    time.time() - self.last_refresh > self.refresh_secs):
                threading.Thread(target=self._learn_thread,
                                 args=(cam,), daemon=True).start()

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and not self.is_learning:
                print("\n Manual background refresh.")
                threading.Thread(target=self._learn_thread,
                                 args=(cam,), daemon=True).start()
            elif key == ord('s'):
                saved += 1
                fn = f"saved_frame_{saved:03d}.png"
                cv2.imwrite(fn, display)
                print(f" Saved: {fn}")

        cam.release()
        cv2.destroyAllWindows()
        self.detector.print_summary()
        print(" Stopped cleanly.")
