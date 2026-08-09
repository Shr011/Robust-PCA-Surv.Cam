# anomaly_detector.py
# Smart anomaly detection with zones, loitering, alerts and logging

import cv2
import numpy as np
import time
import os
import csv
from datetime import datetime
from config import FRAME_WIDTH, FRAME_HEIGHT


class Zone:
    """
    A rectangular region of interest.
    Alerts are only triggered when objects enter this zone.
    """
    def __init__(self, name, x1, y1, x2, y2, color=(0, 0, 255)):
        self.name  = name
        self.x1    = x1
        self.y1    = y1
        self.x2    = x2
        self.y2    = y2
        self.color = color   # BGR color for drawing

    def contains(self, box):
        """Check if a bounding box overlaps with this zone."""
        bx, by, bw, bh = box
        bx2 = bx + bw
        by2 = by + bh
        # Check overlap
        return not (bx2 < self.x1 or bx > self.x2 or
                    by2 < self.y1 or by > self.y2)

    def draw(self, frame, active=False):
        """Draw this zone on the frame."""
        color     = (0, 255, 0) if active else self.color
        thickness = 2
        cv2.rectangle(frame, (self.x1, self.y1),
                              (self.x2, self.y2), color, thickness)
        cv2.putText(frame, self.name,
                    (self.x1 + 4, self.y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame


class TrackedObject:
    """
    Tracks a single detected object over time.
    Used for loitering detection.
    """
    def __init__(self, box, obj_id):
        self.obj_id      = obj_id
        self.box         = box
        self.first_seen  = time.time()
        self.last_seen   = time.time()
        self.zone_name   = None

    @property
    def duration(self):
        return time.time() - self.first_seen

    def update(self, box):
        self.box       = box
        self.last_seen = time.time()


class AnomalyDetector:
    """
    Full anomaly detection system.
    Handles zones, loitering, alerts, logging, screenshots.
    """

    def __init__(
        self,
        loiter_seconds   = 5,       # Alert if object stays this long
        cooldown_seconds = 10,      # Min seconds between repeated alerts
        save_screenshots = True,    # Auto-save on detection
        log_file         = "detection_log.csv",
        screenshot_dir   = "detections"
    ):
        self.loiter_seconds   = loiter_seconds
        self.cooldown_seconds = cooldown_seconds
        self.save_screenshots = save_screenshots
        self.log_file         = log_file
        self.screenshot_dir   = screenshot_dir

        # Zone list — add your custom zones
        self.zones = []

        # Object tracking
        self._tracked        = {}    # id
        self._next_id        = 1
        self._iou_threshold  = 0.3   # Match boxes across frames

        # Alert state
        self._last_alert_time  = 0
        self._alert_active     = False
        self._alert_start_time = 0
        self._alert_duration   = 1.5  # Seconds to show red flash
        self._alert_message    = ""

        # Stats
        self.total_detections  = 0
        self.total_alerts      = 0
        self.session_start     = time.time()

        # Setup files
        self._setup_files()

        # Add default full-frame zone if no zones defined
        self._using_full_frame = True

    # -------------------------------------------------------
    # SETUP
    # -------------------------------------------------------
    def _setup_files(self):
        """Create log file and screenshot directory."""
        os.makedirs(self.screenshot_dir, exist_ok=True)

        # Create CSV log with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "date", "time",
                    "event_type", "zone", "object_count",
                    "duration_seconds", "screenshot_path"
                ])
        print(f"📝 Log file   : {self.log_file}")
        print(f"📸 Screenshots: {self.screenshot_dir}/")

    # -------------------------------------------------------
    # ZONE MANAGEMENT
    # -------------------------------------------------------
    def add_zone(self, name, x1, y1, x2, y2, color=(0, 0, 255)):
        """
        Add a detection zone.
        Coordinates are in pixels relative to FRAME_WIDTH x FRAME_HEIGHT.

        Example:
            detector.add_zone("Entrance", 0, 0, 160, 240)
            detector.add_zone("Exit",   160, 0, 320, 240)
        """
        zone = Zone(name, x1, y1, x2, y2, color)
        self.zones.append(zone)
        self._using_full_frame = False
        print(f"🟥 Zone added: '{name}'  ({x1},{y1}) → ({x2},{y2})")
        return zone

    def add_full_frame_zone(self):
        """Monitor the entire frame as one zone."""
        self.add_zone("Full Frame", 0, 0,
                      FRAME_WIDTH, FRAME_HEIGHT, (0, 200, 255))

    # -------------------------------------------------------
    # OBJECT TRACKING (simple IoU-based)
    # -------------------------------------------------------
    def _iou(self, box1, box2):
        """Calculate Intersection over Union of two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Intersection
        ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
        iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
        intersection = ix * iy

        # Union
        union = w1*h1 + w2*h2 - intersection
        if union == 0:
            return 0.0
        return intersection / union

    def _update_tracking(self, boxes):
        """
        Match new boxes to existing tracked objects.
        Updates positions and marks lost objects.
        """
        matched_ids = set()

        for box in boxes:
            best_id    = None
            best_score = 0

            # Find best matching tracked object
            for obj_id, obj in self._tracked.items():
                score = self._iou(box, obj.box)
                if score > best_score and score > self._iou_threshold:
                    best_score = score
                    best_id    = obj_id

            if best_id is not None:
                # Update existing object
                self._tracked[best_id].update(box)
                matched_ids.add(best_id)
            else:
                # New object
                new_id = self._next_id
                self._next_id += 1
                self._tracked[new_id] = TrackedObject(box, new_id)
                matched_ids.add(new_id)

        # Remove objects not seen in last 2 seconds
        lost_ids = [
            oid for oid, obj in self._tracked.items()
            if time.time() - obj.last_seen > 2.0
        ]
        for oid in lost_ids:
            del self._tracked[oid]

        return matched_ids

    # -------------------------------------------------------
    # ALERT SYSTEM
    # -------------------------------------------------------
    def _trigger_alert(self, event_type, zone_name,
                       obj_count, duration, frame):
        """Fire an alert — log it, save screenshot, set visual flash."""
        now = time.time()

        # Cooldown check
        if now - self._last_alert_time < self.cooldown_seconds:
            return

        self._last_alert_time  = now
        self._alert_active     = True
        self._alert_start_time = now
        self._alert_message    = f"⚠ {event_type} in {zone_name}"
        self.total_alerts     += 1

        # Beep (Windows)
        try:
            import winsound
            winsound.Beep(1000, 300)   # 1000Hz for 300ms
        except Exception:
            pass   # Non-Windows — skip beep

        # Timestamp
        ts        = datetime.now()
        ts_str    = ts.strftime("%Y-%m-%d %H:%M:%S")
        date_str  = ts.strftime("%Y-%m-%d")
        time_str  = ts.strftime("%H:%M:%S")

        # Save screenshot
        screenshot_path = ""
        if self.save_screenshots and frame is not None:
            filename        = ts.strftime(f"{event_type}_%Y%m%d_%H%M%S.png")
            screenshot_path = os.path.join(self.screenshot_dir, filename)
            cv2.imwrite(screenshot_path, frame)

        # Log to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                ts_str, date_str, time_str,
                event_type, zone_name, obj_count,
                f"{duration:.1f}", screenshot_path
            ])

        print(f"🚨 [{time_str}] {event_type} | Zone: {zone_name} | "
              f"Objects: {obj_count} | Duration: {duration:.1f}s")

    # -------------------------------------------------------
    # MAIN PROCESS FUNCTION
    # -------------------------------------------------------
    def process(self, frame, boxes):
        """
        Main function — call this every frame.

        Input:
            frame  → Current BGR frame (for screenshots + display)
            boxes  → List of (x,y,w,h) bounding boxes from Separator

        Returns:
            annotated_frame → Frame with zones, alerts, dashboard drawn on it
            alert_fired     → True if an alert was triggered this frame
        """
        display     = frame.copy()
        alert_fired = False

        self.total_detections += len(boxes)

        # Update object tracking
        self._update_tracking(boxes)

        # Use full frame as zone if none defined
        active_zones = self.zones if self.zones else [
            Zone("Full Frame", 0, 0, FRAME_WIDTH, FRAME_HEIGHT, (0, 200, 255))
        ]

        # ── Check each zone ──
        for zone in active_zones:
            zone_boxes   = [b for b in boxes if zone.contains(b)]
            zone_active  = len(zone_boxes) > 0
            zone.draw(display, active=zone_active)

            if zone_active:
                # Find max duration of objects in this zone
                durations = []
                for obj in self._tracked.values():
                    if zone.contains(obj.box):
                        durations.append(obj.duration)

                max_duration = max(durations) if durations else 0

                # ── Loitering Alert ──
                if max_duration >= self.loiter_seconds:
                    self._trigger_alert(
                        "LOITERING", zone.name,
                        len(zone_boxes), max_duration, display
                    )
                    alert_fired = True

                # ── Intrusion Alert (any entry) ──
                elif len(zone_boxes) > 0:
                    self._trigger_alert(
                        "INTRUSION", zone.name,
                        len(zone_boxes), max_duration, display
                    )
                    alert_fired = True

        # ── Draw tracked objects with IDs and duration ──
        for obj_id, obj in self._tracked.items():
            x, y, w, h = obj.box
            dur         = obj.duration
            color       = (0, 255, 0) if dur < self.loiter_seconds \
                          else (0, 0, 255)

            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            label = f"ID:{obj_id}  {dur:.1f}s"
            cv2.putText(display, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # ── Red flash overlay on alert ──
        if self._alert_active:
            elapsed = time.time() - self._alert_start_time
            if elapsed < self._alert_duration:
                # Pulsing red overlay
                alpha    = 0.3 * (1 - elapsed / self._alert_duration)
                red_overlay        = display.copy()
                red_overlay[:, :]  = (0, 0, 200)
                display = cv2.addWeighted(display, 1-alpha,
                                          red_overlay, alpha, 0)
                # Alert message
                cv2.putText(display, self._alert_message,
                            (10, FRAME_HEIGHT - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
            else:
                self._alert_active = False

        # ── Dashboard ──
        display = self._draw_dashboard(display)

        return display, alert_fired

    # -------------------------------------------------------
    # DASHBOARD
    # -------------------------------------------------------
    def _draw_dashboard(self, frame):
        """Draw live stats panel on top-right of frame."""
        uptime    = time.time() - self.session_start
        h_up, rem = divmod(int(uptime), 3600)
        m_up, s_up = divmod(rem, 60)

        stats = [
            f"Uptime  : {h_up:02d}:{m_up:02d}:{s_up:02d}",
            f"Objects : {len(self._tracked)}",
            f"Alerts  : {self.total_alerts}",
            f"Zones   : {len(self.zones) or 1}",
        ]

        panel_x = FRAME_WIDTH - 175
        panel_y = 45
        panel_h = len(stats) * 20 + 10
        panel_w = 170

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (panel_x - 5, panel_y - 5),
                      (panel_x + panel_w, panel_y + panel_h),
                      (30, 30, 30), -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        for i, line in enumerate(stats):
            cv2.putText(frame, line,
                        (panel_x, panel_y + i * 20 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (200, 255, 200), 1)
        return frame

    # -------------------------------------------------------
    # SESSION SUMMARY
    # -------------------------------------------------------
    def print_summary(self):
        uptime = time.time() - self.session_start
        print(f"\n{'='*45}")
        print(f"    Session Summary")
        print(f"{'='*45}")
        print(f"  Uptime           : {uptime/60:.1f} minutes")
        print(f"  Total Detections : {self.total_detections}")
        print(f"  Total Alerts     : {self.total_alerts}")
        print(f"  Log File         : {self.log_file}")
        print(f"  Screenshots      : {self.screenshot_dir}/")
        print(f"{'='*45}\n")
