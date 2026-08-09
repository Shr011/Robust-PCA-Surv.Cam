# camera.py
# Optimized phone camera with threaded snapshot fetching

import cv2
import numpy as np
import urllib.request
import threading
import time
import collections
from config import (PHONE_STREAM_URL, SNAPSHOT_URL,
                    FRAME_WIDTH, FRAME_HEIGHT, USE_SNAPSHOT_MODE)


class PhoneCamera:
    """
    High-performance phone camera handler.
    - Threaded snapshot fetching for higher FPS
    - Auto-reconnect on Wi-Fi drops
    - Connection quality monitoring
    - Adaptive fetch speed
    """

    def __init__(self, buffer_size=5):
        self.cap             = None
        self.snapshot_mode   = USE_SNAPSHOT_MODE
        self.connected       = False

        # Thread-safe frame buffer (keeps last N frames)
        self._buffer         = collections.deque(maxlen=buffer_size)
        self._buffer_lock    = threading.Lock()

        # Fetch thread control
        self._fetch_thread   = None
        self._stop_event     = threading.Event()

        # Connection quality stats
        self._total_attempts  = 0
        self._total_success   = 0
        self._last_frame_time = None
        self._fetch_times     = collections.deque(maxlen=30)  # Last 30 fetch times
        self._consecutive_fails = 0

    # -------------------------------------------------------
    # CONNECTION QUALITY STATS
    # -------------------------------------------------------
    @property
    def success_rate(self):
        if self._total_attempts == 0:
            return 0.0
        return (self._total_success / self._total_attempts) * 100

    @property
    def avg_fetch_ms(self):
        if not self._fetch_times:
            return 0.0
        return sum(self._fetch_times) / len(self._fetch_times) * 1000

    @property
    def estimated_fps(self):
        if self.avg_fetch_ms == 0:
            return 0.0
        return 1000.0 / self.avg_fetch_ms

    # -------------------------------------------------------
    # CONNECT
    # -------------------------------------------------------
    def connect(self, retries=3):
        print(f"\n Connecting to phone camera...")

        if self.snapshot_mode:
            print(f"   Mode : Snapshot (threaded)")
            print(f"   URL  : {SNAPSHOT_URL}")
            result = self._connect_snapshot()
        else:
            print(f"   Mode : MJPEG Stream")
            result = self._connect_stream(retries)
            if not result:
                print(f"\n  Stream failed  switching to snapshot mode...")
                self.snapshot_mode = True
                result = self._connect_snapshot()

        if result:
            # Start background fetch thread
            self._start_fetch_thread()

        return result

    def _connect_snapshot(self):
        for attempt in range(1, 4):
            try:
                print(f"   Testing connection... attempt {attempt}/3")
                t_start   = time.time()
                response  = urllib.request.urlopen(SNAPSHOT_URL, timeout=5)
                url_bytes = response.read()
                t_end     = time.time()

                arr   = np.frombuffer(url_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    fetch_ms = (t_end - t_start) * 1000
                    print(f" Connected!")
                    print(f"   Resolution : {frame.shape[1]}x{frame.shape[0]}")
                    print(f"   Fetch time : {fetch_ms:.0f}ms")
                    print(f"   Est. FPS   : {1000/fetch_ms:.1f}")
                    self.connected = True
                    return True

            except urllib.error.URLError:
                print(f"    Cannot reach phone.")
            except Exception as e:
                print(f"    {e}")
            time.sleep(1)

        self._print_connection_help()
        return False

    def _connect_stream(self, retries):
        for attempt in range(1, retries + 1):
            try:
                print(f"   Attempt {attempt}/{retries}...")
                self.cap = cv2.VideoCapture(PHONE_STREAM_URL)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f" MJPEG stream connected!")
                        self.connected = True
                        return True
                    self.cap.release()
            except Exception:
                if self.cap:
                    self.cap.release()
            time.sleep(1)
        return False

    def _print_connection_help(self):
        print(f"\n Could not connect. Check:")
        print(f"   1. IP Webcam app is running on phone")
        print(f"   2. Phone and laptop on same Wi-Fi network")
        print(f"   3. Open in browser → {SNAPSHOT_URL}")
        print(f"   4. Update IP in config.py if it changed")

    # -------------------------------------------------------
    # BACKGROUND FETCH THREAD
    # -------------------------------------------------------
    def _start_fetch_thread(self):
        """Starts the background thread that continuously fetches frames."""
        self._stop_event.clear()
        self._fetch_thread = threading.Thread(
            target=self._fetch_loop,
            daemon=True,
            name="CameraFetchThread"
        )
        self._fetch_thread.start()
        print(f" Frame fetch thread started.")

    def _fetch_loop(self):
        """
        Runs in background thread.
        Continuously fetches frames and puts them in buffer.
        """
        while not self._stop_event.is_set():
            t_start = time.time()

            if self.snapshot_mode:
                success, frame = self._fetch_snapshot()
            else:
                success, frame = self._fetch_stream()

            t_end = time.time()
            fetch_time = t_end - t_start

            self._total_attempts += 1

            if success and frame is not None:
                self._total_success     += 1
                self._consecutive_fails  = 0
                self._last_frame_time    = time.time()

                # Add to thread-safe buffer
                with self._buffer_lock:
                    self._buffer.append(frame)

                self._fetch_times.append(fetch_time)

            else:
                self._consecutive_fails += 1

                # Too many failures → try reconnecting
                if self._consecutive_fails >= 10:
                    print(f"\n {self._consecutive_fails} consecutive failures.")
                    print(f"   Attempting reconnect...")
                    self._attempt_reconnect()

            # Small sleep to prevent hammering the phone CPU
            # Adjust based on measured fetch time
            sleep_time = max(0, 0.05 - fetch_time)
            time.sleep(sleep_time)

    def _fetch_snapshot(self):
        """Fetch single JPEG from phone."""
        try:
            response  = urllib.request.urlopen(SNAPSHOT_URL, timeout=3)
            url_bytes = response.read()
            arr       = np.frombuffer(url_bytes, dtype=np.uint8)
            frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return True, frame
        except Exception:
            pass
        return False, None

    def _fetch_stream(self):
        """Fetch from MJPEG stream."""
        try:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return True, frame
        except cv2.error:
            pass
        except Exception:
            pass
        return False, None

    def _attempt_reconnect(self):
        """Try to reconnect after failures."""
        self._consecutive_fails = 0

        for wait in [2, 5, 10]:
            print(f"   Retrying in {wait}s...")
            time.sleep(wait)

            if self.snapshot_mode:
                try:
                    response  = urllib.request.urlopen(SNAPSHOT_URL, timeout=5)
                    url_bytes = response.read()
                    arr       = np.frombuffer(url_bytes, dtype=np.uint8)
                    frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        print(f" Reconnected!")
                        return
                except Exception:
                    pass
            else:
                if self.cap:
                    self.cap.release()
                self.cap = cv2.VideoCapture(PHONE_STREAM_URL)
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f" Reconnected!")
                        return

        print(f" Reconnect failed. Check phone and Wi-Fi.")

    # -------------------------------------------------------
    # READ — main thread calls this
    # -------------------------------------------------------
    def read(self):
        """
        Returns the latest frame from the buffer.
        Non-blocking — returns immediately.
        """
        with self._buffer_lock:
            if self._buffer:
                return True, self._buffer[-1]   # Latest frame
        return False, None

    # -------------------------------------------------------
    # DIAGNOSTICS
    # -------------------------------------------------------
    def print_stats(self):
        print(f"\n Camera Stats:")
        print(f"   Mode          : {'Snapshot' if self.snapshot_mode else 'MJPEG'}")
        print(f"   Success Rate  : {self.success_rate:.1f}%")
        print(f"   Avg Fetch     : {self.avg_fetch_ms:.0f}ms")
        print(f"   Est. FPS      : {self.estimated_fps:.1f}")
        if self._last_frame_time:
            age = time.time() - self._last_frame_time
            print(f"   Last Frame    : {age:.2f}s ago")

    # -------------------------------------------------------
    # RELEASE
    # -------------------------------------------------------
    def release(self):
        print("\n Releasing camera...")
        self._stop_event.set()
        if self._fetch_thread and self._fetch_thread.is_alive():
            self._fetch_thread.join(timeout=3)
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.connected = False
        self.print_stats()
        print(" Camera released.")

    def isOpened(self):
        return self.connected
