# config.py

# ── Phone Camera ──────────────────────────────────
# 🔁 Replace with YOUR phone's IP shown in IP Webcam
PHONE_STREAM_URL  = "http://192.168.1.5:8080/video"
SNAPSHOT_URL      = "http://192.168.1.5:8080/shot.jpg"
USE_SNAPSHOT_MODE = True        # True = stable, False = try MJPEG first

# ── Frame Settings ────────────────────────────────
FRAME_WIDTH       = 160         # Increase to 640 if you want higher detail
FRAME_HEIGHT      = 120         # Increase to 480 if you want higher detail
FPS_TARGET        = 10

# ── RPCA Settings ─────────────────────────────────
LEARN_FRAMES      = 100          # Frames to learn background (more = accurate)
MAX_ITER          = 100         # RPCA max iterations
TOLERANCE         = 1e-5        # RPCA convergence threshold
N_COMPONENTS      = 10          # SVD components (higher = more detail in BG)

# ── Detection Settings ────────────────────────────
THRESHOLD         = 0.12        # Movement sensitivity (lower = more sensitive)
MIN_AREA          = 1200         # Min object size in pixels
REFRESH_MINS      = 5           # Re-learn background every N minutes