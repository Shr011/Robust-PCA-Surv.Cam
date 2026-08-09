# frame_collector.py

import cv2
import numpy as np
from config import PHONE_STREAM_URL, FRAME_WIDTH, FRAME_HEIGHT, FPS_TARGET

def capture_frames(num_frames=200):
    """
    Connects to phone camera and captures 'num_frames' frames.
    Returns a matrix M of shape (pixels, num_frames)
    """

    # Connect to phone camera
    cap = cv2.VideoCapture(PHONE_STREAM_URL)

    if not cap.isOpened():
        print("❌ Cannot connect to phone camera.")
        print("   → Make sure IP Webcam is running on your phone")
        print("   → Make sure phone and laptop are on same Wi-Fi")
        return None

    print(f"✅ Connected to phone camera!")
    print(f"📷 Capturing {num_frames} frames... Please wait.")
    print(f"   (Keep the camera still and pointed at the scene)")

    frames = []       # This list will store all our frames
    count  = 0        # Counter to track how many frames we captured

    while count < num_frames:

        ret, frame = cap.read()   # Read one frame from phone

        if not ret:
            print("⚠️ Missed a frame, retrying...")
            continue

        # Step 1: Resize frame to standard size
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Step 2: Convert to Grayscale (we don't need color for RPCA)
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Step 3: Normalize pixel values from 0-255 to 0.0-1.0
        frame_normalized = frame_gray.astype(np.float64) / 255.0

        # Step 4: Flatten 2D frame (480x640) into 1D column (307200,)
        frame_flat = frame_normalized.flatten()

        # Step 5: Add this column to our list
        frames.append(frame_flat)

        count += 1

        # Show live preview while capturing
        cv2.imshow(f"Capturing Frame {count}/{num_frames} - Press Q to cancel", frame_resized)

        # Allow cancel with Q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⛔ Capture cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(frames) == 0:
        print("❌ No frames were captured.")
        return None

    # Step 6: Stack all flat frames as COLUMNS into matrix M
    # Each frame_flat is a row in our list → transpose to make them columns
    M = np.column_stack(frames)

    print(f"\n✅ Matrix M created successfully!")
    print(f"   Shape : {M.shape}  → ({FRAME_WIDTH*FRAME_HEIGHT} pixels  x  {len(frames)} frames)")
    print(f"   Min value : {M.min():.4f}")
    print(f"   Max value : {M.max():.4f}")

    return M


def save_matrix(M, filename="matrix_M.npy"):
    """
    Saves matrix M to disk so we don't have to recapture every time.
    """
    np.save(filename, M)
    print(f"💾 Matrix saved to '{filename}'")


def load_matrix(filename="matrix_M.npy"):
    """
    Loads a previously saved matrix from disk.
    """
    import os
    if not os.path.exists(filename):
        print(f"❌ File '{filename}' not found. Capture frames first.")
        return None

    M = np.load(filename)
    print(f"📂 Matrix loaded from '{filename}'  →  Shape: {M.shape}")
    return M