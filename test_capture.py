# test_capture.py

from frame_collector import capture_frames, save_matrix, load_matrix
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# CAPTURE FRAMES FROM PHONE
# ----------------------------
M = capture_frames(num_frames=200)  # Capture 200 frames

if M is not None:

    # Save matrix to disk (don't have to recapture every run)
    save_matrix(M, "matrix_M.npy")

    # ----------------------------
    # VISUALIZE WHAT WE CAPTURED
    # ----------------------------

    print("\n Showing visualizations...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Frame Acquisition Results", fontsize=14)

    # Show Frame 1 (first column reshaped back to image)
    from config import FRAME_WIDTH, FRAME_HEIGHT
    frame1 = M[:, 0].reshape(FRAME_HEIGHT, FRAME_WIDTH)
    axes[0].imshow(frame1, cmap='gray')
    axes[0].set_title("Frame 1 (First Captured)")
    axes[0].axis('off')

    # Show Frame 100 (middle column)
    frame100 = M[:, 99].reshape(FRAME_HEIGHT, FRAME_WIDTH)
    axes[1].imshow(frame100, cmap='gray')
    axes[1].set_title("Frame 100 (Middle)")
    axes[1].axis('off')

    # Show Frame 200 (last column)
    frame200 = M[:, -1].reshape(FRAME_HEIGHT, FRAME_WIDTH)
    axes[2].imshow(frame200, cmap='gray')
    axes[2].set_title("Frame 200 (Last Captured)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # Show pixel intensity over time for one pixel (center of frame)
    center_pixel = M[FRAME_HEIGHT//2 * FRAME_WIDTH + FRAME_WIDTH//2, :]

    plt.figure(figsize=(12, 4))
    plt.plot(center_pixel)
    plt.title("Center Pixel Intensity Over 200 Frames")
    plt.xlabel("Frame Number")
    plt.ylabel("Pixel Intensity (0 to 1)")
    plt.grid(True)
    plt.show()

    print("\n  Complete! Matrix M is ready for RPCA.")
