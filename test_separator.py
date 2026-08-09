# test_separator.py
# Visualize the full separation pipeline

import numpy as np
import matplotlib.pyplot as plt
import cv2
from separator import Separator
from config import FRAME_WIDTH, FRAME_HEIGHT

# ----------------------------
# LOAD SAVED MATRICES
# ----------------------------
print("📂 Loading matrices...")

try:
    M = np.load("matrix_M.npy")
    L = np.load("matrix_L.npy")
    S = np.load("matrix_S.npy")
    print(f"✅ M: {M.shape}  L: {L.shape}  S: {S.shape}")
except FileNotFoundError as e:
    print(f"❌ Missing file: {e}")
    print("   → Run test_capture.py and test_rpca.py first.")
    exit()

# ----------------------------
# SETUP SEPARATOR
# ----------------------------
sep = Separator(threshold=0.05, min_area=500)

# ----------------------------
# TEST ON MULTIPLE FRAMES
# ----------------------------
# Pick 3 frames spread across the captured frames
test_frames = [0, S.shape[1]//2, S.shape[1]-1]

fig, axes = plt.subplots(len(test_frames), 4, figsize=(18, 5 * len(test_frames)))
fig.suptitle("Separation Results: Original | Background | Foreground | Detected", fontsize=14)

for row, frame_idx in enumerate(test_frames):

    # Process background and foreground
    bg_img  = sep.process_background(L[:, frame_idx], FRAME_HEIGHT, FRAME_WIDTH)
    fg_mask = sep.process_foreground(S[:, frame_idx], FRAME_HEIGHT, FRAME_WIDTH)

    # Find objects
    boxes   = sep.find_objects(fg_mask)

    # Original frame
    original = (np.clip(M[:, frame_idx], 0, 1) * 255).astype(np.uint8)
    original = original.reshape(FRAME_HEIGHT, FRAME_WIDTH)

    # Draw results
    annotated = sep.draw_results(original, fg_mask, boxes)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Plot all 4 panels for this frame
    axes[row][0].imshow(original,      cmap='gray')
    axes[row][0].set_title(f"Frame {frame_idx+1} — Original")
    axes[row][0].axis('off')

    axes[row][1].imshow(bg_img,        cmap='gray')
    axes[row][1].set_title(f"Frame {frame_idx+1} — Background (L)")
    axes[row][1].axis('off')

    axes[row][2].imshow(fg_mask,       cmap='hot')
    axes[row][2].set_title(f"Frame {frame_idx+1} — Foreground Mask (S)")
    axes[row][2].axis('off')

    axes[row][3].imshow(annotated_rgb)
    axes[row][3].set_title(f"Frame {frame_idx+1} — Detected: {len(boxes)} object(s)")
    axes[row][3].axis('off')

    print(f"Frame {frame_idx+1:>3} → Objects detected: {len(boxes)}")
    for i, (x, y, w, h) in enumerate(boxes):
        print(f"         Box {i+1}: x={x}, y={y}, width={w}px, height={h}px")

plt.tight_layout()
plt.savefig("separation_results.png", dpi=100, bbox_inches='tight')
plt.show()

print("\n💾 Results saved to 'separation_results.png'")
print("✅ Step 4 Complete!")