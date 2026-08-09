# test_rpca.py
# Test the RPCA algorithm on our captured matrix

import numpy as np
import matplotlib.pyplot as plt
from rpca import RobustPCA
from frame_collector import load_matrix
from config import FRAME_WIDTH, FRAME_HEIGHT

# ----------------------------
# LOAD THE SAVED MATRIX
# ----------------------------
print(" Loading saved matrix...")
M = load_matrix("matrix_M.npy")

if M is None:
    print(" matrix_M.npy not found. Run test_capture.py first.")
    exit()

# ----------------------------
# (OPTIONAL) USE SMALLER MATRIX FOR QUICK TEST
# ----------------------------
# If RPCA is too slow, use only first 50 frames
# Comment this block out for full run
USE_QUICK_TEST = True
if USE_QUICK_TEST:
    M = M[:, :50]
    print(f" Quick test mode: using first 50 frames → Shape: {M.shape}")

# ----------------------------
# RUN RPCA
# ----------------------------
rpca    = RobustPCA(max_iter=500)
L, S    = rpca.fit(M)

# ----------------------------
# SHOW RESULTS
# ----------------------------
print("\n Showing Results...")

# Pick frame number 25 to visualize
frame_idx = 24

# Reshape columns back into images
original   = M[:, frame_idx].reshape(FRAME_HEIGHT, FRAME_WIDTH)
background = L[:, frame_idx].reshape(FRAME_HEIGHT, FRAME_WIDTH)
foreground = S[:, frame_idx].reshape(FRAME_HEIGHT, FRAME_WIDTH)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"RPCA Result — Frame {frame_idx + 1}", fontsize=14)

axes[0].imshow(original,   cmap='gray')
axes[0].set_title("Original (M)")
axes[0].axis('off')

axes[1].imshow(background, cmap='gray')
axes[1].set_title("Background — L (Low Rank)")
axes[1].axis('off')

axes[2].imshow(np.abs(foreground), cmap='hot')
axes[2].set_title("Foreground — S (Sparse)")
axes[2].axis('off')

plt.tight_layout()
plt.show()

# ----------------------------
# SAVED L and S FOR NEXT 
# ----------------------------
np.save("matrix_L.npy", L)
np.save("matrix_S.npy", S)
print("\n Saved matrix_L.npy and matrix_S.npy")
print(" Step 3 Complete!")
