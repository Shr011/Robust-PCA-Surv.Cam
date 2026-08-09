# separator.py
# Post-processes L and S matrices into clean background and foreground

import cv2
import numpy as np

class Separator:
    """
    Takes raw L and S matrices from RPCA and produces:
    - Clean background image
    - Clean foreground mask
    - Bounding boxes around detected objects
    """

    def __init__(self, threshold=0.05, min_area=500):
        """
        Parameters:
        -----------
        threshold : Minimum change value to count as foreground.
                    Range: 0.0 to 1.0
                    Lower = more sensitive (catches subtle movement)
                    Higher = less sensitive (ignores small changes)

        min_area  : Minimum pixel area for a detected object.
                    Smaller blobs below this size are ignored.
                    Increase if you get too many false detections.
        """
        self.threshold = threshold
        self.min_area  = min_area

    # --------------------------------------------------
    # PROCESS BACKGROUND (L matrix)
    # --------------------------------------------------
    def process_background(self, L_col, frame_height, frame_width):
        """
        Converts one column of L into a clean background image.
        """

        # Reshape column back into 2D image
        bg = L_col.reshape(frame_height, frame_width)

        # Clip values to valid range 0.0 - 1.0
        bg = np.clip(bg, 0, 1)

        # Convert to uint8 (0-255) for OpenCV display
        bg_uint8 = (bg * 255).astype(np.uint8)

        # Apply slight blur to smooth out artifacts
        bg_clean = cv2.GaussianBlur(bg_uint8, (5, 5), 0)

        return bg_clean

    # --------------------------------------------------
    # PROCESS FOREGROUND (S matrix)
    # --------------------------------------------------
    def process_foreground(self, S_col, frame_height, frame_width):
        """
        Converts one column of S into a clean foreground mask.
        """

        # Reshape column back into 2D image
        fg = S_col.reshape(frame_height, frame_width)

        # Step 1: Take absolute values
        # (we care about ANY change, not direction of change)
        fg_abs = np.abs(fg)

        # Step 2: Apply threshold
        # Pixels below threshold → 0 (background noise, ignore)
        # Pixels above threshold → 1 (real movement, keep)
        fg_thresh = (fg_abs > self.threshold).astype(np.uint8) * 255

        # Step 3: Morphological Opening
        # Removes small isolated noise dots (salt-and-pepper)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_opened   = cv2.morphologyEx(fg_thresh, cv2.MORPH_OPEN, kernel_open)

        # Step 4: Morphological Closing
        # Fills small holes inside detected objects
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_closed    = cv2.morphologyEx(fg_opened, cv2.MORPH_CLOSE, kernel_close)

        # Step 5: Dilation
        # Slightly expand detected regions to cover full object
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_dilated    = cv2.dilate(fg_closed, kernel_dilate, iterations=2)

        return fg_dilated

    # --------------------------------------------------
    # FIND BOUNDING BOXES AROUND DETECTED OBJECTS
    # --------------------------------------------------
    def find_objects(self, fg_mask):
        """
        Finds contours (outlines) of detected objects in foreground mask.
        Returns list of bounding boxes: [(x, y, w, h), ...]
        """

        # Find contours = outlines of white regions in mask
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,       # Only outer contours
            cv2.CHAIN_APPROX_SIMPLE  # Compress contour points
        )

        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Ignore tiny blobs smaller than min_area
            if area < self.min_area:
                continue

            # Get bounding rectangle around this contour
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

        return boxes

    # --------------------------------------------------
    # DRAW RESULTS ON FRAME
    # --------------------------------------------------
    def draw_results(self, original_frame, fg_mask, boxes):
        """
        Draws foreground overlay and bounding boxes on the original frame.
        Returns annotated frame for display.
        """

        # Convert grayscale original to color so we can draw colored boxes
        if len(original_frame.shape) == 2:
            display = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR)
        else:
            display = original_frame.copy()

        # Create red overlay for foreground regions
        overlay       = display.copy()
        red_mask      = np.zeros_like(display)
        red_mask[:,:,2] = fg_mask   # Red channel = foreground mask

        # Blend overlay with original (30% red, 70% original)
        display = cv2.addWeighted(display, 0.7, red_mask, 0.3, 0)

        # Draw green bounding boxes around each detected object
        for (x, y, w, h) in boxes:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"Object ({w}x{h}px)",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # Show object count on top left
        cv2.putText(
            display,
            f"Detected Objects: {len(boxes)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        return display