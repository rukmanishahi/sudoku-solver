"""
grid_detector.py
Detects a 9×9 Sudoku grid in a BGR frame, returns the four corner points
and a perspective-corrected top-down view of the grid.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class GridDetector:
    """Detects the largest square-ish quadrilateral in a video frame."""

    # Warped output size (pixels)
    WARP_SIZE = 450

    def __init__(self) -> None:
        self.last_thresh: Optional[np.ndarray] = None

    # ────────────────────────────────────────────────────────────
    def detect(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parameters
        ----------
        frame : BGR image from webcam

        Returns
        -------
        corners : (4, 1, 2) int32 array in [TL, TR, BR, BL] order, or None
        warped  : (WARP_SIZE, WARP_SIZE) grayscale top-down image, or None
        """
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # Adaptive threshold works well under variable lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=57, C=5,
        )
        self.last_thresh = thresh.copy()

        # Dilate slightly to close small gaps in grid lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None

        # Pick the largest contour that looks like a quadrilateral
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        grid_cnt = None
        for cnt in contours[:5]:
            area = cv2.contourArea(cnt)
            if area < 40_000:           # too small — skip
                break
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                grid_cnt = approx
                break

        if grid_cnt is None:
            return None, None

        corners = self._order_corners(grid_cnt)
        warped  = self._perspective_transform(gray, corners)
        return corners.reshape(4, 1, 2).astype(np.int32), warped

    # ────────────────────────────────────────────────────────────
    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """Return corners in TL → TR → BR → BL order."""
        pts = pts.reshape(4, 2).astype(np.float32)
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        ordered[0] = pts[np.argmin(s)]   # TL  smallest sum
        ordered[2] = pts[np.argmax(s)]   # BR  largest sum
        d = np.diff(pts, axis=1)
        ordered[1] = pts[np.argmin(d)]   # TR  smallest diff
        ordered[3] = pts[np.argmax(d)]   # BL  largest diff
        return ordered

    # ────────────────────────────────────────────────────────────
    def _perspective_transform(
        self, gray: np.ndarray, corners: np.ndarray
    ) -> np.ndarray:
        """Warp the grid region to a square top-down view."""
        s = self.WARP_SIZE
        dst = np.array(
            [[0, 0], [s - 1, 0], [s - 1, s - 1], [0, s - 1]],
            dtype=np.float32,
        )
        M      = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(gray, M, (s, s))
        return warped
