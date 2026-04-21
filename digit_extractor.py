"""
digit_extractor.py
Extracts digits from a warped Sudoku grid image.

Strategy:
  1. Split the 450×450 image into 81 cells.
  2. For each cell, threshold and find the largest centred blob.
  3. Scale the blob to 28×28 and classify with a tiny template-matching /
     contour-feature approach — NO external ML model required.
     Falls back to pytesseract if available for better accuracy.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List

# Try importing pytesseract; it's optional.
try:
    import pytesseract
    _TESS = True
except ImportError:
    _TESS = False


class DigitExtractor:
    """Splits a warped grid into 81 cells and recognises digits 1-9."""

    CELL = 50          # pixels per cell  (450 / 9)
    PAD  = 5           # inner padding to remove grid lines

    def __init__(self) -> None:
        self._tess_cfg = "--oem 3 --psm 10 -c tessedit_char_whitelist=123456789"

    # ────────────────────────────────────────────────────────────
    def extract(
        self, warped: np.ndarray
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """
        Parameters
        ----------
        warped : (450, 450) grayscale image

        Returns
        -------
        grid      : (9, 9) int array  0 = empty, 1-9 = digit
        cell_imgs : list of 81 cell images (for debugging)
        """
        # Ensure correct size
        warped = cv2.resize(warped, (450, 450))

        # Sharpen slightly
        warped = cv2.GaussianBlur(warped, (3, 3), 0)

        grid      = np.zeros((9, 9), dtype=np.int32)
        cell_imgs = []

        for row in range(9):
            for col in range(9):
                cell = self._get_cell(warped, row, col)
                cell_imgs.append(cell)
                digit = self._classify(cell)
                grid[row, col] = digit

        return grid, cell_imgs

    # ────────────────────────────────────────────────────────────
    def _get_cell(self, warped: np.ndarray, row: int, col: int) -> np.ndarray:
        y0 = row * self.CELL + self.PAD
        y1 = (row + 1) * self.CELL - self.PAD
        x0 = col * self.CELL + self.PAD
        x1 = (col + 1) * self.CELL - self.PAD
        return warped[y0:y1, x0:x1].copy()

    # ────────────────────────────────────────────────────────────
    def _classify(self, cell: np.ndarray) -> int:
        """Return digit 1-9 or 0 for empty."""
        # Threshold
        _, thresh = cv2.threshold(
            cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Check if cell is mostly empty
        filled_ratio = np.sum(thresh > 0) / thresh.size
        if filled_ratio < 0.03:
            return 0

        # Remove border noise: erode then check again
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.erode(thresh, kernel, iterations=1)
        if np.sum(cleaned > 0) / cleaned.size < 0.015:
            return 0

        # Try Tesseract first (more accurate)
        if _TESS:
            digit = self._tess_classify(cell)
            if digit != 0:
                return digit

        # Fallback: contour-based heuristic
        return self._contour_classify(thresh)

    # ── Tesseract path ───────────────────────────────────────────
    def _tess_classify(self, cell: np.ndarray) -> int:
        # Scale up for better OCR
        big = cv2.resize(cell, (84, 84), interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(
            big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # Add white border
        bordered = cv2.copyMakeBorder(
            thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
        )
        # Invert back: Tesseract expects dark text on light background
        bordered = cv2.bitwise_not(bordered)
        try:
            raw = pytesseract.image_to_string(bordered, config=self._tess_cfg).strip()
            if raw and raw[0].isdigit() and raw[0] != '0':
                return int(raw[0])
        except Exception:
            pass
        return 0

    # ── Contour heuristic path ───────────────────────────────────
    def _contour_classify(self, thresh: np.ndarray) -> int:
        """Very rough digit classification using contour geometry."""
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0

        # Use the largest contour
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 40:
            return 0

        # Bounding box aspect ratio
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)

        # Solidity
        hull    = cv2.convexHull(cnt)
        hull_a  = cv2.contourArea(hull)
        solid   = area / max(hull_a, 1)

        # Count holes (interior contours) — useful for 0, 4, 6, 8, 9
        _, hier = cv2.findContours(
            thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        holes = 0
        if hier is not None and len(hier[0]) > 0:
            for h_item in hier[0]:
                if h_item[3] != -1:   # has parent → it's a hole
                    holes += 1

        # Normalise position of centroid within cell
        M   = cv2.moments(cnt)
        if M["m00"] == 0:
            return 0
        cx  = M["m10"] / M["m00"]
        cy  = M["m01"] / M["m00"]
        cH, cW = thresh.shape
        rx  = cx / cW       # 0..1 left→right
        ry  = cy / cH       # 0..1 top→bottom

        # Heuristic rules (ordered by discriminability)
        # These are necessarily imprecise without a trained model.
        # Tesseract is strongly preferred for accuracy.
        if holes >= 2:
            return 8

        if holes == 1:
            if aspect < 0.55:
                return 4 if ry < 0.55 else 9
            return 6 if rx < 0.45 else 0

        # 0 holes
        if solid > 0.92 and aspect > 0.65:
            return 1 if aspect < 0.35 else 7

        if solid > 0.80:
            return 3 if aspect > 0.55 else 5

        if aspect < 0.45:
            return 1

        return 2   # default guess for remaining cases
