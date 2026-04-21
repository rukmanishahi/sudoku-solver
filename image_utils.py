"""
image_utils.py
Drawing helpers, perspective overlays, and frame-composition utilities.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


# ── Colour constants (BGR) ───────────────────────────────────────
COLOR_PINK     = (136,  15, 232)   # #E80F88
COLOR_VIOLET   = (204,   6, 248)   # #F806CC
COLOR_WHITE    = (255, 255, 255)
COLOR_BLACK    = (  0,   0,   0)
COLOR_GREEN    = ( 80, 220,  80)
COLOR_DARK_BG  = ( 20,  20,  30)


# ────────────────────────────────────────────────────────────────
def overlay_solution(
    frame: np.ndarray,
    corners: Optional[np.ndarray],
    warped: np.ndarray,
    puzzle: np.ndarray,
    solution: np.ndarray,
    color_new:   Tuple[int, int, int] = COLOR_PINK,
    color_given: Tuple[int, int, int] = COLOR_VIOLET,
) -> np.ndarray:
    """
    Draw the solved digits onto the original frame using inverse perspective.

    Parameters
    ----------
    frame    : original BGR frame
    corners  : (4,1,2) corner points detected in `frame`
    warped   : 450×450 warped grid (grayscale)
    puzzle   : (9,9) original puzzle (0 = empty)
    solution : (9,9) solved grid
    """
    if corners is None:
        return frame

    h_warp = w_warp = 450
    cell   = h_warp // 9

    # Create a blank canvas the same size as warped
    digit_layer = np.zeros((h_warp, w_warp, 3), dtype=np.uint8)

    for row in range(9):
        for col in range(9):
            orig_val = puzzle[row, col]
            sol_val  = solution[row, col]

            if orig_val != 0:
                continue    # skip given digits (already visible)

            # Centre of this cell
            cx = col * cell + cell // 2
            cy = row * cell + cell // 2

            color = color_new
            text  = str(sol_val)

            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness  = 2

            tw, th = cv2.getTextSize(text, font, font_scale, thickness)[0]
            tx = cx - tw // 2
            ty = cy + th // 2

            # Shadow for legibility
            cv2.putText(digit_layer, text, (tx+1, ty+1),
                        font, font_scale, COLOR_BLACK, thickness + 1, cv2.LINE_AA)
            cv2.putText(digit_layer, text, (tx, ty),
                        font, font_scale, color, thickness, cv2.LINE_AA)

    # ── Inverse perspective transform ────────────────────────────
    pts_src = np.array(
        [[0, 0], [w_warp-1, 0], [w_warp-1, h_warp-1], [0, h_warp-1]],
        dtype=np.float32,
    )
    pts_dst = corners.reshape(4, 2).astype(np.float32)
    M_inv   = cv2.getPerspectiveTransform(pts_src, pts_dst)

    fh, fw  = frame.shape[:2]
    unwarped = cv2.warpPerspective(digit_layer, M_inv, (fw, fh))

    # Blend: only paint where we drew digits (non-black pixels)
    mask     = cv2.cvtColor(unwarped, cv2.COLOR_BGR2GRAY)
    _, mask  = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bg       = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg       = cv2.bitwise_and(unwarped, unwarped, mask=mask)
    result   = cv2.add(bg, fg)
    return result


# ────────────────────────────────────────────────────────────────
def draw_status_bar(
    frame: np.ndarray,
    fps: float,
    solved: bool,
    solve_ms: float,
) -> None:
    """Draw a translucent status bar at the bottom of `frame` (in-place)."""
    h, w = frame.shape[:2]
    bar_h = 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), COLOR_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    status = (
        f"FPS: {fps:.1f}   |   "
        + (f"SOLVED in {solve_ms*1000:.1f} ms" if solved else "Searching…")
    )
    cv2.putText(frame, status, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_PINK, 1, cv2.LINE_AA)

    # Small coloured dot indicator
    dot_color = COLOR_GREEN if solved else COLOR_PINK
    cv2.circle(frame, (w - 20, h - bar_h // 2), 7, dot_color, -1)


# ────────────────────────────────────────────────────────────────
def resize_with_aspect(
    img: np.ndarray, max_w: int, max_h: int
) -> np.ndarray:
    """Resize while preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


# ────────────────────────────────────────────────────────────────
def stack_images(
    img_list: list,
    cols: int = 2,
    scale: float = 1.0,
) -> np.ndarray:
    """Tile a list of images into a grid. All images resized to same dims."""
    if not img_list:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Ensure all are BGR
    bgr = []
    for im in img_list:
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        bgr.append(im)

    h0, w0 = bgr[0].shape[:2]
    h0 = int(h0 * scale)
    w0 = int(w0 * scale)
    bgr = [cv2.resize(im, (w0, h0)) for im in bgr]

    rows = []
    for i in range(0, len(bgr), cols):
        row_imgs = bgr[i:i+cols]
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros((h0, w0, 3), dtype=np.uint8))
        rows.append(np.hstack(row_imgs))
    return np.vstack(rows)
