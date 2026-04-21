"""
generate_sample.py
Run once to create assets/sample_sudoku.jpg for offline testing.
Usage: py -3.14 generate_sample.py
"""

import cv2
import numpy as np
import os

PUZZLE = [
    [0, 0, 3,  0, 2, 0,  6, 0, 0],
    [9, 0, 0,  3, 0, 5,  0, 0, 1],
    [0, 0, 1,  8, 0, 6,  4, 0, 0],
    [0, 0, 8,  1, 0, 2,  9, 0, 0],
    [7, 0, 0,  0, 0, 0,  0, 0, 8],
    [0, 0, 6,  7, 0, 8,  2, 0, 0],
    [0, 0, 2,  6, 0, 9,  5, 0, 0],
    [8, 0, 0,  2, 0, 3,  0, 0, 9],
    [0, 0, 5,  0, 1, 0,  3, 0, 0],
]

def generate():
    size   = 450
    cell   = size // 9
    img    = np.ones((size, size, 3), dtype=np.uint8) * 255

    # Draw grid lines
    for i in range(10):
        thickness = 3 if i % 3 == 0 else 1
        color     = (0, 0, 0)
        cv2.line(img, (i * cell, 0), (i * cell, size), color, thickness)
        cv2.line(img, (0, i * cell), (size, i * cell), color, thickness)

    # Draw digits
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in range(9):
        for c in range(9):
            val = PUZZLE[r][c]
            if val != 0:
                x = c * cell + cell // 2 - 8
                y = r * cell + cell // 2 + 8
                cv2.putText(img, str(val), (x, y), font, 0.8,
                            (0, 0, 0), 2, cv2.LINE_AA)

    os.makedirs("assets", exist_ok=True)
    path = os.path.join("assets", "sample_sudoku.jpg")
    cv2.imwrite(path, img)
    print(f"[OK] Saved {path}")

if __name__ == "__main__":
    generate()
