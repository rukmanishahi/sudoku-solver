#Run: py -3.14 main.py
import cv2
import numpy as np
import sys
import time
from vision.grid_detector import GridDetector
from vision.digit_extractor import DigitExtractor
from solver.sudoku_solver import SudokuSolver
from utils.image_utils import (
    overlay_solution,
    draw_status_bar,
    resize_with_aspect,
    stack_images,)
# colours for grid
COLOR_PINK= (136,  15, 232)   # #E80F88  (BGR)
COLOR_VIOLET= (204,   6, 248)   # #F806CC  (BGR)
COLOR_WHITE= (255, 255, 255)
COLOR_GREEN= ( 80, 220,  80)
COLOR_RED= ( 60,  60, 220)
def main() -> None:
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    detector= GridDetector()
    extractor= DigitExtractor()
    solver=SudokuSolver()

    #state 
    solved_grid= None
    last_puzzle= None
    last_warp= None
    last_corners= None
    solve_time= 0.0
    fps_timer= time.time()
    fps= 0
    frame_count= 0

    print("=" * 60)
    print("  Sudoku Solver — OpenCV Edition")
    print("  Press  Q  to quit")
    print("=" * 60)
    while True:
        ret, frame= cap.read()
        if not ret:
            print("[WARN] Dropped frame — retrying…")
            continue

        frame_count += 1
        if frame_count % 15 == 0:
            elapsed= time.time() - fps_timer
            fps= 15 / max(elapsed, 1e-6)
            fps_timer= time.time()

        display= frame.copy()

        #1. Detect grid
        corners, warped = detector.detect(frame)

        thresh_vis = None
        if warped is not None:
            thresh_vis= detector.last_thresh        # exposed by detector

            #2. Extract digits
            puzzle, cell_imgs= extractor.extract(warped)

            puzzle_changed= (last_puzzle is None or
                              not np.array_equal(puzzle, last_puzzle))

            if puzzle_changed and puzzle is not None:
                t0 = time.time()
                solution = solver.solve(puzzle.copy())
                solve_time = time.time() - t0

                if solution is not None:
                    solved_grid = solution
                    last_puzzle= puzzle.copy()
                    last_warp= warped.copy()
                    last_corners= corners.copy() if corners is not None else None

                    # Print to terminal
                    print("\n" + "─" * 37)
                    print(f"  Puzzle solved in {solve_time*1000:.1f} ms")
                    print("─" * 37)
                    solver.print_grid(solved_grid)
                else:
                    print("[WARN] No solution found for detected puzzle.")
                    solved_grid = None

            # 3. Overlay solution
            if solved_grid is not None and last_puzzle is not None:
                display = overlay_solution(
                    display,
                    last_corners if last_corners is not None else corners,
                    warped,
                    last_puzzle,
                    solved_grid,
                    color_new=COLOR_PINK,
                    color_given=COLOR_VIOLET,
                )

            # Draw contour on display
            if corners is not None:
                cv2.drawContours(display, [corners], -1, COLOR_VIOLET, 2)

        #HUD
        draw_status_bar(display, fps, solved_grid is not None, solve_time)
        #Build side panel
        h, w= display.shape[:2]
        panel_w= 320

        if warped is not None and thresh_vis is not None:
            warp_small= resize_with_aspect(warped,    panel_w, panel_w)
            thresh_small= resize_with_aspect(thresh_vis, panel_w, panel_w)
            if len(thresh_small.shape)== 2:
                thresh_small = cv2.cvtColor(thresh_small, cv2.COLOR_GRAY2BGR)

            # pad to same height
            def pad_h(img, target):
                dh= target - img.shape[0]
                if dh > 0:
                    img= cv2.copyMakeBorder(img, 0, dh, 0, 0,
                                             cv2.BORDER_CONSTANT, value=0)
                return img[:target]

            row_h= h // 2
            ws= pad_h(warp_small,   row_h)
            ts= pad_h(thresh_small, row_h)
            if len(ws.shape)== 2:
                ws= cv2.cvtColor(ws, cv2.COLOR_GRAY2BGR)
            if len(ts.shape)== 2:
                ts= cv2.cvtColor(ts, cv2.COLOR_GRAY2BGR)
            side_top= ws
            side_bot= ts
            side_top= ws
            side_bot= ts
            # labels
            cv2.putText(side_top, "Warped Grid",(8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_PINK, 1, cv2.LINE_AA)
            cv2.putText(side_bot, "Threshold",(8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_PINK, 1, cv2.LINE_AA)

            side_panel= np.vstack([side_top, side_bot])
            side_panel= pad_h(side_panel, h)
            composite= np.hstack([display, side_panel])
        else:
            # blank side panel
            side_panel= np.zeros((h, panel_w, 3), dtype=np.uint8)
            cv2.putText(side_panel, "Searching for",(10, h//2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PINK, 1)
            cv2.putText(side_panel, "Sudoku grid…",(10, h//2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PINK, 1)
            composite= np.hstack([display, side_panel])

        cv2.imshow("Sudoku Solver  [Q to quit]", composite)

        key=cv2.waitKey(1) & 0xFF
        if key== ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Exited cleanly.")


if __name__== "__main__":
    main()
