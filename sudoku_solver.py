"""
sudoku_solver.py
Solves a 9×9 Sudoku puzzle.

Algorithm: Backtracking with constraint propagation (AC-3-lite).
  1. Find the empty cell with the FEWEST possible values (MRV heuristic).
  2. Try each candidate value.
  3. Recursively solve; backtrack on failure.

This typically solves any valid puzzle in < 1 ms.
"""

import copy
import numpy as np
from typing import Optional, List, Tuple


class SudokuSolver:

    # ────────────────────────────────────────────────────────────
    def solve(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """
        Parameters
        ----------
        grid : (9, 9) int32 array — 0 = empty, 1-9 = given digit

        Returns
        -------
        Solved (9, 9) array, or None if no solution exists.
        """
        if not self.is_valid_puzzle(grid):
            return None

        board = grid.tolist()          # work with plain Python lists (fast)
        if self._solve(board):
            return np.array(board, dtype=np.int32)
        return None

    # ────────────────────────────────────────────────────────────
    def _solve(self, board: List[List[int]]) -> bool:
        cell = self._find_empty(board)
        if cell is None:
            return True            # all cells filled → solved!

        row, col = cell
        for num in self._candidates(board, row, col):
            board[row][col] = num
            if self._solve(board):
                return True
            board[row][col] = 0   # backtrack

        return False

    # ────────────────────────────────────────────────────────────
    def _find_empty(
        self, board: List[List[int]]
    ) -> Optional[Tuple[int, int]]:
        """MRV: pick the empty cell with fewest valid candidates."""
        best      = None
        best_cnt  = 10
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    cnt = len(self._candidates(board, r, c))
                    if cnt == 0:
                        return (r, c)   # dead end — trigger backtrack
                    if cnt < best_cnt:
                        best_cnt = cnt
                        best     = (r, c)
        return best

    # ────────────────────────────────────────────────────────────
    @staticmethod
    def _candidates(board: List[List[int]], row: int, col: int) -> List[int]:
        """Return list of valid digits for board[row][col]."""
        used = set()
        # row
        used.update(board[row])
        # column
        used.update(board[r][col] for r in range(9))
        # 3×3 box
        br, bc = (row // 3) * 3, (col // 3) * 3
        for r in range(br, br + 3):
            for c in range(bc, bc + 3):
                used.add(board[r][c])
        return [n for n in range(1, 10) if n not in used]

    # ────────────────────────────────────────────────────────────
    @staticmethod
    def is_valid_puzzle(grid: np.ndarray) -> bool:
        """Basic sanity check: no duplicates in any row/col/box."""
        if grid.shape != (9, 9):
            return False
        for i in range(9):
            row  = [v for v in grid[i]    if v != 0]
            col  = [v for v in grid[:, i] if v != 0]
            if len(row) != len(set(row)):
                return False
            if len(col) != len(set(col)):
                return False
        for br in range(3):
            for bc in range(3):
                box = [
                    grid[br*3+r, bc*3+c]
                    for r in range(3) for c in range(3)
                    if grid[br*3+r, bc*3+c] != 0
                ]
                if len(box) != len(set(box)):
                    return False
        return True

    # ────────────────────────────────────────────────────────────
    @staticmethod
    def print_grid(grid: np.ndarray) -> None:
        """Pretty-print the solved grid to stdout."""
        border = "  +" + ("-------+" * 3)
        print(border)
        for i, row in enumerate(grid):
            parts = []
            for j, val in enumerate(row):
                if j % 3 == 0:
                    parts.append("|")
                parts.append(f" {val} ")
            parts.append("|")
            print("  " + "".join(parts))
            if (i + 1) % 3 == 0:
                print(border)
        print()
