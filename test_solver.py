"""
test_solver.py
Pytest tests for the SudokuSolver class.
Run: pytest tests/test_solver.py -v
"""

import sys
import os
import numpy as np
import pytest

# Make sure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solver.sudoku_solver import SudokuSolver


# ── Fixtures ─────────────────────────────────────────────────────

# "Easy" puzzle from Project Euler #96
EASY_PUZZLE = np.array([
    [0, 0, 3,  0, 2, 0,  6, 0, 0],
    [9, 0, 0,  3, 0, 5,  0, 0, 1],
    [0, 0, 1,  8, 0, 6,  4, 0, 0],

    [0, 0, 8,  1, 0, 2,  9, 0, 0],
    [7, 0, 0,  0, 0, 0,  0, 0, 8],
    [0, 0, 6,  7, 0, 8,  2, 0, 0],

    [0, 0, 2,  6, 0, 9,  5, 0, 0],
    [8, 0, 0,  2, 0, 3,  0, 0, 9],
    [0, 0, 5,  0, 1, 0,  3, 0, 0],
], dtype=np.int32)

EASY_SOLUTION = np.array([
    [4, 8, 3,  9, 2, 1,  6, 5, 7],
    [9, 6, 7,  3, 4, 5,  8, 2, 1],
    [2, 5, 1,  8, 7, 6,  4, 9, 3],

    [5, 4, 8,  1, 3, 2,  9, 7, 6],
    [7, 2, 9,  5, 6, 4,  1, 3, 8],
    [1, 3, 6,  7, 9, 8,  2, 4, 5],

    [3, 7, 2,  6, 8, 9,  5, 1, 4],
    [8, 1, 4,  2, 5, 3,  7, 6, 9],
    [6, 9, 5,  4, 1, 7,  3, 8, 2],
], dtype=np.int32)

# "Hard" / near-minimal puzzle (17 clues)
HARD_PUZZLE = np.array([
    [8, 0, 0,  0, 0, 0,  0, 0, 0],
    [0, 0, 3,  6, 0, 0,  0, 0, 0],
    [0, 7, 0,  0, 9, 0,  2, 0, 0],

    [0, 5, 0,  0, 0, 7,  0, 0, 0],
    [0, 0, 0,  0, 4, 5,  7, 0, 0],
    [0, 0, 0,  1, 0, 0,  0, 3, 0],

    [0, 0, 1,  0, 0, 0,  0, 6, 8],
    [0, 0, 8,  5, 0, 0,  0, 1, 0],
    [0, 9, 0,  0, 0, 0,  4, 0, 0],
], dtype=np.int32)

# A puzzle with no solution (two 1s in same row)
INVALID_PUZZLE = np.array([
    [1, 1, 0,  0, 0, 0,  0, 0, 0],
    [0, 0, 0,  0, 0, 0,  0, 0, 0],
    [0, 0, 0,  0, 0, 0,  0, 0, 0],

    [0, 0, 0,  0, 0, 0,  0, 0, 0],
    [0, 0, 0,  0, 0, 0,  0, 0, 0],
    [0, 0, 0,  0, 0, 0,  0, 0, 0],

    [0, 0, 0,  0, 0, 0,  0, 0, 0],
    [0, 0, 0,  0, 0, 0,  0, 0, 0],
    [0, 0, 0,  0, 0, 0,  0, 0, 0],
], dtype=np.int32)

# Completely empty grid
EMPTY_PUZZLE = np.zeros((9, 9), dtype=np.int32)

# Already-solved grid
FULL_PUZZLE = EASY_SOLUTION.copy()


# ── Helper ────────────────────────────────────────────────────────
def is_valid_solution(grid: np.ndarray) -> bool:
    digits = set(range(1, 10))
    for i in range(9):
        if set(grid[i])        != digits: return False
        if set(grid[:, i])     != digits: return False
    for br in range(3):
        for bc in range(3):
            box = set(grid[br*3:br*3+3, bc*3:bc*3+3].flatten())
            if box != digits: return False
    return True


# ── Tests ─────────────────────────────────────────────────────────

class TestSolverEasy:
    def setup_method(self):
        self.solver = SudokuSolver()

    def test_returns_ndarray(self):
        result = self.solver.solve(EASY_PUZZLE.copy())
        assert isinstance(result, np.ndarray)

    def test_shape(self):
        result = self.solver.solve(EASY_PUZZLE.copy())
        assert result.shape == (9, 9)

    def test_correct_solution(self):
        result = self.solver.solve(EASY_PUZZLE.copy())
        np.testing.assert_array_equal(result, EASY_SOLUTION)

    def test_solution_is_valid(self):
        result = self.solver.solve(EASY_PUZZLE.copy())
        assert is_valid_solution(result)

    def test_givens_preserved(self):
        result = self.solver.solve(EASY_PUZZLE.copy())
        for r in range(9):
            for c in range(9):
                if EASY_PUZZLE[r, c] != 0:
                    assert result[r, c] == EASY_PUZZLE[r, c]


class TestSolverHard:
    def setup_method(self):
        self.solver = SudokuSolver()

    def test_hard_puzzle_solvable(self):
        result = self.solver.solve(HARD_PUZZLE.copy())
        assert result is not None

    def test_hard_solution_valid(self):
        result = self.solver.solve(HARD_PUZZLE.copy())
        assert is_valid_solution(result)


class TestSolverEdgeCases:
    def setup_method(self):
        self.solver = SudokuSolver()

    def test_invalid_puzzle_returns_none(self):
        result = self.solver.solve(INVALID_PUZZLE.copy())
        assert result is None

    def test_empty_puzzle_returns_valid_solution(self):
        result = self.solver.solve(EMPTY_PUZZLE.copy())
        assert result is not None
        assert is_valid_solution(result)

    def test_already_solved_returns_same(self):
        result = self.solver.solve(FULL_PUZZLE.copy())
        np.testing.assert_array_equal(result, FULL_PUZZLE)

    def test_wrong_shape_returns_none(self):
        bad = np.zeros((8, 8), dtype=np.int32)
        result = self.solver.solve(bad)
        assert result is None


class TestValidPuzzle:
    def setup_method(self):
        self.solver = SudokuSolver()

    def test_valid_easy(self):
        assert self.solver.is_valid_puzzle(EASY_PUZZLE)

    def test_valid_hard(self):
        assert self.solver.is_valid_puzzle(HARD_PUZZLE)

    def test_invalid_row_duplicate(self):
        assert not self.solver.is_valid_puzzle(INVALID_PUZZLE)

    def test_invalid_col_duplicate(self):
        bad = np.zeros((9, 9), dtype=np.int32)
        bad[0, 0] = 5
        bad[1, 0] = 5
        assert not self.solver.is_valid_puzzle(bad)

    def test_wrong_shape(self):
        assert not self.solver.is_valid_puzzle(np.zeros((8, 9), dtype=np.int32))


class TestPrintGrid:
    """Smoke test — just make sure print_grid doesn't raise."""
    def test_print_does_not_raise(self, capsys):
        SudokuSolver.print_grid(EASY_SOLUTION)
        captured = capsys.readouterr()
        assert "1" in captured.out
        assert "9" in captured.out
