# 🧩 Sudoku Solver — OpenCV Real-Time Edition

Detects a Sudoku grid from your **webcam**, extracts digits, solves the puzzle,
and overlays the solution back onto the live feed — all in real time.

---

## ✨ Features

| Feature | Detail |
|---|---|
| Grid detection | Adaptive threshold + contour analysis |
| Perspective correction | Homography warp to top-down view |
| Digit recognition | pytesseract OCR (+ contour fallback) |
| Solver | Backtracking + MRV heuristic (< 1 ms) |
| Overlay | Inverse-perspective digit rendering |
| Visual | Three-pane display with HUD |
| Colours | **#E80F88** / **#F806CC** brand palette |

---

## 📋 Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.14 (via `py` launcher) |
| Tesseract OCR | 5.x (recommended) |
| Webcam | Any USB/built-in |

---

## 🔧 Step-by-Step Setup in VS Code

### 1 — Install Tesseract (Windows)

> Tesseract is **optional but strongly recommended** for accurate digit recognition.
> Without it the solver falls back to a contour heuristic.

1. Download the **Tesseract installer** from:
   <https://github.com/UB-Mannheim/tesseract/wiki>
   (e.g. `tesseract-ocr-w64-setup-5.x.x.exe`)

2. Run the installer. **Note the install path** (default:
   `C:\Program Files\Tesseract-OCR`).

3. Add the path to your system `PATH`:
   - Open *Start → Edit the system environment variables*
   - Under *System Variables* → find `Path` → Edit → New
   - Add `C:\Program Files\Tesseract-OCR`
   - Click OK everywhere, then **restart VS Code**.

4. Verify: open a new terminal and run:
   ```
   tesseract --version
   ```

### 2 — Open the project in VS Code

```
File → Open Folder → select the `sudoku_solver` folder
```

### 3 — Create a virtual environment

Open the **VS Code Integrated Terminal** (`Ctrl + `` ` ``) and run:

```powershell
py -3.14 -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

> If you see a script execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then try activating again.

### 4 — Install dependencies

```powershell
pip install -r requirements.txt
```

### 5 — Select the interpreter in VS Code

- Press `Ctrl+Shift+P` → *Python: Select Interpreter*
- Choose `.venv\Scripts\python.exe` (the one inside your project folder)

### 6 — (Optional) Generate a sample image for testing

```powershell
py -3.14 generate_sample.py
```

This creates `assets/sample_sudoku.jpg` — you can hold it up to your webcam.

---

## ▶️ Running the Project

```powershell
py -3.14 main.py
```

Or via VS Code: press **F5** (uses the `Run Sudoku Solver` launch config).

### Controls

| Key | Action |
|---|---|
| `Q` or `Esc` | Quit |

---

## 🧪 Running Tests

```powershell
pytest tests/ -v
```

Expected output:
```
tests/test_solver.py::TestSolverEasy::test_returns_ndarray        PASSED
tests/test_solver.py::TestSolverEasy::test_shape                  PASSED
tests/test_solver.py::TestSolverEasy::test_correct_solution       PASSED
...
17 passed in 0.XXs
```

---

## 📁 Project Structure

```
sudoku_solver/
├── main.py                  # Entry point — webcam loop
├── generate_sample.py       # One-time helper to create a test image
│
├── vision/
│   ├── grid_detector.py     # Contour-based grid finder + warp
│   └── digit_extractor.py   # Per-cell digit classification
│
├── solver/
│   └── sudoku_solver.py     # Backtracking solver + validator
│
├── utils/
│   └── image_utils.py       # Drawing, overlay, HUD helpers
│
├── tests/
│   └── test_solver.py       # Pytest suite (17 tests)
│
├── assets/
│   └── sample_sudoku.jpg    # Generated sample (after generate_sample.py)
│
├── requirements.txt
├── README.md
└── .vscode/
    ├── settings.json
    └── launch.json
```

---

## 🎨 Colour Palette

| Colour | Hex | Use |
|---|---|---|
| Hot Pink | `#E80F88` | Solved digit overlay, UI accents |
| Violet | `#F806CC` | Grid border, given-digit hints |

---

## 🔍 How It Works

```
Webcam frame
    │
    ▼
GridDetector
  ├─ Grayscale + Gaussian blur
  ├─ Adaptive threshold (inverted)
  ├─ Dilate to close grid lines
  ├─ Find largest quadrilateral contour
  └─ Perspective warp → 450×450 top-down grid
    │
    ▼
DigitExtractor
  ├─ Split into 81 cells (9×9)
  ├─ Otsu threshold per cell
  ├─ Discard mostly-empty cells → 0
  └─ Tesseract OCR (or contour heuristic)
    │
    ▼
SudokuSolver
  ├─ Validate puzzle (no row/col/box conflicts)
  ├─ Backtracking with MRV heuristic
  └─ Return solved 9×9 grid
    │
    ▼
Overlay
  ├─ Draw solved digits onto warped canvas
  ├─ Inverse perspective transform back to frame
  └─ Alpha-blend onto original frame
```

---

## 🛠️ Troubleshooting

### Camera not opening
```
[ERROR] Cannot open webcam. Check your camera connection.
```
- Make sure no other app (Zoom, Teams, etc.) is using the camera.
- Try changing the camera index in `main.py`:
  ```python
  cap = cv2.VideoCapture(1)   # try 0, 1, 2 …
  ```

### pytesseract not found / ImportError
- The solver falls back to a contour-based heuristic automatically.
- For better accuracy, install Tesseract (see setup step 1).

### Grid not detected
- Ensure good lighting with no glare on the puzzle.
- Hold the puzzle flat and straight-on to the camera.
- Make sure the grid occupies at least ½ of the frame.

### `py` command not found
```powershell
py install --refresh
```
Or reinstall Python 3.14 from <https://www.python.org/downloads/> and check
*"Add to PATH"* during setup.

### ExecutionPolicy error when activating venv
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Tests fail with `ModuleNotFoundError`
Make sure you run pytest from the project root with the venv activated:
```powershell
cd sudoku_solver
.venv\Scripts\Activate.ps1
pytest tests/ -v
```

---

## 📄 Licence

MIT — free to use and modify.
