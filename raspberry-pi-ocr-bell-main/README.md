Windows OCR Webcam — README

Quick overview
- Purpose: Use a webcam (or a photo) to detect red/pink seven-segment clock digits (e.g. "12:00") on Windows and report the detected time.
- Main script: `windows_ocr_webcam.py` (in this repository root).

Prerequisites
- Python 3.8+ installed on Windows.
- Tesseract OCR for Windows installed. Default path used by the script is `C:\Program Files\Tesseract-OCR\tesseract.exe`. If Tesseract is installed elsewhere, pass `--tesseract-path`.
- It is recommended to use a virtual environment to install Python dependencies.

Quick start (recommended)
1. Create and activate a virtualenv (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Run image-mode (useful for testing with a photo):

```powershell
python windows_ocr_webcam.py --image clock.jpg --show-preprocessed --debug-ocr --min-area 20 --save-on-detect
```

3. Run webcam mode:

```powershell
python windows_ocr_webcam.py --display --resize-factor 0.6 --frame-skip 1 --save-on-detect
```

Key command-line flags (summary)
- `--image <path>`: Run OCR on a single image file instead of opening the webcam.
- `--camera <index>`: Camera index (default 0).
- `--tesseract-path <path>`: Path to tesseract executable (Windows default is provided).
- `--min-area <int>`: Minimum contour area to consider (default 800). Lower to 20-50 for narrow digits.
- `--display`: Show annotated webcam video window.
- `--show-preprocessed`: Show preprocessed crops used for OCR (saves files `preprocessed_region_*.png`).
- `--debug-ocr`: Print raw OCR and seven-seg debug info to the console.
- `--resize-factor <float>`: Downscale frames for faster processing (e.g. 0.5).
- `--frame-skip <int>`: Skip N frames between OCR passes (reduces CPU usage).
- `--save-on-detect`: Save a snapshot image when a time is detected.
- `--cooldown <float>`: Seconds to wait before repeating the same detection (webcam mode).

How detection reports look
- When a time is found the script prints a clear message such as:

```
Time detected: 12:00 at 2026-01-30 16:56:15
```

- In image mode the script overlays the detected time on the image and shows it in a window (press any key to close). In webcam mode the overlay flashes briefly.

Tuning tips (most common issues)
- Missing narrow digits (like '1'):
  - Lower `--min-area` (try `--min-area 20`).
  - Run with `--show-preprocessed --debug-ocr` to inspect the crops (`preprocessed_region_*.png`) and review segment debug output.
  - The script already includes a seven-segment recognizer and a vertical-stroke fallback for narrow digits; you can tune `--min-area` and reduce noise by increasing `--resize-factor`.

- Webcam lag / high CPU:
  - Use `--resize-factor` (e.g. 0.5) to process smaller frames.
  - Use `--frame-skip 1` or higher to only process every Nth frame.

- Tesseract / dependency errors:
  - Make sure the `--tesseract-path` points to the correct `tesseract.exe` on Windows.
  - If you see OpenCV / NumPy import errors, ensure you installed the pinned `numpy==1.26.4` from `requirements.txt` inside the virtualenv.

Debugging workflow
1. Run with `--image` and `--show-preprocessed --debug-ocr` to get preprocessed crops and console OCR/7-seg logs.
2. Inspect `preprocessed_region_*.png` to verify segments are visible and connected.
3. Adjust `--min-area`, `--resize-factor`, or rerun with different lighting/camera angles.

Files of interest
- `windows_ocr_webcam.py` — main script you run.
- `requirements.txt` — Python dependencies (install inside a virtualenv).
- `preprocessed_region_*.png` — generated when `--show-preprocessed` is used (diagnostic).

If you want me to:
- Add a persistent overlay option for webcam mode (press key to dismiss),
- Write detections to a `detections.log` file,
- Or commit the README changes to a repo branch,
say which and I will implement it.
# raspberry-pi-ocr-bell