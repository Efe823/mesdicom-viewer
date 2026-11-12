# DICOM Viewer

A desktop DICOM viewer and measurement tool for MRI / PET / CT, written in Python (PyQt6 + VTK).
*created by meş*

> ⚠️ Research and education use only — **not for clinical diagnosis**.

## Features

- **2D viewer** – single files, folders and multi-slice series, 16-bit data, window/level presets, invert, zoom/pan
- **Series picker** – handles anonymised data where every image has its own Series UID
- **Measurements** – two clicks → red line + exact distance in mm (uses `PixelSpacing`), export to JSON
- **Print / PDF report** – image with measurements, patient & series details, measurement table
- **Interactive 3D view** (VTK) – atlas-style tissue layers (outer / middle / inner), peel slider,
  layer colours / opacity / solo, cut-away planes, MIP for PET, bright-region highlight
- **3D export** – coloured `.glb` (glTF 2.0) model with slice plane and measurement rods
- **AI-ready** – plug-in `SegmentationBackend` interface (template for a PyTorch 3D U-Net), overlays in 2D and 3D
- Runs on **macOS, Windows and Linux**

## Install

### Windows (easiest)
1. Click **Code → Download ZIP** on this page and extract it.
2. Double-click **`Install DICOM Viewer.bat`** — it installs Python (if needed), all packages and the shortcuts.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 dicom_viewer.py                     # start
python3 dicom_viewer.py --install-launcher  # optional: app icon / desktop shortcut
```

## Usage

| Action | How |
|---|---|
| Open | `Open Folder…` on a series folder (or drag & drop) |
| Change slice | scroll / arrow keys |
| Window / level | right-drag, or the W/L sliders and presets |
| Measure | Measure tool (M): click point A, click point B |
| 3D | View → 2D + 3D (Ctrl/⌘ 2); drag to rotate, scroll to zoom |
| Diagnose problems | `python3 dicom_viewer.py --selftest` |

## Requirements

Python 3.10 – 3.13 (64-bit) · see [`requirements.txt`](requirements.txt).
The 3D view needs OpenGL 3.2+; without it the 2D viewer still works.

## License

Add a license of your choice (for example MIT) before sharing this publicly.
