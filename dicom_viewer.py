#!/usr/bin/env python3
"""
DICOM Viewer & Measurement Tool for macOS
=========================================

A PyQt6 desktop application for viewing single- and multi-slice DICOM series
(MRI, PET, CT, ...) with window/level control and calibrated distance
measurement in millimetres.

Architecture (top to bottom; each layer only depends on the ones above it)
-------------------------------------------------------------------------
1. CORE (Qt-free, numpy + pydicom only)
     DicomVolume          - the 3D voxel volume + spacing/geometry/metadata.
                            This is the object you hand to deep-learning or
                            3D-visualisation code (see .to_torch(), .affine).
     DicomLoader          - reads files/folders/multi-frame files, groups by
                            series, sorts slices spatially, applies rescale.
     apply_window()       - fast vectorised window/level -> uint8.
     distance_mm()        - Euclidean distance using PixelSpacing.
     Measurement          - plain data record of one distance measurement.
2. AI HOOKS (Qt-free)
     SegmentationBackend  - abstract interface for segmentation models.
     ThresholdBackend     - tiny demo backend so the overlay pipeline works
                            without PyTorch installed.
     TorchUNet3DBackend   - template showing how to plug in a PyTorch 3D U-Net.
2b. 3D EXPORT (Qt-free; scikit-image + scipy, imported only when used)
     export_glb()         - iso-surface (marching cubes) + slice plane +
                            measurement rods + segmentation -> .glb (glTF 2.0)
     write_glb()          - dependency-free GLB writer for any MeshPart list
2c. 3D ANATOMY MODEL (Qt/VTK-free)
     AnatomyModel         - body mask, depth-below-skin layers, peeling,
                            bright-region + segmentation surfaces
3. QT WIDGETS
     Viewer3D             - VTK GPU volume renderer beside the 2D canvas:
                            layer list, peel slider, cut-away, MIP
     ImageCanvas          - QGraphicsView: image, pan/zoom, W/L drag,
                            two-click measurement, segmentation overlay.
     ControlPanel         - sidebar with all controls.
     ReportRenderer       - printable / PDF measurement report.
4. APPLICATION
     MainWindow           - wires the panel to the canvas, menus, threading.

Install: pip install pydicom PyQt6 numpy scikit-image vtk
Run:     python3 dicom_viewer.py [file.dcm | folder] [--no-3d] [--selftest]
           --no-3d     start without the 3D engine (View → 2D + 3D turns it on later)
           --selftest  print versions/import times and test OpenGL, no window
           --install-launcher    macOS: ~/Applications/DICOM Viewer.app + Desktop shortcut
                                 Windows: Start menu + Desktop shortcuts · Linux: menu entry
           --uninstall-launcher  remove them again
"""

from __future__ import annotations

import faulthandler
import json
import math
import os
import struct
import sys
import time
import traceback
import zlib
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

try:
    import pydicom
except ImportError:  # handled in main() with a clear message
    pydicom = None


# =============================================================================
# 1. CORE  (no Qt imports below this line until section 3)
# =============================================================================

@dataclass
class DicomVolume:
    """A 3D image volume reconstructed from a DICOM series.

    Attributes
    ----------
    voxels : np.ndarray, float32, shape (Z, Y, X)
        Pixel values in *modality units* (RescaleSlope/Intercept applied),
        e.g. Hounsfield units for CT, Bq/ml (or raw counts) for PET, arbitrary
        signal intensity for MRI. Slices are sorted along the patient-space
        slice normal, so voxels[z] is spatially ordered.
    pixel_spacing : (row_spacing_mm, col_spacing_mm)
        DICOM PixelSpacing: [0] = distance between adjacent ROWS (vertical,
        Y direction), [1] = distance between adjacent COLUMNS (horizontal, X).
    slice_spacing : float
        Distance between slice centres in mm (from positions when available).
    spacing_calibrated : bool
        False if no spacing tag was found and 1.0 mm was assumed.
    """

    voxels: np.ndarray
    pixel_spacing: tuple[float, float] = (1.0, 1.0)
    slice_spacing: float = 1.0
    spacing_calibrated: bool = True
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)          # ImagePositionPatient of slice 0
    orientation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # ImageOrientationPatient
    default_window: Optional[tuple[float, float]] = None           # (center, width) from tags
    invert: bool = False                                           # MONOCHROME1
    modality: str = ""
    units: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    source_files: list[str] = field(default_factory=list)

    # ---- convenience -------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.voxels.shape)  # type: ignore[return-value]

    @property
    def num_slices(self) -> int:
        return int(self.voxels.shape[0])

    @property
    def spacing_zyx(self) -> tuple[float, float, float]:
        """Voxel spacing in (Z, Y, X) order, matching voxels.shape."""
        return (self.slice_spacing, self.pixel_spacing[0], self.pixel_spacing[1])

    @property
    def value_range(self) -> tuple[float, float]:
        return float(self.voxels.min()), float(self.voxels.max())

    @property
    def affine(self) -> np.ndarray:
        """4x4 voxel(index i=col, j=row, k=slice) -> patient (LPS, mm) matrix.

        Useful for exporting to NIfTI (nibabel; flip X/Y for RAS), VTK, or
        any 3D toolkit that needs world coordinates.
        """
        row_dir = np.array(self.orientation[:3], dtype=float)   # direction of increasing column index
        col_dir = np.array(self.orientation[3:6], dtype=float)  # direction of increasing row index
        normal = np.cross(row_dir, col_dir)
        a = np.eye(4)
        a[:3, 0] = row_dir * self.pixel_spacing[1]
        a[:3, 1] = col_dir * self.pixel_spacing[0]
        a[:3, 2] = normal * self.slice_spacing
        a[:3, 3] = self.origin
        return a

    def to_torch(self, device: str = "cpu", add_batch_channel: bool = True):
        """Return voxels as a torch tensor shaped (1, 1, Z, Y, X) for 3D CNNs.

        PyTorch is imported lazily so the viewer works without it.
        """
        import torch  # noqa: WPS433 (deliberately lazy)

        t = torch.from_numpy(np.ascontiguousarray(self.voxels))
        if add_batch_channel:
            t = t[None, None]
        return t.to(device)

    def normalized(self, method: str = "zscore") -> np.ndarray:
        """Intensity-normalised copy, typical pre-processing for DL models."""
        v = self.voxels
        if method == "zscore":
            return ((v - v.mean()) / (v.std() + 1e-8)).astype(np.float32)
        if method == "minmax":
            lo, hi = self.value_range
            return ((v - lo) / (hi - lo + 1e-8)).astype(np.float32)
        raise ValueError(f"Unknown normalisation: {method}")

    def save_npz(self, path: str) -> None:
        """Export voxels + geometry for downstream pipelines (3D viewers, training)."""
        np.savez_compressed(
            path,
            voxels=self.voxels,
            spacing_zyx=np.array(self.spacing_zyx),
            affine=self.affine,
            metadata=json.dumps(self.metadata),
        )


# ---- tag helpers ------------------------------------------------------------

def _first_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """DICOM numeric tags may be a number, a MultiValue list, or missing."""
    if value is None:
        return default
    try:
        if isinstance(value, (list, tuple)) or hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            value = list(value)[0]
        return float(value)
    except (TypeError, ValueError, IndexError):
        return default


def _float_list(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def _nested(ds: Any, *path: str) -> Any:
    """Safely walk functional-group sequences: _nested(ds, 'SharedFunctionalGroupsSequence', 'PixelMeasuresSequence', 'PixelSpacing')."""
    node = ds
    for name in path:
        if node is None:
            return None
        node = getattr(node, name, None)
        # Sequences: take the first item
        if node is not None and not isinstance(node, (str, bytes)) and hasattr(node, "__len__") \
                and hasattr(node, "__getitem__") and name.endswith("Sequence"):
            node = node[0] if len(node) else None
    return node


def _find_tag(ds: Any, tag: str, frame_index: Optional[int] = None) -> Any:
    """Look up a tag at top level, then in shared and per-frame functional groups
    (Enhanced MR/PET/CT multi-frame objects store geometry there)."""
    seq_for_tag = {
        "PixelSpacing": "PixelMeasuresSequence",
        "SliceThickness": "PixelMeasuresSequence",
        "ImageOrientationPatient": "PlaneOrientationSequence",
        "ImagePositionPatient": "PlanePositionSequence",
        "RescaleSlope": "PixelValueTransformationSequence",
        "RescaleIntercept": "PixelValueTransformationSequence",
        "WindowCenter": "FrameVOILUTSequence",
        "WindowWidth": "FrameVOILUTSequence",
    }
    value = getattr(ds, tag, None)
    if value is not None:
        return value
    sub = seq_for_tag.get(tag)
    if sub is None:
        return None
    if frame_index is not None:
        per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
        if per_frame is not None and frame_index < len(per_frame):
            value = _nested(per_frame[frame_index], sub, tag)
            if value is not None:
                return value
    return _nested(ds, "SharedFunctionalGroupsSequence", sub, tag)


def _to_grayscale(arr: np.ndarray, samples_per_pixel: int) -> np.ndarray:
    """Convert RGB pixel data (rare in MR/PET, e.g. fused screenshots) to luminance."""
    if samples_per_pixel == 3 and arr.shape[-1] == 3:
        return (arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114).astype(np.float32)
    return arr


@dataclass
class _Slice:
    """Intermediate per-2D-slice record used while assembling a volume."""
    pixels: np.ndarray            # float32 (Y, X), modality units
    position: Optional[list[float]]
    orientation: Optional[list[float]]
    instance: int
    source: str


@dataclass
class SeriesInfo:
    """Header-only description of one series found during a folder scan."""
    key: tuple
    series_number: int
    description: str
    modality: str
    files: list[str]
    n_images: int

    @property
    def label(self) -> str:
        num = f"#{self.series_number} " if self.series_number >= 0 else ""
        desc = self.description or "(no description)"
        return f"{num}{desc}  ·  {self.modality}  ·  {self.n_images} img"


class DicomLoader:
    """Reads DICOM files into a :class:`DicomVolume`.

    Handles:
      * a single 2D file, a single multi-frame file, or many files / a folder
      * mixed folders (keeps the series with the most slices)
      * spatial slice sorting via ImagePositionPatient · slice-normal,
        falling back to InstanceNumber
      * per-slice RescaleSlope / RescaleIntercept (important for PET, where
        every slice can have its own slope)
      * Enhanced multi-frame functional groups for spacing/orientation
    """

    # Many scanners write files without extension (or "1.2.840...123"), so a
    # folder scan accepts everything except obviously non-DICOM types.
    SKIP_EXTENSIONS = {".txt", ".json", ".xml", ".csv", ".md", ".html", ".htm", ".pdf", ".jpg",
                       ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".zip", ".gz", ".nii",
                       ".py", ".log", ".ini", ".exe", ".db", ".ds_store"}

    # ---- series identity -----------------------------------------------------
    @staticmethod
    def series_key(ds: Any) -> tuple:
        """Key used to decide which images belong to the same 3D series.

        SeriesInstanceUID alone is NOT reliable: many anonymisers (and some
        public sample datasets) give every single image its own UID, which
        would split a 20-slice series into 20 one-slice "series". Instead we
        group by what actually defines a stack: series number/description,
        modality, matrix size and slice orientation.
        """
        orient = _float_list(_find_tag(ds, "ImageOrientationPatient", 0))
        orient_key = tuple(round(v, 2) for v in orient) if orient and len(orient) == 6 else None
        num = getattr(ds, "SeriesNumber", None)
        desc = str(getattr(ds, "SeriesDescription", "") or "")
        if num in (None, "") and not desc:
            # Nothing descriptive: fall back to the UID.
            return ("uid", str(getattr(ds, "SeriesInstanceUID", "unknown")))
        return (str(getattr(ds, "Modality", "")), str(num), desc,
                int(getattr(ds, "Rows", 0) or 0), int(getattr(ds, "Columns", 0) or 0), orient_key)

    @classmethod
    def scan(cls, paths: Sequence[str], progress: Optional[Callable[[int, int], None]] = None) -> list[SeriesInfo]:
        """Read headers only (fast) and list the series present, by series number."""
        if pydicom is None:
            raise RuntimeError("pydicom is not installed. Run: pip install pydicom")
        groups: dict[tuple, SeriesInfo] = {}
        total = len(paths)
        for i, p in enumerate(paths):
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            except Exception:
                ds = None
            if ds is not None and getattr(ds, "Rows", None):  # image objects only
                key = cls.series_key(ds)
                info = groups.get(key)
                if info is None:
                    num = getattr(ds, "SeriesNumber", None)
                    info = groups[key] = SeriesInfo(
                        key=key,
                        series_number=int(num) if str(num).lstrip("-").isdigit() else -1,
                        description=str(getattr(ds, "SeriesDescription", "") or ""),
                        modality=str(getattr(ds, "Modality", "") or "?"),
                        files=[], n_images=0)
                info.files.append(p)
                info.n_images += cls._frame_count(ds)
            if progress and (i % 16 == 0 or i == total - 1):
                progress(i + 1, total)
        series = list(groups.values())
        series.sort(key=lambda s: (s.series_number if s.series_number >= 0 else 10**9, s.description))
        return series

    @classmethod
    def load_series(cls, info: SeriesInfo, progress: Optional[Callable[[int, int], None]] = None) -> DicomVolume:
        return cls.load_files(info.files, progress)

    # ---- public API ----------------------------------------------------------
    @classmethod
    def load_path(cls, path: str, progress: Optional[Callable[[int, int], None]] = None) -> DicomVolume:
        if os.path.isdir(path):
            return cls.load_files(cls.collect_files(path), progress)
        return cls.load_files([path], progress)

    @staticmethod
    def collect_files(folder: str) -> list[str]:
        files: list[str] = []
        for root, _dirs, names in os.walk(folder):
            for name in sorted(names):
                if name.startswith(".") or name.upper() == "DICOMDIR":
                    continue
                if os.path.splitext(name)[1].lower() not in DicomLoader.SKIP_EXTENSIONS:
                    files.append(os.path.join(root, name))
        return files

    @classmethod
    def load_files(cls, paths: Sequence[str], progress: Optional[Callable[[int, int], None]] = None) -> DicomVolume:
        if pydicom is None:
            raise RuntimeError("pydicom is not installed. Run: pip install pydicom")
        datasets: list[tuple[str, Any]] = []
        errors: list[str] = []
        total = len(paths)
        for i, p in enumerate(paths):
            try:
                ds = pydicom.dcmread(p, force=True)
                if "PixelData" in ds:
                    datasets.append((p, ds))
            except Exception as exc:  # non-DICOM or corrupt file: skip it, keep going
                errors.append(f"{os.path.basename(p)}: {exc}")
            if progress and (i % 8 == 0 or i == total - 1):
                progress(i + 1, total)
        if not datasets:
            detail = ("\n" + "\n".join(errors[:5])) if errors else ""
            raise ValueError(f"No DICOM images with pixel data were found.{detail}")
        return cls.from_datasets(datasets)

    @classmethod
    def from_datasets(cls, datasets: Sequence[tuple[str, Any]]) -> DicomVolume:
        """Assemble a volume from already-read (path, dataset) pairs.

        Kept separate from file I/O so it can be unit-tested, or fed datasets
        received over the network (e.g. from a PACS C-MOVE / DICOMweb).
        """
        # 1) Group by series (see series_key) and keep the biggest one.
        groups: dict[tuple, list[tuple[str, Any]]] = {}
        for path, ds in datasets:
            groups.setdefault(cls.series_key(ds), []).append((path, ds))
        series = max(groups.values(), key=lambda g: sum(cls._frame_count(d) for _p, d in g))
        ref = series[0][1]

        # 2) Expand every file into 2D slices (multi-frame files -> many slices).
        slices: list[_Slice] = []
        for path, ds in series:
            slices.extend(cls._extract_slices(path, ds))

        # 3) Keep only slices that match the most common in-plane shape.
        shapes: dict[tuple[int, ...], int] = {}
        for s in slices:
            shapes[s.pixels.shape] = shapes.get(s.pixels.shape, 0) + 1
        common_shape = max(shapes, key=shapes.get)
        slices = [s for s in slices if s.pixels.shape == common_shape]

        # 4) Sort spatially.
        slices, slice_spacing = cls._sort_slices(slices, ref)

        voxels = np.stack([s.pixels for s in slices]).astype(np.float32, copy=False)

        # 5) Geometry & display defaults.
        spacing_raw = _float_list(_find_tag(ref, "PixelSpacing", 0)) \
            or _float_list(getattr(ref, "ImagerPixelSpacing", None))
        calibrated = bool(spacing_raw and len(spacing_raw) >= 2 and spacing_raw[0] > 0 and spacing_raw[1] > 0)
        pixel_spacing = (spacing_raw[0], spacing_raw[1]) if calibrated else (1.0, 1.0)

        wc = _first_float(_find_tag(ref, "WindowCenter", 0))
        ww = _first_float(_find_tag(ref, "WindowWidth", 0))
        default_window = (wc, ww) if wc is not None and ww is not None and ww > 0 else None

        first = slices[0]
        orientation = tuple(first.orientation) if first.orientation and len(first.orientation) == 6 \
            else (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        origin = tuple(first.position) if first.position and len(first.position) == 3 else (0.0, 0.0, 0.0)

        modality = str(getattr(ref, "Modality", ""))
        units = "HU" if modality == "CT" else str(getattr(ref, "Units", "") or "")

        return DicomVolume(
            voxels=voxels,
            pixel_spacing=pixel_spacing,
            slice_spacing=slice_spacing,
            spacing_calibrated=calibrated,
            origin=origin,  # type: ignore[arg-type]
            orientation=orientation,
            default_window=default_window,
            invert=str(getattr(ref, "PhotometricInterpretation", "")) == "MONOCHROME1",
            modality=modality,
            units=units,
            metadata=cls._summarize(ref, voxels.shape, pixel_spacing, slice_spacing, calibrated),
            source_files=sorted({s.source for s in slices}),
        )

    # ---- internals -----------------------------------------------------------
    @staticmethod
    def _frame_count(ds: Any) -> int:
        try:
            return max(1, int(getattr(ds, "NumberOfFrames", 1) or 1))
        except (TypeError, ValueError):
            return 1

    @classmethod
    def _extract_slices(cls, path: str, ds: Any) -> list[_Slice]:
        try:
            arr = ds.pixel_array  # decoded, native dtype (uint16/int16/...)
        except Exception as exc:  # compressed syntaxes need extra plugins
            raise RuntimeError(
                f"Cannot decode pixel data in {os.path.basename(path)} ({exc}).\n"
                "Compressed DICOM needs extra decoders: pip install pylibjpeg pylibjpeg-libjpeg "
                "pylibjpeg-openjpeg  (or python-gdcm)."
            ) from exc

        spp = int(getattr(ds, "SamplesPerPixel", 1) or 1)
        n_frames = cls._frame_count(ds)
        if n_frames == 1:
            arr = arr[None, ...]
        arr = _to_grayscale(arr, spp)

        instance = int(_first_float(getattr(ds, "InstanceNumber", None), 0) or 0)
        out: list[_Slice] = []
        for f in range(arr.shape[0]):
            fi = f if n_frames > 1 else None
            slope = _first_float(_find_tag(ds, "RescaleSlope", fi), 1.0) or 1.0
            intercept = _first_float(_find_tag(ds, "RescaleIntercept", fi), 0.0) or 0.0
            px = arr[f].astype(np.float32)
            if slope != 1.0 or intercept != 0.0:
                px = px * np.float32(slope) + np.float32(intercept)
            out.append(_Slice(
                pixels=px,
                position=_float_list(_find_tag(ds, "ImagePositionPatient", fi)),
                orientation=_float_list(_find_tag(ds, "ImageOrientationPatient", fi)),
                instance=instance * 10000 + f,  # keeps frame order within a file
                source=path,
            ))
        return out

    @staticmethod
    def _sort_slices(slices: list[_Slice], ref: Any) -> tuple[list[_Slice], float]:
        """Sort along the slice normal; return (sorted_slices, slice_spacing_mm)."""
        fallback = _first_float(getattr(ref, "SpacingBetweenSlices", None)) \
            or _first_float(_find_tag(ref, "SliceThickness", 0)) or 1.0
        if len(slices) == 1:
            return slices, float(fallback)

        have_geometry = all(s.position and len(s.position) == 3 for s in slices) and \
            slices[0].orientation and len(slices[0].orientation) == 6
        if have_geometry:
            o = np.array(slices[0].orientation, dtype=float)
            normal = np.cross(o[:3], o[3:])
            dists = np.array([np.dot(normal, s.position) for s in slices])
            order = np.argsort(dists, kind="stable")
            slices = [slices[i] for i in order]
            gaps = np.diff(dists[order])
            gaps = gaps[np.abs(gaps) > 1e-4]  # ignore duplicates
            spacing = float(np.median(np.abs(gaps))) if gaps.size else float(fallback)
            return slices, spacing
        slices = sorted(slices, key=lambda s: s.instance)
        return slices, float(fallback)

    @staticmethod
    def _summarize(ds: Any, shape: tuple[int, ...], spacing: tuple[float, float],
                   slice_spacing: float, calibrated: bool) -> dict[str, str]:
        def g(name: str) -> str:
            v = getattr(ds, name, "")
            return str(v) if v not in (None, "") else "—"

        return {
            "Patient": g("PatientName"),
            "Patient ID": g("PatientID"),
            "Study date": g("StudyDate"),
            "Modality": g("Modality"),
            "Series": g("SeriesDescription"),
            "Manufacturer": g("Manufacturer"),
            "Dimensions": f"{shape[2]} × {shape[1]} × {shape[0]} (X×Y×Z)",
            "Pixel spacing": (f"{spacing[0]:.4g} × {spacing[1]:.4g} mm (row × col)"
                              if calibrated else "not found — 1 px = 1 mm assumed"),
            "Slice spacing": f"{slice_spacing:.4g} mm",
            "Slice thickness": (f"{_first_float(_find_tag(ds, 'SliceThickness', 0)):.3g} mm"
                                if _first_float(_find_tag(ds, 'SliceThickness', 0)) else "—"),
            "Series number": g("SeriesNumber"),
            "Institution": g("InstitutionName"),
            "TR / TE": (f"{g('RepetitionTime')} / {g('EchoTime')}"
                        if getattr(ds, "RepetitionTime", None) else "—"),
            "Bits stored": g("BitsStored"),
            "Photometric": g("PhotometricInterpretation"),
        }


def orientation_labels(orientation: Sequence[float]) -> dict[str, str]:
    """Patient-direction letters for the four image edges (like PACS viewers).

    DICOM patient space is LPS: +x = patient Left, +y = Posterior, +z = Head.
    The image's right edge points along the row direction cosine, the bottom
    edge along the column direction cosine.
    """
    def letter(vec: np.ndarray) -> str:
        names = (("L", "R"), ("P", "A"), ("S", "I"))
        # Up to two letters for oblique planes, strongest component first.
        out = ""
        for i in np.argsort(-np.abs(vec))[:2]:
            if abs(vec[i]) > 0.25:
                out += names[i][0] if vec[i] > 0 else names[i][1]
        return out or "?"

    o = np.asarray(orientation, dtype=float)
    row, col = o[:3], o[3:6]
    return {"right": letter(row), "left": letter(-row), "bottom": letter(col), "top": letter(-col)}


# ---- window / level ---------------------------------------------------------

def apply_window(image: np.ndarray, center: float, width: float, invert: bool = False) -> np.ndarray:
    """Map modality values to 0..255 with a linear window (continuous form of DICOM PS3.3 C.11.2.1.2,
    which also works for float data such as PET SUV with sub-unit widths).

    Values below center - width/2 become black, above center + width/2 white.
    Vectorised: ~1 ms for a 512×512 slice.
    """
    width = max(float(width), 1e-6)
    lo = float(center) - width / 2.0
    out = (image.astype(np.float32, copy=False) - np.float32(lo)) * np.float32(255.0 / width)
    np.clip(out, 0, 255, out=out)
    out = out.astype(np.uint8)
    if invert:
        np.subtract(255, out, out=out)
    return out


def auto_window(volume: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5) -> tuple[float, float]:
    """Robust window from percentiles; ignores the background/air peak and hot spots.

    Subsamples large volumes so it stays fast.
    """
    flat = volume.ravel()
    if flat.size > 2_000_000:
        flat = flat[:: flat.size // 2_000_000]
    lo, hi = np.percentile(flat, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = float(flat.min()), float(flat.max())
    width = max(float(hi - lo), 1e-3)
    return float(lo + width / 2.0), width


# ---- measurement --------------------------------------------------------------

def distance_mm(p1: tuple[float, float], p2: tuple[float, float],
                pixel_spacing: tuple[float, float]) -> float:
    """Euclidean distance between two points given as (x=col, y=row) pixel coords.

    PixelSpacing[0] is the row spacing (applies to the Y / row-index difference)
    and PixelSpacing[1] the column spacing (applies to X / column-index), so
    anisotropic pixels are handled correctly.
    """
    dx_mm = (p2[0] - p1[0]) * pixel_spacing[1]
    dy_mm = (p2[1] - p1[1]) * pixel_spacing[0]
    return math.hypot(dx_mm, dy_mm)


@dataclass
class Measurement:
    """One two-point distance measurement on a given slice (pixel coordinates)."""
    id: int
    slice_index: int
    p1: tuple[float, float]
    p2: tuple[float, float]
    distance_mm: float

    @property
    def distance_px(self) -> float:
        return math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "slice": self.slice_index, "p1": self.p1, "p2": self.p2,
                "distance_mm": round(self.distance_mm, 3)}


def format_dicom_date(value: str) -> str:
    """'20070720' -> '2007-07-20'; anything else is returned unchanged."""
    v = (value or "").strip()
    if len(v) == 8 and v.isdigit():
        return f"{v[:4]}-{v[4:6]}-{v[6:]}"
    return v or "—"


def measurement_table_rows(measurements: Sequence[Measurement],
                           pixel_spacing: tuple[float, float]) -> list[list[str]]:
    """Rows for the printed measurement table (sorted by id).

    Columns: #, distance (mm), |Δx| (mm), |Δy| (mm), start (x, y px), end (x, y px)
    """
    rows = []
    for m in sorted(measurements, key=lambda m: m.id):
        dx = abs(m.p2[0] - m.p1[0]) * pixel_spacing[1]
        dy = abs(m.p2[1] - m.p1[1]) * pixel_spacing[0]
        rows.append([f"#{m.id}", f"{m.distance_mm:.2f} mm", f"{dx:.2f}", f"{dy:.2f}",
                     f"({m.p1[0]:.1f}, {m.p1[1]:.1f})", f"({m.p2[0]:.1f}, {m.p2[1]:.1f})"])
    return rows


def slices_to_print(measurements: Sequence[Measurement], current: int, all_measured: bool) -> list[int]:
    """Which slice indices get a page: the current one, or every slice with measurements."""
    if all_measured:
        measured = sorted({m.slice_index for m in measurements})
        if measured:
            return measured
    return [current]


# =============================================================================
# 2. AI / SEGMENTATION HOOKS  (Qt-free)
# =============================================================================

class SegmentationBackend(ABC):
    """Interface for any segmentation model the viewer can run.

    Implement `predict` to return an integer label map with the SAME shape as
    volume.voxels (Z, Y, X): 0 = background, 1..N = structures. The viewer
    runs it on a worker thread and renders the result as a coloured overlay.
    """

    name: str = "Unnamed"

    @abstractmethod
    def predict(self, volume: DicomVolume,
                progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        ...


class ThresholdBackend(SegmentationBackend):
    """Demo backend: labels voxels above the 85th percentile.

    Exists so the overlay pipeline can be exercised without PyTorch. Replace it
    with a real model (see TorchUNet3DBackend).
    """

    name = "Demo: intensity threshold"

    def __init__(self, percentile: float = 85.0) -> None:
        self.percentile = percentile

    def predict(self, volume, progress=None):
        thr = np.percentile(volume.voxels[:: max(1, volume.num_slices // 16)], self.percentile)
        mask = np.zeros(volume.shape, dtype=np.uint8)
        for z in range(volume.num_slices):
            mask[z] = volume.voxels[z] > thr
            if progress:
                progress(z + 1, volume.num_slices)
        return mask


class TorchUNet3DBackend(SegmentationBackend):
    """Template for a PyTorch 3D U-Net (e.g. MONAI `UNet`, nnU-Net export, custom).

    Usage:
        backend = TorchUNet3DBackend("model.pt", target_spacing=(1.0, 1.0, 1.0))
        window.register_backend(backend)

    The model file is expected to be TorchScript (torch.jit.save) taking a
    (1, 1, Z, Y, X) float tensor and returning (1, C, Z, Y, X) logits.
    """

    name = "PyTorch 3D U-Net"

    def __init__(self, model_path: str, device: Optional[str] = None,
                 patch_size: tuple[int, int, int] = (96, 96, 96)) -> None:
        self.model_path = model_path
        self.device = device
        self.patch_size = patch_size
        self._model = None

    def _load(self):
        import torch
        if self.device is None:
            self.device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.jit.load(self.model_path, map_location=self.device).eval()

    def predict(self, volume, progress=None):
        import torch
        if self._model is None:
            self._load()
        x = torch.from_numpy(volume.normalized("zscore"))[None, None].to(self.device)
        with torch.no_grad():
            # For large volumes, swap this for sliding-window inference, e.g.
            # monai.inferers.sliding_window_inference(x, self.patch_size, 4, self._model)
            logits = self._model(x)
        if progress:
            progress(1, 1)
        return logits.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)


# Colour table for label overlays (RGBA, label index -> colour).
LABEL_COLORS = np.array([
    [0, 0, 0, 0],
    [0, 200, 255, 255],
    [60, 220, 90, 255],
    [255, 190, 0, 255],
    [200, 90, 255, 255],
    [255, 120, 160, 255],
    [120, 255, 230, 255],
    [255, 255, 120, 255],
], dtype=np.uint8)


def label_overlay_rgba(labels: np.ndarray, opacity: float) -> np.ndarray:
    """(Y, X) int labels -> (Y, X, 4) uint8 RGBA overlay."""
    lut = LABEL_COLORS.copy()
    lut[1:, 3] = int(np.clip(opacity, 0, 1) * 255)
    return lut[np.clip(labels, 0, len(lut) - 1).astype(np.intp)]


# =============================================================================
# 2b. 3D MODEL EXPORT  (Qt-free; needs scikit-image + scipy, imported lazily)
# =============================================================================
#
# Pipeline:  volume --(block-average in-plane, interpolate between slices to
#            ~isotropic voxels, Gaussian blur)--> marching cubes iso-surface
#            --> voxel index -> patient mm (DicomVolume.affine)
#            --> drop small noise fragments --> Taubin smoothing
#            --> glTF 2.0 binary (.glb) with: surface, segmentation surfaces,
#            the current slice as a textured plane, measurements as red rods.
#
# Coordinates: DICOM patient space is LPS (x=Left, y=Posterior, z=Superior), mm.
# glTF is Y-up, Z toward the viewer, metres. We map (x, y, z)_LPS ->
# (x, z, -y) * 0.001, a proper rotation, so the head stands upright with the
# face toward the camera and the patient's left on the viewer's right.

@dataclass
class MeshPart:
    """One mesh (= one glTF node/material) in glTF coordinates (metres)."""
    name: str
    vertices: np.ndarray                     # (N, 3) float32
    faces: np.ndarray                        # (M, 3) uint32
    color: tuple[float, float, float, float]
    normals: Optional[np.ndarray] = None     # (N, 3) float32
    uvs: Optional[np.ndarray] = None         # (N, 2) float32
    texture_png: Optional[bytes] = None
    double_sided: bool = False
    unlit: bool = False


@dataclass
class Export3DOptions:
    scheme: str = "layers"                   # "layers" = coloured anatomy layers · "iso" = single surface
    boundaries: tuple[float, float] = (4.0, 12.0)   # layer depths (mm) for scheme "layers"
    highlight_percent: float = 2.0           # brightest % of inner tissue -> highlighted structure
    iso_level: Optional[float] = None        # scheme "iso": None -> Otsu threshold
    detail: int = 2                          # in-plane downsampling (1 = full resolution)
    blur_sigma: float = 1.0                  # Gaussian blur in (resampled) voxels
    smooth_iterations: int = 12              # Taubin smoothing passes
    min_fragment: float = 0.03               # drop pieces smaller than this fraction of the largest
    max_voxels: int = 30_000_000             # memory guard for the resampled grid
    surface_opacity: float = 0.30           # skin (layers) / surface (iso) opacity
    surface_color: tuple[float, float, float] = (0.93, 0.79, 0.70)
    include_slice: bool = True
    slice_index: int = 0
    window: tuple[float, float] = (0.0, 1.0)   # (center, width) for the slice texture
    invert: bool = False
    include_measurements: bool = True
    measurements: list = field(default_factory=list)
    labels: Optional[np.ndarray] = None        # segmentation (Z, Y, X) or None


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold — separates background from tissue; the default iso level."""
    flat = values.ravel()
    if flat.size > 4_000_000:
        flat = flat[:: flat.size // 4_000_000]
    hist, edges = np.histogram(flat, bins=bins)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-12)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between[:-1]))])


def _lps_to_gltf(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1) * 0.001


def _index_to_lps(volume: DicomVolume, ijk: np.ndarray) -> np.ndarray:
    """(N, 3) voxel indices (i=col, j=row, k=slice) -> (N, 3) LPS mm."""
    a = volume.affine
    return ijk @ a[:3, :3].T + a[:3, 3]


def _resample_grid(arr: np.ndarray, volume: DicomVolume, detail: int,
                   max_voxels: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Block-average in-plane by `detail` and linearly interpolate between slices
    so voxels are roughly cubic (MRI slices are often 5 mm apart vs 0.5 mm pixels).

    Returns (grid (Z', Y', X') float32, k_of_z (orig slice index for each new z), d).
    """
    d = max(1, int(detail))
    z, y, x = arr.shape
    yc, xc = (y // d) * d, (x // d) * d
    a = arr[:, :yc, :xc].astype(np.float32, copy=False)
    if d > 1:
        a = a.reshape(z, yc // d, d, xc // d, d).mean(axis=(2, 4), dtype=np.float32)
    inplane = min(volume.pixel_spacing) * d
    factor = max(1.0, volume.slice_spacing / inplane) if inplane > 0 else 1.0
    new_z = int(round((z - 1) * factor)) + 1
    new_z = int(max(z, min(new_z, max_voxels // max(1, a.shape[1] * a.shape[2]))))
    k_of_z = np.linspace(0.0, z - 1.0, new_z)
    if new_z != z:
        i0 = np.floor(k_of_z).astype(int)
        i1 = np.minimum(i0 + 1, z - 1)
        t = (k_of_z - i0).astype(np.float32)[:, None, None]
        a = a[i0] * (1.0 - t) + a[i1] * t
    return np.ascontiguousarray(a, dtype=np.float32), k_of_z, d


def _compact(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces]


def _adjacency(n: int, faces: np.ndarray):
    from scipy import sparse
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    adj = sparse.coo_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, n)).tocsr()
    adj.data[:] = 1.0
    return adj


def _drop_fragments(verts, faces, min_fraction):
    """Remove disconnected noise blobs (e.g. background speckle in MRI)."""
    from scipy.sparse.csgraph import connected_components
    _n, comp = connected_components(_adjacency(len(verts), faces), directed=False)
    face_comp = comp[faces[:, 0]]
    counts = np.bincount(face_comp)
    keep = counts >= max(1, counts.max() * min_fraction)
    return _compact(verts, faces[keep[face_comp]])


def _taubin_smooth(verts, faces, iterations, lam=0.5, mu=-0.53):
    """Taubin λ|μ smoothing: removes the voxel 'staircase' without shrinking the mesh."""
    if iterations <= 0:
        return verts
    from scipy import sparse
    adj = _adjacency(len(verts), faces)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    w = sparse.diags(1.0 / np.maximum(deg, 1)) @ adj
    v = verts.astype(np.float64)
    for _ in range(iterations):
        v = v + lam * (w @ v - v)
        v = v + mu * (w @ v - v)
    return v


def _orient_outward(verts, faces):
    """Flip winding if the closed surface has negative signed volume."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    if np.einsum("ij,ij->", v0, np.cross(v1, v2)) < 0:
        faces = faces[:, [0, 2, 1]]
    return faces


def _vertex_normals(verts, faces):
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    vn = np.zeros_like(verts)
    flat = faces.ravel()
    for c in range(3):
        vn[:, c] = np.bincount(flat, weights=np.repeat(fn[:, c], 3), minlength=len(verts))
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    return vn


def extract_surface(volume: DicomVolume, data: np.ndarray, level: float,
                    opts: Export3DOptions) -> tuple[np.ndarray, np.ndarray]:
    """Iso-surface of `data` (same shape as volume.voxels) in glTF metres.

    Returns (vertices float32 (N,3), faces uint32 (M,3)); empty arrays if no surface.
    """
    try:
        from scipy.ndimage import gaussian_filter
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise RuntimeError("3D export needs scikit-image:  pip install scikit-image") from exc

    grid, k_of_z, d = _resample_grid(data, volume, opts.detail, opts.max_voxels)
    if opts.blur_sigma > 0:
        grid = gaussian_filter(grid, sigma=opts.blur_sigma)
    lo, hi = float(grid.min()), float(grid.max())
    if not lo < level < hi:
        raise ValueError(f"Iso level {level:.4g} is outside the data range ({lo:.4g} … {hi:.4g}).")
    # Pad with "outside" values so surfaces touching the volume edge are closed.
    grid = np.pad(grid, 1, mode="constant", constant_values=lo - abs(hi - lo) * 0.01 - 1.0)
    verts, faces, _n, _v = marching_cubes(grid, level=level, allow_degenerate=False)
    if len(faces) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)
    verts -= 1.0                                              # undo padding
    zz, yy, xx = verts[:, 0], verts[:, 1], verts[:, 2]
    k = np.interp(zz, np.arange(len(k_of_z)), k_of_z)         # back to original slice index
    j = yy * d + (d - 1) / 2.0                                # block centre -> original pixel index
    i = xx * d + (d - 1) / 2.0
    lps = _index_to_lps(volume, np.stack([i, j, k], axis=1))

    faces = faces.astype(np.int64)
    lps, faces = _drop_fragments(lps, faces, opts.min_fragment)
    lps = _taubin_smooth(lps, faces, opts.smooth_iterations)
    v = _lps_to_gltf(lps)
    faces = _orient_outward(v, faces)
    return v.astype(np.float32), faces.astype(np.uint32)


def encode_png_gray(u8: np.ndarray) -> bytes:
    """Minimal 8-bit grayscale PNG encoder (no Pillow/Qt needed)."""
    h, w = u8.shape
    raw = np.hstack([np.zeros((h, 1), np.uint8), np.ascontiguousarray(u8, np.uint8)]).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _slice_plane(volume: DicomVolume, opts: Export3DOptions) -> MeshPart:
    z = int(np.clip(opts.slice_index, 0, volume.num_slices - 1))
    rows, cols = volume.shape[1], volume.shape[2]
    # Corners of the pixel grid (pixel centres are integer indices, edges at ±0.5).
    ijk = np.array([[-0.5, -0.5, z], [cols - 0.5, -0.5, z],
                    [cols - 0.5, rows - 0.5, z], [-0.5, rows - 0.5, z]], dtype=float)
    v = _lps_to_gltf(_index_to_lps(volume, ijk)).astype(np.float32)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)   # glTF: origin top-left
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)   # winding agrees with n below
    n = np.cross(v[1] - v[0], v[3] - v[0])
    n = (n / max(np.linalg.norm(n), 1e-12)).astype(np.float32)
    tex = apply_window(volume.voxels[z], opts.window[0], opts.window[1], opts.invert)
    return MeshPart(name=f"Slice {z + 1}", vertices=v, faces=faces, color=(1, 1, 1, 1),
                    normals=np.tile(n, (4, 1)), uvs=uv, texture_png=encode_png_gray(tex),
                    double_sided=True, unlit=True)


def _rod(p0: np.ndarray, p1: np.ndarray, radius: float, sides: int = 14):
    """Closed cylinder between two points (glTF coords)."""
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        return None
    a = axis / length
    helper = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, helper)
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w
    verts = np.vstack([p0 + radius * ring, p1 + radius * ring, p0, p1])
    f = []
    for s in range(sides):
        t = (s + 1) % sides
        f += [[s, t, sides + t], [s, sides + t, sides + s]]          # side
        f += [[2 * sides, t, s], [2 * sides + 1, sides + s, sides + t]]  # caps
    return verts, np.array(f, dtype=np.int64)


def _measurement_rods(volume: DicomVolume, measurements, radius_m: float) -> Optional[MeshPart]:
    all_v, all_f, offset = [], [], 0
    for m in measurements:
        # Canvas coords have pixel centres at +0.5; the affine expects integer centres.
        ijk = np.array([[m.p1[0] - 0.5, m.p1[1] - 0.5, m.slice_index],
                        [m.p2[0] - 0.5, m.p2[1] - 0.5, m.slice_index]], dtype=float)
        p = _lps_to_gltf(_index_to_lps(volume, ijk))
        rod = _rod(p[0], p[1], radius_m)
        if rod is None:
            continue
        v, f = rod
        f = _orient_outward(v, f)
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)
    if not all_v:
        return None
    v = np.vstack(all_v)
    f = np.vstack(all_f)
    names = ", ".join(f"#{m.id} {m.distance_mm:.2f} mm" for m in measurements)
    return MeshPart(name=f"Measurements: {names}", vertices=v.astype(np.float32),
                    faces=f.astype(np.uint32), color=(0.95, 0.12, 0.12, 1.0),
                    normals=_vertex_normals(v, f).astype(np.float32))


def build_3d_model(volume: DicomVolume, opts: Export3DOptions,
                   progress: Optional[Callable[[int, int], None]] = None) -> tuple[list[MeshPart], dict]:
    """Create all mesh parts for the GLB. Raises ValueError for unusable input."""
    if volume.num_slices < 2:
        raise ValueError("A 3D model needs a series with at least 2 slices.\n"
                         "Open the whole series folder (Open Folder…), not a single image.")
    steps, step = 4, 0

    def tick():
        nonlocal step
        step += 1
        if progress:
            progress(step, steps)

    level = otsu_threshold(volume.voxels) if opts.iso_level is None else float(opts.iso_level)
    parts: list[MeshPart] = []
    v, f = extract_surface(volume, volume.voxels, level, opts)
    if len(f) == 0:
        raise ValueError("No surface found at this iso level — try a lower value.")
    r, g, b = opts.surface_color
    parts.append(MeshPart(name=f"Surface (iso {level:.4g})", vertices=v, faces=f,
                          color=(r, g, b, float(np.clip(opts.surface_opacity, 0.05, 1.0))),
                          normals=_vertex_normals(v.astype(np.float64), f.astype(np.int64)).astype(np.float32),
                          double_sided=opts.surface_opacity < 1.0))
    tick()

    if opts.labels is not None and opts.labels.shape == volume.shape:
        for lab in [int(x) for x in np.unique(opts.labels) if x != 0]:
            mask = (opts.labels == lab).astype(np.float32)
            try:
                lv, lf = extract_surface(volume, mask, 0.5, opts)
            except ValueError:
                continue
            if len(lf):
                c = LABEL_COLORS[min(lab, len(LABEL_COLORS) - 1)] / 255.0
                parts.append(MeshPart(name=f"Segmentation label {lab}", vertices=lv, faces=lf,
                                      color=(float(c[0]), float(c[1]), float(c[2]), 1.0),
                                      normals=_vertex_normals(lv.astype(np.float64),
                                                              lf.astype(np.int64)).astype(np.float32)))
    tick()

    if opts.include_slice:
        parts.append(_slice_plane(volume, opts))
    extent = float(np.linalg.norm(v.max(0) - v.min(0)))
    if opts.include_measurements and opts.measurements:
        rods = _measurement_rods(volume, opts.measurements, radius_m=max(0.0006, extent * 0.0035))
        if rods is not None:
            parts.append(rods)
    tick()

    center = ((v.max(0) + v.min(0)) / 2.0).astype(float)
    md = volume.metadata
    extras = {   # no patient-identifying data in the 3D file
        "generator_note": "DICOM Viewer 3D export. Units: metres. +Y = superior, +Z = anterior, +X = patient left.",
        "modality": volume.modality, "series": md.get("Series", ""),
        "iso_level": level, "voxel_spacing_mm_zyx": list(volume.spacing_zyx),
        "measurements_mm": [round(m.distance_mm, 3) for m in opts.measurements] if opts.include_measurements else [],
    }
    tick()
    return parts, {"center": center, "extras": extras}


def write_glb(path: str, parts: Sequence[MeshPart], center: Optional[np.ndarray] = None,
              extras: Optional[dict] = None) -> int:
    """Write glTF 2.0 binary. Returns file size in bytes. Pure Python + numpy."""
    binary = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []
    materials: list[dict] = []
    meshes: list[dict] = []
    nodes: list[dict] = [{"name": "DICOM model", "children": []}]
    images: list[dict] = []
    textures: list[dict] = []
    uses_unlit = False

    def add_view(data: bytes, target: Optional[int]) -> int:
        while len(binary) % 4:
            binary.append(0)
        view = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        binary.extend(data)
        views.append(view)
        return len(views) - 1

    def add_accessor(arr: np.ndarray, component: int, kind: str, target: Optional[int],
                     with_bounds: bool = False) -> int:
        acc = {"bufferView": add_view(arr.tobytes(), target), "componentType": component,
               "count": int(arr.shape[0]), "type": kind}
        if with_bounds:
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    for part in parts:
        if len(part.faces) == 0:
            continue
        verts = np.ascontiguousarray(part.vertices, dtype=np.float32)
        attrs = {"POSITION": add_accessor(verts, 5126, "VEC3", 34962, with_bounds=True)}
        if part.normals is not None:
            attrs["NORMAL"] = add_accessor(np.ascontiguousarray(part.normals, np.float32), 5126, "VEC3", 34962)
        if part.uvs is not None:
            attrs["TEXCOORD_0"] = add_accessor(np.ascontiguousarray(part.uvs, np.float32), 5126, "VEC2", 34962)
        idx = np.ascontiguousarray(part.faces, dtype=np.uint32).reshape(-1)
        indices = add_accessor(idx, 5125, "SCALAR", 34963)

        pbr = {"baseColorFactor": [float(c) for c in part.color], "metallicFactor": 0.0,
               "roughnessFactor": 0.65}
        if part.texture_png is not None:
            images.append({"bufferView": add_view(part.texture_png, None), "mimeType": "image/png"})
            textures.append({"source": len(images) - 1, "sampler": 0})
            pbr["baseColorTexture"] = {"index": len(textures) - 1}
        mat = {"name": part.name, "pbrMetallicRoughness": pbr, "doubleSided": bool(part.double_sided),
               "alphaMode": "BLEND" if part.color[3] < 0.999 else "OPAQUE"}
        if part.unlit:
            mat["extensions"] = {"KHR_materials_unlit": {}}
            uses_unlit = True
        materials.append(mat)
        meshes.append({"name": part.name, "primitives": [
            {"attributes": attrs, "indices": indices, "material": len(materials) - 1, "mode": 4}]})
        nodes.append({"name": part.name, "mesh": len(meshes) - 1})
        nodes[0]["children"].append(len(nodes) - 1)

    if center is not None:
        nodes[0]["translation"] = [float(-c) for c in center]
    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "DICOM Viewer (PyQt6) GLB exporter"},
        "scene": 0, "scenes": [{"name": "DICOM", "nodes": [0]}],
        "nodes": nodes, "meshes": meshes, "materials": materials,
        "accessors": accessors, "bufferViews": views,
        "buffers": [{"byteLength": 0}],
    }
    if images:
        gltf.update(images=images, textures=textures,
                    samplers=[{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}])
    if uses_unlit:
        gltf["extensionsUsed"] = ["KHR_materials_unlit"]
    if extras:
        gltf["asset"]["extras"] = extras

    while len(binary) % 4:
        binary.append(0)
    gltf["buffers"][0]["byteLength"] = len(binary)
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(binary), 0x004E4942) + bytes(binary))
    return total


def export_glb(volume: DicomVolume, path: str, opts: Optional[Export3DOptions] = None,
               progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """One-call export, usable from scripts:  export_glb(DicomLoader.load_path(folder), "head.glb")"""
    opts = opts or Export3DOptions(slice_index=volume.num_slices // 2,
                                   window=volume.default_window or auto_window(volume.voxels))
    builder = build_layered_model if opts.scheme == "layers" else build_3d_model
    parts, info = builder(volume, opts, progress)
    size = write_glb(path, parts, info["center"], info["extras"])
    return {"path": path, "bytes": size, "parts": [p.name for p in parts],
            "triangles": int(sum(len(p.faces) for p in parts)),
            "iso_level": info["extras"].get("iso_level")}


# =============================================================================
# 2c. 3D ANATOMY LAYER MODEL  (Qt/VTK-free; numpy + scipy + scikit-image)
# =============================================================================
#
# "Atlas-style" peeling needs to know what is outside and what is inside.
# MRI intensities are not calibrated (unlike CT Hounsfield units), so tissue
# classes can't be read reliably from brightness alone. Instead we:
#   1. find the body with an Otsu threshold + hole filling + largest component,
#   2. compute every voxel's DEPTH below the body surface in millimetres
#      (Euclidean distance transform with the real voxel spacing),
#   3. split the body into concentric layers by depth
#      (default 0–4 mm outer, 4–12 mm middle, >12 mm inner),
#   4. extract inner "structures" as surfaces: the brightest inner voxels
#      (CSF on T2, hot spots on PET) and any AI segmentation labels.
# The renderer receives one 8-bit channel per layer, so each layer gets an
# independent colour/opacity transfer function and blends correctly at the
# boundaries. Peeling = blanking voxels shallower than the peel depth.

@dataclass
class AnatomyLayer:
    key: str
    name: str
    color: tuple[float, float, float]
    opacity: float
    visible: bool = True
    kind: str = "volume"                       # "volume" (ray-cast channel) | "surface" (mesh)
    channel: int = 0                           # volume layers: channel index 0..2
    vertices: Optional[np.ndarray] = None      # surface layers: (N, 3) LPS mm
    faces: Optional[np.ndarray] = None         # (M, 3)
    normals: Optional[np.ndarray] = None       # (N, 3)
    info: str = ""


TISSUE_LAYER_DEFAULTS = (
    # opacity is per millimetre of tissue: 0.5 makes a few mm of skin look solid,
    # 0.08 lets you see structures several cm deep inside the inner layer
    ("outer", "Outer layer · skin / scalp", (0.96, 0.80, 0.69), 0.50),
    ("middle", "Middle layer · skull / fat / muscle", (0.93, 0.90, 0.80), 0.22),
    ("inner", "Inner layer · brain / organs", (0.90, 0.60, 0.62), 0.08),
)


def tissue_transfer_function(color: Sequence[float], opacity: float, visible: bool,
                             floor: float = 0.25) -> tuple[list, list]:
    """Transfer-function points for one layer channel.

    Channel value 0 = "not this layer" (fully transparent); 1..255 = intensity
    inside the layer. Darker voxels get a darker shade of the layer colour and
    a reduced (but non-zero, `floor`) opacity so dark tissue such as bone on MRI
    still shows the layer's shape.
    Returns ([(x, r, g, b), ...], [(x, alpha), ...]).
    """
    a = float(opacity) if visible else 0.0
    r, g, b = (float(c) for c in color)
    colors = [(0.0, 0.0, 0.0, 0.0), (1.0, r * 0.30, g * 0.30, b * 0.30), (255.0, r, g, b)]
    alphas = [(0.0, 0.0), (1.0, a * floor), (255.0, a)]
    return colors, alphas


def mip_transfer_function(colormap: str = "gray") -> tuple[list, list]:
    """Maximum-intensity-projection colours: 'gray' or 'hot' (PET convention)."""
    if colormap == "hot":
        colors = [(0, 0, 0, 0), (85, 0.9, 0.0, 0.0), (170, 1.0, 0.75, 0.0), (255, 1.0, 1.0, 1.0)]
    else:
        colors = [(0, 0, 0, 0), (255, 1.0, 1.0, 1.0)]
    alphas = [(0, 0.0), (12, 0.0), (255, 1.0)]
    return colors, alphas


class AnatomyModel:
    """Everything the 3D viewer draws, computed from a DicomVolume.

    Arrays live on a resampled grid (Z', Y', X') of ≤ max_dim pixels in-plane and
    ~isotropic voxels; `grid_to_lps` maps grid indices (x', y', z', 1) to patient
    LPS millimetres, so the same model drives VTK, GLB export or any other
    3D toolkit.
    """

    def __init__(self, volume: DicomVolume, max_dim: int = 256, max_voxels: int = 16_000_000,
                 boundaries: tuple[float, float] = (4.0, 12.0),
                 progress: Optional[Callable[[int, int], None]] = None) -> None:
        try:
            from scipy import ndimage
        except ImportError as exc:
            raise RuntimeError("The 3D view needs scikit-image + scipy:  pip install scikit-image") from exc
        if volume.num_slices < 2:
            raise ValueError("The 3D view needs a series with at least 2 slices "
                             "(use Open Folder… on the series).")

        def tick(i: int) -> None:
            if progress:
                progress(i, 5)

        z, y, x = volume.shape
        d = max(1, int(math.ceil(max(y, x) / max_dim)))
        grid, k_of_z, d = _resample_grid(volume.voxels, volume, d, max_voxels)
        # Blur more across slices than within them: interpolating between thick
        # (e.g. 5 mm) slices otherwise leaves horizontal bands in the 3D view.
        per_slice = (len(k_of_z) - 1) / max(1, z - 1)        # grid slices per original slice
        grid = ndimage.gaussian_filter(grid, (max(0.6, 0.4 * per_slice), 0.6, 0.6))
        self.grid = grid
        self.shape = grid.shape
        self._k_of_z, self._d = k_of_z, d
        self._source_spacing = volume.spacing_zyx
        kscale = (z - 1) / (len(k_of_z) - 1) if len(k_of_z) > 1 else 1.0
        g = np.array([[d, 0, 0, (d - 1) / 2.0],
                      [0, d, 0, (d - 1) / 2.0],
                      [0, 0, kscale, 0],
                      [0, 0, 0, 1]], dtype=float)
        self.grid_to_lps = volume.affine @ g
        ps_r, ps_c = volume.pixel_spacing
        self.voxel_mm = (volume.slice_spacing * kscale, ps_r * d, ps_c * d)   # (z, y, x)
        tick(1)

        # 1) body mask
        body = grid > otsu_threshold(grid)
        pad = 3
        body = np.pad(body, pad)   # so closing doesn't erode the scan's top/bottom faces
        body = ndimage.binary_closing(body, structure=np.ones((3, 3, 3), bool), iterations=2)[pad:-pad, pad:-pad, pad:-pad]
        for zi in range(body.shape[0]):
            body[zi] = ndimage.binary_fill_holes(body[zi])
        lab, n = ndimage.label(body)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            body = (sizes >= 0.1 * sizes.max())[lab]
        self.body = body
        tick(2)

        # 2) depth below the surface (mm)
        self.depth = ndimage.distance_transform_edt(body, sampling=self.voxel_mm).astype(np.float32)
        self.max_depth = float(self.depth.max())
        tick(3)

        # 3) intensity -> 1..255 (0 is reserved for "empty")
        vals = grid[body] if body.any() else grid.ravel()
        lo, hi = np.percentile(vals, [1.0, 99.5])
        self.norm = np.clip((grid - lo) / max(float(hi - lo), 1e-6) * 254.0 + 1.0, 1, 255).astype(np.uint8)

        self.boundaries = tuple(boundaries)
        self.layers = [AnatomyLayer(key=k, name=nm, color=c, opacity=o, channel=i)
                       for i, (k, nm, c, o) in enumerate(TISSUE_LAYER_DEFAULTS)]
        self.components = np.zeros(self.shape + (3,), dtype=np.uint8)   # renderer input, updated in place
        self.mip = np.zeros(self.shape, dtype=np.uint8)
        self.peel_mm = 0.0
        self.layer_index = np.zeros(self.shape, dtype=np.uint8)
        self.update_layer_index()
        self.update_scalars(0.0)
        tick(5)

    # ---- geometry ---------------------------------------------------------------
    @property
    def bounds_lps(self) -> tuple[np.ndarray, np.ndarray]:
        z, y, x = self.shape
        corners = np.array([[i, j, k, 1.0] for i in (0, x - 1) for j in (0, y - 1) for k in (0, z - 1)])
        pts = (corners @ self.grid_to_lps.T)[:, :3]
        return pts.min(0), pts.max(0)

    # ---- layers & peeling ----------------------------------------------------------
    def set_boundaries(self, outer_mm: float, middle_mm: float) -> None:
        self.boundaries = (float(outer_mm), float(middle_mm))
        self.update_layer_index()
        self.update_scalars(self.peel_mm)

    def update_layer_index(self) -> None:
        b1, b2 = sorted(self.boundaries)
        idx = 1 + (self.depth >= b1).astype(np.uint8) + (self.depth >= b2).astype(np.uint8)
        self.layer_index = np.where(self.body, idx, 0).astype(np.uint8)   # 0 = outside, 1..3

    def update_scalars(self, peel_mm: float) -> None:
        """Refresh the renderer arrays IN PLACE (the VTK image shares this memory)."""
        self.peel_mm = float(peel_mm)
        visible = self.body & (self.depth >= self.peel_mm) if self.peel_mm > 0 else self.body
        for c in range(3):
            np.copyto(self.components[..., c], np.where(visible & (self.layer_index == c + 1), self.norm, 0))
        if self.peel_mm > 0:
            np.copyto(self.mip, np.where(visible, self.norm, 0))
        else:
            np.copyto(self.mip, self.norm)   # MIP shows everything (PET uptake outside the mask too)

    def layer_volume_ml(self, channel: int) -> float:
        return float(np.count_nonzero(self.layer_index == channel + 1) * np.prod(self.voxel_mm) / 1000.0)

    # ---- structures (surfaces) -------------------------------------------------------
    def _surface(self, mask: np.ndarray, min_fragment: float = 0.002, blur: float = 0.8):
        from scipy.ndimage import gaussian_filter
        from skimage.measure import marching_cubes
        f = gaussian_filter(mask.astype(np.float32), blur)
        if f.max() <= 0.5:
            return None
        f = np.pad(f, 1)
        verts, faces, _n, _v = marching_cubes(f, 0.5, allow_degenerate=False)
        verts -= 1.0
        xyz1 = np.c_[verts[:, 2], verts[:, 1], verts[:, 0], np.ones(len(verts))]
        lps = (xyz1 @ self.grid_to_lps.T)[:, :3]
        lps, faces = _drop_fragments(lps, faces.astype(np.int64), min_fragment)
        lps = _taubin_smooth(lps, faces, 8)
        faces = _orient_outward(lps, faces)
        return lps.astype(np.float32), faces.astype(np.int64), _vertex_normals(lps, faces).astype(np.float32)

    def bright_regions_layer(self, top_percent: float = 2.0) -> Optional[AnatomyLayer]:
        """Surface around the brightest `top_percent` % of the inner layer
        (fluid/CSF on T2 MRI, tracer hot spots on PET — candidate lesions)."""
        region = self.body & (self.layer_index == 3)
        if np.count_nonzero(region) < 200:
            region = self.body
        if not region.any():
            return None
        thr = float(np.percentile(self.grid[region], 100.0 - float(top_percent)))
        mask = (self.grid > thr) & region
        surf = self._surface(mask)
        if surf is None:
            return None
        v, f, n = surf
        return AnatomyLayer(key="bright", name=f"Bright regions · top {top_percent:g}% of inner tissue",
                            color=(1.0, 0.82, 0.18), opacity=0.9, kind="surface",
                            vertices=v, faces=f, normals=n,
                            info="CSF/fluid on T2 MRI · tracer hot spots on PET (candidate lesions)")

    def label_layers(self, labels: np.ndarray) -> list[AnatomyLayer]:
        """Surfaces for every AI segmentation label (labels shaped like volume.voxels)."""
        z, y, x = labels.shape
        gz, gy, gx = self.shape
        kz = np.clip(np.rint(self._k_of_z).astype(int), 0, z - 1)
        jj = np.clip(np.arange(gy) * self._d + self._d // 2, 0, y - 1)
        ii = np.clip(np.arange(gx) * self._d + self._d // 2, 0, x - 1)
        lab = labels[kz][:, jj][:, :, ii]                       # nearest-neighbour onto the grid
        out = []
        for value in [int(v) for v in np.unique(lab) if v != 0]:
            surf = self._surface(lab == value, min_fragment=0.0)
            if surf is None:
                continue
            v, f, n = surf
            c = LABEL_COLORS[min(value, len(LABEL_COLORS) - 1)][:3] / 255.0
            ml = np.count_nonzero(labels == value) * float(np.prod(self._source_spacing)) / 1000.0 \
                if hasattr(self, "_source_spacing") else 0.0
            out.append(AnatomyLayer(key=f"seg{value}", name=f"Segmentation label {value}"
                                    + (f" · {ml:.1f} ml" if ml else ""),
                                    color=(float(c[0]), float(c[1]), float(c[2])), opacity=1.0,
                                    kind="surface", vertices=v, faces=f, normals=n))
        return out


# ---- coloured, layered GLB export -----------------------------------------------------

# Easy-to-read anatomy colours (glTF RGB 0..1) and opacities; None = use the
# "Opacity" slider. Outer shells are see-through so inner structures show.
LAYER_GLB_STYLE = {
    "outer": ("Skin / outer surface", (0.95, 0.74, 0.60), None),
    "middle": ("Middle layer (skull / fat / muscle)", (0.93, 0.89, 0.74), 0.22),
    "inner": ("Inner layer (brain region)", (0.86, 0.45, 0.52), 0.55),
}
HIGHLIGHT_COLOR = (1.0, 0.72, 0.08)          # bright regions: fluid on T2 / hot spots on PET


def _part_from_lps(name: str, v_lps: np.ndarray, faces: np.ndarray, n_lps: np.ndarray,
                   color: Sequence[float], alpha: float) -> MeshPart:
    """Mesh in patient LPS mm -> MeshPart in glTF metres (same rotation for normals)."""
    n = np.asarray(n_lps, dtype=np.float64)
    return MeshPart(name=name, vertices=_lps_to_gltf(v_lps).astype(np.float32),
                    faces=np.asarray(faces, dtype=np.uint32),
                    color=(float(color[0]), float(color[1]), float(color[2]), float(np.clip(alpha, 0.05, 1.0))),
                    normals=np.stack([n[:, 0], n[:, 2], -n[:, 1]], axis=1).astype(np.float32),
                    double_sided=alpha < 0.999)


def build_layered_model(volume: DicomVolume, opts: Export3DOptions,
                        progress: Optional[Callable[[int, int], None]] = None) -> tuple[list[MeshPart], dict]:
    """GLB parts that match the in-app 3D atlas view: coloured skin / middle /
    inner shells, highlighted bright regions, segmentation labels, the current
    slice and measurement rods."""
    if volume.num_slices < 2:
        raise ValueError("A 3D model needs a series with at least 2 slices.\n"
                         "Open the whole series folder (Open Folder…), not a single image.")
    steps = 4

    def tick(i: int) -> None:
        if progress:
            progress(i, steps)

    max_dim = {1: 320, 2: 224, 4: 128}.get(int(opts.detail), 224)
    model = AnatomyModel(volume, max_dim=max_dim, boundaries=tuple(opts.boundaries))
    tick(1)
    b1, b2 = sorted(model.boundaries)
    parts: list[MeshPart] = []
    for key, mask in (("outer", model.body), ("middle", model.depth >= b1), ("inner", model.depth >= b2)):
        surf = model._surface(mask, min_fragment=0.02)
        if surf is None:
            continue
        name, color, alpha = LAYER_GLB_STYLE[key]
        parts.append(_part_from_lps(name, *surf, color, opts.surface_opacity if alpha is None else alpha))
    if not parts:
        raise ValueError("No body surface was found in this series.")
    tick(2)

    bright = model.bright_regions_layer(opts.highlight_percent)
    if bright is not None:
        parts.append(_part_from_lps(f"Bright regions · top {opts.highlight_percent:g}% "
                                    "(fluid on T2 / hot spots on PET)",
                                    bright.vertices, bright.faces, bright.normals, HIGHLIGHT_COLOR, 1.0))
    if opts.labels is not None and opts.labels.shape == volume.shape:
        for lab in model.label_layers(opts.labels):
            parts.append(_part_from_lps(lab.name, lab.vertices, lab.faces, lab.normals, lab.color, 1.0))
    tick(3)

    skin = parts[0].vertices
    if opts.include_slice:
        parts.append(_slice_plane(volume, opts))
    extent = float(np.linalg.norm(skin.max(0) - skin.min(0)))
    if opts.include_measurements and opts.measurements:
        rods = _measurement_rods(volume, opts.measurements, radius_m=max(0.0006, extent * 0.0035))
        if rods is not None:
            parts.append(rods)
    tick(4)
    md = volume.metadata
    extras = {
        "generator_note": "DICOM Viewer 3D export (coloured layers). Units: metres. "
                          "+Y = superior, +Z = anterior, +X = patient left.",
        "legend": {p.name: [round(c, 3) for c in p.color[:3]] for p in parts},
        "layer_depths_mm": [b1, b2],
        "modality": volume.modality, "series": md.get("Series", ""),
        "measurements_mm": [round(m.distance_mm, 3) for m in opts.measurements] if opts.include_measurements else [],
    }
    center = ((skin.max(0) + skin.min(0)) / 2.0).astype(float)
    return parts, {"center": center, "extras": extras}


# =============================================================================
# 3. QT WIDGETS
# =============================================================================

from PyQt6.QtCore import (QEvent, QObject, QPointF, QRectF, QRunnable, Qt,  # noqa: E402
                          QSettings, QThreadPool, QTimer, QUrl, pyqtSignal)
from PyQt6.QtGui import (QAction, QBrush, QColor, QFont, QImage,  # noqa: E402
                         QDesktopServices, QIcon, QKeySequence, QPainter, QPen, QPixmap,
                         QTransform)
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup,  # noqa: E402
                             QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QFrame, QGraphicsEllipseItem,
                             QGraphicsItem, QGraphicsLineItem, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsScene,
                             QGraphicsSimpleTextItem, QGraphicsView, QGroupBox,
                             QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QMainWindow, QMessageBox, QProgressBar, QPushButton,
                             QScrollArea, QSizePolicy, QSlider, QSplitter,
                             QToolButton, QVBoxLayout, QWidget)
from PyQt6.QtPrintSupport import QPrintDialog, QPrinter, QPrintPreviewDialog  # noqa: E402

# VTK is optional and loaded LAZILY (after the main window is on screen):
# importing it pulls in hundreds of native libraries, which on macOS can take
# a long time on first launch (security scan) — the 2D viewer must not wait.
HAVE_VTK = False
VTK_IMPORT_ERROR = ""


def _load_vtk() -> bool:
    """Import VTK on demand and publish its names as module globals."""
    global HAVE_VTK, VTK_IMPORT_ERROR
    if HAVE_VTK:
        return True
    try:
        # No QVTKRenderWindowInteractor: embedding a native VTK OpenGL view in a
        # PyQt6 window can freeze the whole app on macOS. VTK renders OFF-SCREEN
        # and the frames are shown in a plain Qt widget (VtkImageView).
        import vtkmodules.vtkRenderingFreeType     # noqa: F401
        import vtkmodules.vtkRenderingOpenGL2      # noqa: F401  (OpenGL backend)
        import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401  (GPU ray-casting)
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import (vtkCellArray, vtkImageData, vtkPiecewiseFunction,
                                                   vtkPlane, vtkPolyData)
        from vtkmodules.vtkCommonMath import vtkMatrix4x4
        from vtkmodules.vtkFiltersCore import vtkTubeFilter
        from vtkmodules.vtkFiltersSources import vtkLineSource, vtkPlaneSource
        from vtkmodules.vtkRenderingAnnotation import vtkAnnotatedCubeActor
        from vtkmodules.vtkRenderingCore import (vtkActor, vtkColorTransferFunction, vtkPolyDataMapper,
                                                 vtkRenderer, vtkRenderWindow, vtkTexture, vtkVolume,
                                                 vtkVolumeProperty, vtkWindowToImageFilter)
        from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
    except Exception as exc:  # ImportError, or a broken/incompatible VTK build
        VTK_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return False
    globals().update({k: v for k, v in locals().items() if k != "exc"})
    HAVE_VTK = True
    return True


# ---- platform helpers ------------------------------------------------------------------
IS_MAC = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"


def _sc(mac: str, other: str) -> str:
    """Shortcut text for UI hints: '⌘O' on macOS, 'Ctrl+O' elsewhere."""
    return mac if IS_MAC else other


def default_save_dir() -> str:
    """The user's Desktop (OneDrive-redirected Desktops included), else home."""
    try:
        from PyQt6.QtCore import QStandardPaths
        path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DesktopLocation)
        if path and os.path.isdir(path):
            return path
    except Exception:
        pass
    return os.path.expanduser("~")


# ---- startup diagnostics -------------------------------------------------------------
_T0 = time.perf_counter()
DEBUG = False


def log(msg: str) -> None:
    """Timestamped progress line on stderr — shows where a stuck start-up stopped."""
    print(f"[dicom-viewer {time.perf_counter() - _T0:6.2f}s] {msg}", file=sys.stderr, flush=True)


class Tool:
    PAN = "pan"
    WINDOW = "window"
    MEASURE = "measure"


MEASURE_COLOR = QColor(255, 45, 45)


class ImageCanvas(QGraphicsView):
    """Central viewing canvas.

    Scene layout: a single QGraphicsPixmapItem holds the current slice. All
    annotations (measurement lines, markers, labels, overlays) are *children*
    of that item, so their coordinates are image pixel coordinates (x = col,
    y = row) regardless of zoom, pan, or the aspect-ratio correction applied
    for anisotropic pixels.

    Mouse:
      Measure tool  : click point A, click point B -> red line + distance.
      Pan tool      : left-drag pans.
      W/L tool      : left-drag adjusts window (←→ width, ↑↓ level).
      Right-drag    : window/level in any tool.
      Scroll        : change slice.  Ctrl(⌘)+scroll or trackpad pinch: zoom.
      Double-click  : fit to window.
    """

    sliceChangeRequested = pyqtSignal(int)          # +1 / -1
    windowDragged = pyqtSignal(float, float)        # d_center, d_width (in data units)
    measurementAdded = pyqtSignal(object)           # Measurement
    cursorInfo = pyqtSignal(str)                    # status-bar text

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        # Full updates so the corner annotations never leave smears while panning.
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(12, 12, 14)))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMouseTracking(True)
        self.setAcceptDrops(False)  # the main window handles drops
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)

        self._pixmap_item = QGraphicsPixmapItem()
        # Bilinear interpolation when magnified, like clinical viewers (toggle in View menu).
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self._pixmap_item)
        self._overlay_item = QGraphicsPixmapItem(self._pixmap_item)
        self._overlay_item.setTransformationMode(Qt.TransformationMode.FastTransformation)

        self._image_shape: Optional[tuple[int, int]] = None   # (rows, cols)
        self._pixel_spacing = (1.0, 1.0)
        self._values: Optional[np.ndarray] = None             # current slice, modality units
        self._units = ""
        self._slice_index = 0
        self._tool = Tool.MEASURE
        self._first_point: Optional[QPointF] = None
        self._preview_line: Optional[QGraphicsLineItem] = None
        self._preview_marker: Optional[QGraphicsEllipseItem] = None
        self._items_by_measurement: dict[int, list[QGraphicsItem]] = {}
        self._measurements: dict[int, Measurement] = {}
        self._next_id = 1
        self._wl_drag_origin: Optional[QPointF] = None
        self._wheel_accum = 0
        self._has_fitted = False
        self._auto_fit = True                        # keep fitting on resize until the user zooms
        self._annotations: dict[str, list[str]] = {}
        self._orient: dict[str, str] = {}
        self.show_annotations = True

        self.hint = QLabel("Open a DICOM file or folder  (" + _sc("⌘O / ⇧⌘O", "Ctrl+O / Ctrl+Shift+O")
                           + ")\nor drop it here", self)
        self.hint.setObjectName("emptyHint")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.hint.setGeometry(self.rect())
        if self._auto_fit:
            self.fit()

    def set_smooth(self, on: bool) -> None:
        mode = Qt.TransformationMode.SmoothTransformation if on else Qt.TransformationMode.FastTransformation
        self._pixmap_item.setTransformationMode(mode)

    # ---- corner annotations (drawn in screen space, independent of zoom) ----------
    def set_annotations(self, corners: dict[str, list[str]], orient: dict[str, str]) -> None:
        """corners: keys 'tl', 'tr', 'bl', 'br' -> lines of text."""
        self._annotations = corners
        self._orient = orient
        self.viewport().update()

    def zoom_percent(self) -> float:
        return self.transform().m11() * 100.0

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        if not self.show_annotations or not self._image_shape:
            return
        painter.save()
        painter.resetTransform()                         # viewport (screen) coordinates
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        font = QFont()
        font.setPointSize(13)
        painter.setFont(font)
        fm = painter.fontMetrics()
        vw, vh = self.viewport().width(), self.viewport().height()
        lh, pad = fm.height(), 8
        text_color, shadow = QColor(255, 176, 150), QColor(0, 0, 0, 220)

        def draw(x: float, y: float, text: str) -> None:
            painter.setPen(shadow)
            painter.drawText(QPointF(x + 1, y + 1), text)
            painter.setPen(text_color)
            painter.drawText(QPointF(x, y), text)

        corners = dict(self._annotations)
        corners["bl"] = list(corners.get("bl", [])) + [f"Zoom: {self.zoom_percent():.0f}%"]
        for corner, lines in corners.items():
            for i, line in enumerate(lines):
                w = fm.horizontalAdvance(line)
                x = pad if corner[1] == "l" else vw - pad - w
                y = pad + fm.ascent() + i * lh if corner[0] == "t" \
                    else vh - pad - fm.descent() - (len(lines) - 1 - i) * lh
                draw(x, y, line)
        o = self._orient
        if o:
            draw((vw - fm.horizontalAdvance(o["top"])) / 2, pad + fm.ascent(), o["top"])
            draw((vw - fm.horizontalAdvance(o["bottom"])) / 2, vh - pad - fm.descent(), o["bottom"])
            draw(pad, vh / 2, o["left"])
            draw(vw - pad - fm.horizontalAdvance(o["right"]), vh / 2, o["right"])
        painter.restore()

    # ---- image ---------------------------------------------------------------
    def set_image(self, display_u8: np.ndarray, values: np.ndarray, slice_index: int,
                  pixel_spacing: tuple[float, float], units: str = "") -> None:
        """Show an 8-bit display slice; `values` are the raw modality values for readout."""
        rows, cols = display_u8.shape
        buf = np.ascontiguousarray(display_u8)
        qimg = QImage(buf.data, cols, rows, cols, QImage.Format.Format_Grayscale8).copy()
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self._values = values
        self._units = units
        new_geometry = self._image_shape != (rows, cols) or self._pixel_spacing != pixel_spacing
        self._image_shape = (rows, cols)
        self._pixel_spacing = pixel_spacing

        if new_geometry:
            # Correct aspect ratio for non-square pixels: scale Y by row/col spacing.
            ratio = pixel_spacing[0] / pixel_spacing[1] if pixel_spacing[1] > 0 else 1.0
            self._pixmap_item.setTransform(QTransform.fromScale(1.0, ratio))
            self.scene().setSceneRect(self._pixmap_item.mapRectToScene(QRectF(0, 0, cols, rows)))

        if slice_index != self._slice_index or new_geometry:
            self._slice_index = slice_index
            self.cancel_measurement()
            self._refresh_measurement_visibility()
        if not self._has_fitted or new_geometry:
            self.fit()
            self._has_fitted = True

    def set_overlay(self, rgba: Optional[np.ndarray]) -> None:
        if rgba is None:
            self._overlay_item.setPixmap(QPixmap())
            return
        rows, cols = rgba.shape[:2]
        buf = np.ascontiguousarray(rgba)
        qimg = QImage(buf.data, cols, rows, cols * 4, QImage.Format.Format_RGBA8888).copy()
        self._overlay_item.setPixmap(QPixmap.fromImage(qimg))

    def clear(self) -> None:
        self.clear_measurements()
        self._pixmap_item.setPixmap(QPixmap())
        self.set_overlay(None)
        self._image_shape = None
        self._values = None
        self._has_fitted = False
        self._auto_fit = True
        self._annotations, self._orient = {}, {}

    def fit(self) -> None:
        self._auto_fit = True
        if self._image_shape:
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor: float) -> None:
        current = self.transform().m11()
        if 0.05 < current * factor < 80:
            self._auto_fit = False
            self.scale(factor, factor)

    # ---- tools -----------------------------------------------------------------
    def set_tool(self, tool: str) -> None:
        self._tool = tool
        self.cancel_measurement()
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if tool == Tool.PAN
                         else QGraphicsView.DragMode.NoDrag)
        cursor = {Tool.PAN: Qt.CursorShape.OpenHandCursor,
                  Tool.WINDOW: Qt.CursorShape.SizeAllCursor,
                  Tool.MEASURE: Qt.CursorShape.CrossCursor}[tool]
        self.viewport().setCursor(cursor)

    def _to_pixel(self, view_pos: QPointF) -> Optional[QPointF]:
        """View coords -> image pixel coords (x=col, y=row), or None if outside the image."""
        if not self._image_shape:
            return None
        p = self._pixmap_item.mapFromScene(self.mapToScene(view_pos.toPoint()))
        rows, cols = self._image_shape
        if 0 <= p.x() < cols and 0 <= p.y() < rows:
            return p
        return None

    # ---- measurement management ----------------------------------------------------
    def cancel_measurement(self) -> None:
        self._first_point = None
        for item in (self._preview_line, self._preview_marker):
            if item is not None and item.scene() is not None:
                self.scene().removeItem(item)
        self._preview_line = None
        self._preview_marker = None

    def _marker(self, p: QPointF) -> QGraphicsEllipseItem:
        """Small endpoint marker that stays the same on-screen size at any zoom."""
        r = 4.0
        m = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r, self._pixmap_item)
        m.setPos(p)
        m.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        m.setPen(QPen(QColor(255, 255, 255), 1.2))
        m.setBrush(QBrush(MEASURE_COLOR))
        m.setZValue(12)
        return m

    def _line(self, p1: QPointF, p2: QPointF, dashed: bool = False) -> QGraphicsLineItem:
        line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y(), self._pixmap_item)
        pen = QPen(MEASURE_COLOR, 2.0)
        pen.setCosmetic(True)  # constant 2 px width regardless of zoom
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        line.setPen(pen)
        line.setZValue(10)
        return line

    def add_measurement(self, p1: tuple[float, float], p2: tuple[float, float],
                        slice_index: Optional[int] = None) -> Measurement:
        """Create a measurement programmatically (also used by mouse handling)."""
        m = Measurement(
            id=self._next_id,
            slice_index=self._slice_index if slice_index is None else slice_index,
            p1=p1, p2=p2,
            distance_mm=distance_mm(p1, p2, self._pixel_spacing),
        )
        self._next_id += 1
        a, b = QPointF(*p1), QPointF(*p2)
        line = self._line(a, b)
        label = QGraphicsSimpleTextItem(f"{m.distance_mm:.2f} mm", self._pixmap_item)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        label.setFont(font)
        label.setBrush(QBrush(QColor(255, 235, 235)))
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        mid = (a + b) / 2.0
        label.setPos(mid)
        label.setZValue(13)
        # Dark backing box for legibility on bright tissue.
        br = label.boundingRect().adjusted(-4, -2, 4, 2)
        box = QGraphicsRectItem(br, label)
        box.setBrush(QBrush(QColor(0, 0, 0, 150)))
        box.setPen(QPen(Qt.PenStyle.NoPen))
        box.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)
        label.setTransform(QTransform.fromTranslate(8, -br.height() - 4))

        self._items_by_measurement[m.id] = [line, self._marker(a), self._marker(b), label]
        self._measurements[m.id] = m
        self._refresh_measurement_visibility()
        return m

    def remove_measurement(self, mid: int) -> None:
        for item in self._items_by_measurement.pop(mid, []):
            if item.scene() is not None:
                self.scene().removeItem(item)
        self._measurements.pop(mid, None)

    def clear_measurements(self) -> None:
        for mid in list(self._measurements):
            self.remove_measurement(mid)
        self.cancel_measurement()

    def highlight_measurement(self, mid: Optional[int]) -> None:
        for k, items in self._items_by_measurement.items():
            line = items[0]
            pen = line.pen()
            pen.setWidthF(4.0 if k == mid else 2.0)
            line.setPen(pen)

    def measurements(self) -> list[Measurement]:
        return list(self._measurements.values())

    def _refresh_measurement_visibility(self) -> None:
        """Only show measurements belonging to the current slice."""
        for mid, items in self._items_by_measurement.items():
            visible = self._measurements[mid].slice_index == self._slice_index
            for it in items:
                it.setVisible(visible)

    # ---- events ------------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        pos = event.position()
        if event.button() == Qt.MouseButton.RightButton or \
                (event.button() == Qt.MouseButton.LeftButton and self._tool == Tool.WINDOW):
            self._wl_drag_origin = pos
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._tool == Tool.MEASURE:
            p = self._to_pixel(pos)
            if p is None:
                return
            if self._first_point is None:
                self._first_point = p
                self._preview_marker = self._marker(p)
                self._preview_line = self._line(p, p, dashed=True)
            else:
                a, b = self._first_point, p
                self.cancel_measurement()
                if (a - b).manhattanLength() > 1e-6:
                    m = self.add_measurement((a.x(), a.y()), (b.x(), b.y()))
                    self.measurementAdded.emit(m)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        if self._wl_drag_origin is not None:
            delta = pos - self._wl_drag_origin
            self._wl_drag_origin = pos
            # Positive x = wider window, positive y (drag down) = higher level.
            self.windowDragged.emit(delta.y(), delta.x())
            event.accept()
            return
        p = self._to_pixel(pos)
        if self._first_point is not None and self._preview_line is not None and p is not None:
            a = self._first_point
            self._preview_line.setLine(a.x(), a.y(), p.x(), p.y())
            d = distance_mm((a.x(), a.y()), (p.x(), p.y()), self._pixel_spacing)
            self.cursorInfo.emit(f"Measuring… {d:.2f} mm  (click to finish, Esc to cancel)")
        elif p is not None and self._values is not None:
            c, r = int(p.x()), int(p.y())
            v = self._values[r, c]
            self.cursorInfo.emit(
                f"Slice {self._slice_index + 1}   x {c}  y {r}   value {v:.6g} {self._units}".rstrip())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._wl_drag_origin is not None and event.button() in (
                Qt.MouseButton.RightButton, Qt.MouseButton.LeftButton):
            self._wl_drag_origin = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        # Qt replaces the 2nd press of a fast double-click with this event, so
        # in Measure mode treat it as a normal click (otherwise point B is lost).
        if self._tool == Tool.MEASURE or event.button() != Qt.MouseButton.LeftButton:
            self.mousePressEvent(event)
        else:
            self.fit()

    def wheelEvent(self, event) -> None:
        mods = event.modifiers()
        dy = event.angleDelta().y()
        if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            self.zoom(1.0015 ** dy)
            event.accept()
            return
        # Accumulate small trackpad deltas so one "notch" (120) = one slice.
        self._wheel_accum += dy
        while abs(self._wheel_accum) >= 120:
            step = -1 if self._wheel_accum > 0 else 1
            self._wheel_accum += 120 * step
            self.sliceChangeRequested.emit(step)
        event.accept()

    def viewportEvent(self, event) -> bool:
        # macOS trackpad pinch-to-zoom arrives as a native gesture.
        if event.type() == QEvent.Type.NativeGesture and \
                event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            self.zoom(1.0 + event.value())
            return True
        return super().viewportEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.cancel_measurement()
            return
        super().keyPressEvent(event)


# ---- sidebar -------------------------------------------------------------------

class ControlPanel(QWidget):
    """Left sidebar. Exposes widgets as attributes; MainWindow wires the signals."""

    WINDOW_PRESETS = {
        "DICOM default": None,
        "Auto (1–99.5 %)": "auto",
        "Full range": "full",
        "CT Brain (40/80)": (40, 80),
        "CT Soft tissue (50/400)": (50, 400),
        "CT Lung (−600/1500)": (-600, 1500),
        "CT Bone (400/1800)": (400, 1800),
    }
    SLIDER_STEPS = 1000

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.setMaximumWidth(380)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)

        title = QLabel("DICOM Viewer")
        title.setObjectName("title")
        lay.addWidget(title)

        # File
        row = QHBoxLayout()
        self.open_file_btn = QPushButton("Open File…")
        self.open_folder_btn = QPushButton("Open Folder…")
        row.addWidget(self.open_file_btn)
        row.addWidget(self.open_folder_btn)
        lay.addLayout(row)
        self.series_combo = QComboBox()
        self.series_combo.setToolTip("Series found in the opened files/folder")
        self.series_combo.setEnabled(False)
        self.series_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.series_combo.setMinimumContentsLength(18)
        series_row = QHBoxLayout()
        series_lbl = QLabel("Series")
        series_lbl.setObjectName("metaKey")
        series_row.addWidget(series_lbl)
        series_row.addWidget(self.series_combo, 1)
        lay.addLayout(series_row)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        lay.addWidget(self.progress)

        # Tools
        tools = QGroupBox("Tools")
        tl = QHBoxLayout(tools)
        self.tool_group = QButtonGroup(self)
        self.tool_buttons: dict[str, QToolButton] = {}
        for tool, label, tip in ((Tool.MEASURE, "Measure", "Two clicks → distance in mm  (M)"),
                                 (Tool.PAN, "Pan", "Drag to pan  (P)"),
                                 (Tool.WINDOW, "W / L", "Drag to adjust window/level  (W)")):
            b = QToolButton()
            b.setText(label)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.tool_group.addButton(b)
            self.tool_buttons[tool] = b
            tl.addWidget(b)
        self.tool_buttons[Tool.MEASURE].setChecked(True)
        lay.addWidget(tools)

        # Slice
        sl = QGroupBox("Slice")
        sll = QVBoxLayout(sl)
        self.slice_label = QLabel("—")
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setEnabled(False)
        sll.addWidget(self.slice_label)
        sll.addWidget(self.slice_slider)
        lay.addWidget(sl)

        # Window / level
        wl = QGroupBox("Window / Level")
        wll = QFormLayout(wl)
        wll.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.WINDOW_PRESETS.keys()))
        self.center_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        for s in (self.center_slider, self.width_slider):
            s.setRange(0, self.SLIDER_STEPS)
        self.center_spin = QDoubleSpinBox()
        self.width_spin = QDoubleSpinBox()
        for s in (self.center_spin, self.width_spin):
            s.setKeyboardTracking(False)
            s.setDecimals(2)
        self.invert_check = QCheckBox("Invert grayscale")
        wll.addRow("Preset", self.preset_combo)
        wll.addRow("Level", self.center_spin)
        wll.addRow("", self.center_slider)
        wll.addRow("Width", self.width_spin)
        wll.addRow("", self.width_slider)
        wll.addRow("", self.invert_check)
        lay.addWidget(wl)

        # Measurements
        ms = QGroupBox("Measurements")
        msl = QVBoxLayout(ms)
        self.measure_list = QListWidget()
        self.measure_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.measure_list.setMinimumHeight(90)
        row = QHBoxLayout()
        self.delete_measure_btn = QPushButton("Delete")
        self.clear_measure_btn = QPushButton("Clear All")
        self.export_measure_btn = QPushButton("Export…")
        row.addWidget(self.delete_measure_btn)
        row.addWidget(self.clear_measure_btn)
        row.addWidget(self.export_measure_btn)
        self.calib_warning = QLabel("⚠︎ No PixelSpacing tag — distances shown in pixels-as-mm.")
        self.calib_warning.setObjectName("warning")
        self.calib_warning.setWordWrap(True)
        self.calib_warning.setVisible(False)
        msl.addWidget(self.measure_list)
        msl.addLayout(row)
        msl.addWidget(self.calib_warning)
        lay.addWidget(ms)

        # Print
        pr = QGroupBox("Print")
        prl = QVBoxLayout(pr)
        self.print_patient_check = QCheckBox("Include patient details")
        self.print_patient_check.setChecked(True)
        self.print_all_check = QCheckBox("All slices with measurements")
        self.print_all_check.setToolTip("One page per measured slice (otherwise only the current slice)")
        row = QHBoxLayout()
        self.print_btn = QPushButton("Print…")
        self.print_btn.setToolTip("Print the image with its measurements  (" + _sc("⌘P", "Ctrl+P") + ")")
        self.preview_btn = QPushButton("Preview…")
        self.pdf_btn = QPushButton("Save PDF…")
        for b in (self.print_btn, self.preview_btn, self.pdf_btn):
            b.setEnabled(False)
            row.addWidget(b)
        prl.addWidget(self.print_patient_check)
        prl.addWidget(self.print_all_check)
        prl.addLayout(row)
        lay.addWidget(pr)

        # 3D model export
        g3 = QGroupBox("3D Model (GLB)")
        f3 = QFormLayout(g3)
        f3.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.glb_style_combo = QComboBox()
        self.glb_style_combo.addItem("Coloured anatomy layers", "layers")
        self.glb_style_combo.addItem("Single surface (iso level)", "iso")
        self.glb_style_combo.setToolTip("Coloured layers: skin, middle layer, brain region, bright "
                                        "regions and segmentation in distinct colours (same layers "
                                        "and highlight settings as the 3D view)")
        self.iso_auto_check = QCheckBox("Auto")
        self.iso_auto_check.setChecked(True)
        self.iso_auto_check.setToolTip("Otsu threshold: separates background from tissue (gives the skin surface)")
        self.iso_spin = QDoubleSpinBox()
        self.iso_spin.setKeyboardTracking(False)
        self.iso_spin.setEnabled(False)
        self.iso_spin.setToolTip("Surface threshold in image units. Raise it to peel away skin/fat "
                                 "and reveal brighter inner structures.")
        iso_row = QHBoxLayout()
        iso_row.addWidget(self.iso_spin, 1)
        iso_row.addWidget(self.iso_auto_check)
        self.detail_combo = QComboBox()
        for text, factor in (("High (full res)", 1), ("Medium (½)", 2), ("Low (¼, fast)", 4)):
            self.detail_combo.addItem(text, factor)
        self.detail_combo.setCurrentIndex(1)
        self.surface_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.surface_opacity_slider.setRange(10, 100)
        self.surface_opacity_slider.setValue(30)
        self.surface_opacity_slider.setToolTip("Skin (outer surface) opacity — below 100 % you can see inside")
        self.glb_slice_check = QCheckBox("Current slice as image plane")
        self.glb_slice_check.setChecked(True)
        self.glb_measure_check = QCheckBox("Measurements as red rods")
        self.glb_measure_check.setChecked(True)
        self.glb_seg_check = QCheckBox("Segmentation surfaces")
        self.glb_seg_check.setChecked(True)
        self.glb_open_check = QCheckBox("Open after export")
        self.glb_open_check.setChecked(True)
        self.export_glb_btn = QPushButton("Export 3D Model (.glb)…")
        self.export_glb_btn.setEnabled(False)
        f3.addRow("Style", self.glb_style_combo)
        f3.addRow("Iso level", iso_row)
        f3.addRow("Detail", self.detail_combo)
        f3.addRow("Opacity", self.surface_opacity_slider)
        for w in (self.glb_slice_check, self.glb_measure_check, self.glb_seg_check,
                  self.glb_open_check, self.export_glb_btn):
            f3.addRow("", w)
        lay.addWidget(g3)

        # Segmentation
        seg = QGroupBox("AI Segmentation")
        sgl = QFormLayout(seg)
        self.backend_combo = QComboBox()
        self.run_seg_btn = QPushButton("Run on Volume")
        self.run_seg_btn.setEnabled(False)
        self.overlay_check = QCheckBox("Show overlay")
        self.overlay_check.setChecked(True)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(45)
        sgl.addRow("Model", self.backend_combo)
        sgl.addRow("", self.run_seg_btn)
        sgl.addRow("", self.overlay_check)
        sgl.addRow("Opacity", self.opacity_slider)
        lay.addWidget(seg)

        # Metadata
        md = QGroupBox("Study Information")
        self.meta_layout = QFormLayout(md)
        self.meta_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        lay.addWidget(md)

        lay.addStretch(1)

    def set_metadata(self, meta: dict[str, str]) -> None:
        while self.meta_layout.rowCount():
            self.meta_layout.removeRow(0)
        for k, v in meta.items():
            val = QLabel(v)
            val.setWordWrap(True)
            val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            key = QLabel(k)
            key.setObjectName("metaKey")
            self.meta_layout.addRow(key, val)


# =============================================================================
# 4. APPLICATION
# =============================================================================

# ---- printing ------------------------------------------------------------------

def _align(*flags) -> int:
    """Combine Qt alignment/text flags into the plain int QPainter.drawText expects."""
    out = 0
    for f in flags:
        out |= int(f.value)
    return out


def _gray_qimage(u8: np.ndarray) -> QImage:
    rows, cols = u8.shape
    buf = np.ascontiguousarray(u8)
    return QImage(buf.data, cols, rows, cols, QImage.Format.Format_Grayscale8).copy()


def _rgba_qimage(rgba: np.ndarray) -> QImage:
    rows, cols = rgba.shape[:2]
    buf = np.ascontiguousarray(rgba)
    return QImage(buf.data, cols, rows, cols * 4, QImage.Format.Format_RGBA8888).copy()


class ReportRenderer:
    """Paints a one-page-per-slice measurement report with QPainter.

    Page layout:   header (patient / study / series)
                   image with the red measurement lines, labels and R/L/A/P letters
                   measurement table
                   footer (spacing, W/L, print date, disclaimer)

    Every size is expressed in units of page_width / 1000 and drawn directly in
    the painter's device coordinates, so the result is identical on a 600-dpi
    printer, a PDF, or the on-screen preview. (Rendering the on-screen
    QGraphicsScene instead would print the zoom-independent labels at
    screen-pixel size, i.e. microscopically small on paper.)
    """

    def __init__(self, volume: DicomVolume, center: float, width: float, invert: bool,
                 measurements: Sequence[Measurement], labels: Optional[np.ndarray] = None,
                 overlay_opacity: float = 0.45, include_patient: bool = True) -> None:
        self.volume = volume
        self.center, self.width, self.invert = center, width, invert
        self.measurements = list(measurements)
        self.labels = labels
        self.overlay_opacity = overlay_opacity
        self.include_patient = include_patient

    # ---- entry point -----------------------------------------------------------------
    def print_to(self, printer: QPrinter, slices: Sequence[int]) -> None:
        painter = QPainter()
        if not painter.begin(printer):
            raise RuntimeError("Could not start printing — is the printer available?")
        try:
            area = QRectF(printer.pageLayout().paintRectPixels(printer.resolution()))
            page = QRectF(0, 0, area.width(), area.height())  # painter origin = printable area
            for i, z in enumerate(slices):
                if i:
                    printer.newPage()
                self.paint_page(painter, page, z, i + 1, len(slices))
        finally:
            painter.end()

    # ---- page -------------------------------------------------------------------------
    def paint_page(self, p: QPainter, page: QRectF, z: int, page_no: int, n_pages: int) -> None:
        v = self.volume
        u = page.width() / 1000.0
        ink, muted, rule = QColor(22, 22, 26), QColor(110, 110, 120), QColor(205, 205, 212)
        p.save()
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
                         | QPainter.RenderHint.TextAntialiasing)
        left, right = Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignRight
        vcenter = Qt.AlignmentFlag.AlignVCenter

        # Header -----------------------------------------------------------------
        y = page.top()
        title_h = 40 * u
        p.setFont(self._font(u, 26, bold=True))
        p.setPen(ink)
        p.drawText(QRectF(page.left(), y, page.width(), title_h), _align(left, vcenter),
                   "DICOM Measurement Report")
        p.setFont(self._font(u, 13))
        p.setPen(muted)
        p.drawText(QRectF(page.left(), y, page.width(), title_h), _align(right, vcenter),
                   f"Page {page_no} / {n_pages}")
        y += title_h + 6 * u

        md = v.metadata
        if self.include_patient:
            col_a = [("Patient", md.get("Patient", "—")), ("Patient ID", md.get("Patient ID", "—")),
                     ("Study date", format_dicom_date(md.get("Study date", "")))]
        else:
            col_a = [("Patient", "(not printed)"),
                     ("Study date", format_dicom_date(md.get("Study date", "")))]
        col_b = [("Series", f"{md.get('Series', '—')}  (#{md.get('Series number', '—')})"),
                 ("Modality", f"{v.modality or '—'}   {md.get('Manufacturer', '')}".strip()),
                 ("Slice", f"{z + 1} / {v.num_slices}    z = {z * v.slice_spacing:.1f} mm")]
        row_h, half = 21 * u, page.width() / 2
        for i, (k, val) in enumerate(col_a):
            self._key_value(p, QRectF(page.left(), y + i * row_h, half - 10 * u, row_h), k, val, u, ink, muted)
        for i, (k, val) in enumerate(col_b):
            self._key_value(p, QRectF(page.left() + half, y + i * row_h, half, row_h), k, val, u, ink, muted)
        y += max(len(col_a), len(col_b)) * row_h + 10 * u
        p.setPen(QPen(rule, max(1.0, 1.2 * u)))
        p.drawLine(QPointF(page.left(), y), QPointF(page.right(), y))
        y += 14 * u

        # Body: image + table share the remaining height -------------------------
        on_slice = sorted((m for m in self.measurements if m.slice_index == z), key=lambda m: m.id)
        footer_h = 46 * u
        available = page.bottom() - footer_h - y
        tbl_row = 22 * u
        table_h = (len(on_slice) + 1) * tbl_row + 34 * u if on_slice else 34 * u
        table_h = min(table_h, available * 0.38)
        image_area = QRectF(page.left(), y, page.width(), available - table_h - 14 * u)
        self._paint_image(p, image_area, z, on_slice, u)
        y = image_area.bottom() + 14 * u
        self._paint_table(p, QRectF(page.left(), y, page.width(), table_h), on_slice, u, ink, muted, rule)

        # Footer -------------------------------------------------------------------
        fy = page.bottom() - footer_h + 10 * u
        p.setPen(QPen(rule, max(1.0, 1.2 * u)))
        p.drawLine(QPointF(page.left(), fy), QPointF(page.right(), fy))
        p.setFont(self._font(u, 11))
        p.setPen(muted)
        spacing = (f"Pixel spacing {v.pixel_spacing[0]:.4g} × {v.pixel_spacing[1]:.4g} mm"
                   if v.spacing_calibrated else "⚠ No PixelSpacing tag — distances are in pixels")
        info = (f"{spacing}   ·   W {self.width:.6g} / L {self.center:.6g}   ·   "
                f"Printed {datetime.now():%Y-%m-%d %H:%M}")
        line = QRectF(page.left(), fy + 6 * u, page.width(), footer_h - 16 * u)
        p.drawText(line, _align(left, vcenter), info)
        p.drawText(line, _align(right, vcenter), "Research use only — not for diagnosis")
        p.restore()

    # ---- pieces -------------------------------------------------------------------------
    @staticmethod
    def _font(u: float, size: float, bold: bool = False) -> QFont:
        f = QFont()
        f.setPixelSize(max(1, int(round(size * u))))
        f.setBold(bold)
        return f

    def _key_value(self, p, rect, key, value, u, ink, muted) -> None:
        key_w = 92 * u
        p.setFont(self._font(u, 12))
        p.setPen(muted)
        p.drawText(QRectF(rect.left(), rect.top(), key_w, rect.height()),
                   _align(Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignVCenter), key)
        p.setFont(self._font(u, 13, bold=True))
        p.setPen(ink)
        val_rect = QRectF(rect.left() + key_w, rect.top(), rect.width() - key_w, rect.height())
        text = p.fontMetrics().elidedText(str(value), Qt.TextElideMode.ElideRight, int(val_rect.width()))
        p.drawText(val_rect, _align(Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignVCenter), text)

    def _paint_image(self, p: QPainter, area: QRectF, z: int,
                     measurements: Sequence[Measurement], u: float) -> None:
        v = self.volume
        rows, cols = v.shape[1], v.shape[2]
        # Fit by PHYSICAL size so anisotropic pixels keep the correct aspect ratio.
        w_mm, h_mm = cols * v.pixel_spacing[1], rows * v.pixel_spacing[0]
        scale = min(area.width() / w_mm, area.height() / h_mm)
        tw, th = w_mm * scale, h_mm * scale
        target = QRectF(area.center().x() - tw / 2, area.center().y() - th / 2, tw, th)

        p.save()
        p.fillRect(area, QColor(0, 0, 0))
        p.setClipRect(area)
        display = apply_window(v.voxels[z], self.center, self.width, self.invert)
        p.drawImage(target, _gray_qimage(display))
        if self.labels is not None:
            p.drawImage(target, _rgba_qimage(label_overlay_rgba(self.labels[z], self.overlay_opacity)))

        def to_page(pt: tuple[float, float]) -> QPointF:
            return QPointF(target.left() + pt[0] / cols * tw, target.top() + pt[1] / rows * th)

        # Measurements: red line, white-ringed endpoints, numbered label.
        label_font = self._font(u, 14, bold=True)
        for m in measurements:
            a, b = to_page(m.p1), to_page(m.p2)
            pen = QPen(MEASURE_COLOR, 2.6 * u)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(a, b)
            p.setPen(QPen(QColor(255, 255, 255), 1.3 * u))
            p.setBrush(QBrush(MEASURE_COLOR))
            for q in (a, b):
                p.drawEllipse(q, 4.5 * u, 4.5 * u)

            unit = "mm" if v.spacing_calibrated else "px"
            text = f"#{m.id}  {m.distance_mm:.2f} {unit}"
            p.setFont(label_font)
            fm = p.fontMetrics()
            bw, bh = fm.horizontalAdvance(text) + 12 * u, fm.height() + 6 * u
            mid = (a + b) / 2.0
            bx = min(max(mid.x() + 8 * u, area.left() + 4 * u), area.right() - bw - 4 * u)
            by = min(max(mid.y() - bh - 8 * u, area.top() + 4 * u), area.bottom() - bh - 4 * u)
            box = QRectF(bx, by, bw, bh)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(0, 0, 0, 175)))
            p.drawRoundedRect(box, 3 * u, 3 * u)
            p.setPen(QColor(255, 236, 236))
            p.drawText(box, _align(Qt.AlignmentFlag.AlignCenter), text)

        # Orientation letters and W/L inside the black panel.
        p.setFont(self._font(u, 15, bold=True))
        p.setPen(QColor(255, 176, 150))
        o = orientation_labels(v.orientation)
        pad = 8 * u
        c = Qt.AlignmentFlag
        p.drawText(area.adjusted(pad, pad, -pad, -pad), _align(c.AlignHCenter, c.AlignTop), o["top"])
        p.drawText(area.adjusted(pad, pad, -pad, -pad), _align(c.AlignHCenter, c.AlignBottom), o["bottom"])
        p.drawText(area.adjusted(pad, pad, -pad, -pad), _align(c.AlignLeft, c.AlignVCenter), o["left"])
        p.drawText(area.adjusted(pad, pad, -pad, -pad), _align(c.AlignRight, c.AlignVCenter), o["right"])
        p.setFont(self._font(u, 12))
        p.drawText(area.adjusted(pad, pad, -pad, -pad), _align(c.AlignLeft, c.AlignBottom),
                   f"W {self.width:.6g}  L {self.center:.6g}")
        p.restore()

    def _paint_table(self, p, rect: QRectF, measurements, u, ink, muted, rule) -> None:
        a = Qt.AlignmentFlag
        p.save()
        p.setFont(self._font(u, 14, bold=True))
        p.setPen(ink)
        title_h = 26 * u
        p.drawText(QRectF(rect.left(), rect.top(), rect.width(), title_h), _align(a.AlignLeft, a.AlignVCenter),
                   f"Measurements on this slice ({len(measurements)})")
        if not measurements:
            p.setFont(self._font(u, 12))
            p.setPen(muted)
            p.drawText(QRectF(rect.left() + 300 * u, rect.top(), rect.width() - 300 * u, title_h),
                       _align(a.AlignLeft, a.AlignVCenter), "None — draw a line with the Measure tool first.")
            p.restore()
            return

        headers = ["#", "Distance", "|Δx| mm", "|Δy| mm", "Start (x, y px)", "End (x, y px)"]
        fractions = [0.07, 0.17, 0.13, 0.13, 0.25, 0.25]
        xs = [rect.left()]
        for f in fractions[:-1]:
            xs.append(xs[-1] + f * rect.width())
        widths = [f * rect.width() for f in fractions]
        row_h = 22 * u
        y = rect.top() + title_h

        def row(cells, yy, bold=False, color=ink, shade=False):
            if shade:
                p.fillRect(QRectF(rect.left(), yy, rect.width(), row_h), QColor(244, 244, 247))
            p.setFont(self._font(u, 12, bold=bold))
            p.setPen(color)
            for x, w, txt in zip(xs, widths, cells):
                p.drawText(QRectF(x + 6 * u, yy, w - 8 * u, row_h), _align(a.AlignLeft, a.AlignVCenter), txt)

        row(headers, y, bold=True, color=muted)
        y += row_h
        p.setPen(QPen(rule, max(1.0, u)))
        p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        data = measurement_table_rows(measurements, self.volume.pixel_spacing)
        fit = max(0, int((rect.bottom() - y) // row_h))
        if len(data) > fit:                          # leave a line for the "+N more" note
            fit = max(0, fit - 1)
        for i, cells in enumerate(data[:fit]):
            row(cells, y, bold=False, shade=(i % 2 == 1))
            y += row_h
        if len(data) > fit:
            row([f"… {len(data) - fit} more — use Export Measurements (.json) for the full list"],
                y, color=muted)
        p.restore()


# ---- 3D anatomy viewer (VTK) -------------------------------------------------------

class LayerRow(QWidget):
    """One row of the layer list: visibility · colour · name · opacity · solo."""

    changed = pyqtSignal()
    soloToggled = pyqtSignal(object, bool)

    def __init__(self, layer: AnatomyLayer, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.layer = layer
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 1, 0, 1)
        h.setSpacing(6)
        self.check = QCheckBox()
        self.check.setChecked(layer.visible)
        self.check.setToolTip("Show / hide")
        self.color_btn = QToolButton()
        self.color_btn.setFixedSize(18, 18)
        self.color_btn.setToolTip("Change colour")
        self._update_swatch()
        self.name = QLabel(layer.name)
        self.name.setToolTip(layer.info or layer.name)
        self.name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.name.setMinimumWidth(60)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(round(layer.opacity * 100)))
        self.slider.setFixedWidth(96)
        self.slider.setToolTip("Opacity")
        self.solo = QToolButton()
        self.solo.setText("Solo")
        self.solo.setCheckable(True)
        self.solo.setToolTip("Show only this layer / structure")
        for w in (self.check, self.color_btn):
            h.addWidget(w)
        h.addWidget(self.name, 1)
        h.addWidget(self.slider)
        h.addWidget(self.solo)
        self.check.toggled.connect(self._visible)
        self.color_btn.clicked.connect(self._pick_color)
        self.slider.valueChanged.connect(self._opacity)
        self.solo.toggled.connect(lambda on: self.soloToggled.emit(self, on))

    def set_visible_silently(self, on: bool) -> None:
        self.check.blockSignals(True)
        self.check.setChecked(on)
        self.check.blockSignals(False)
        self.layer.visible = on

    def _update_swatch(self) -> None:
        r, g, b = (int(c * 255) for c in self.layer.color)
        self.color_btn.setStyleSheet(f"QToolButton {{ background: rgb({r},{g},{b}); "
                                     "border: 1px solid #555; border-radius: 4px; padding: 0; }}")

    def _visible(self, on: bool) -> None:
        self.layer.visible = on
        self.changed.emit()

    def _opacity(self, value: int) -> None:
        self.layer.opacity = value / 100.0
        self.changed.emit()

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor.fromRgbF(*self.layer.color), self, "Layer colour")
        if c.isValid():
            self.layer.color = (c.redF(), c.greenF(), c.blueF())
            self._update_swatch()
            self.changed.emit()


class VtkImageView(QWidget):
    """Shows frames rendered off-screen by VTK and turns mouse input into
    camera signals. A plain QWidget: nothing native/OpenGL is embedded.

    Left-drag rotate · Shift/middle-drag pan · right-drag, scroll or pinch zoom ·
    double-click or R = front view.
    """

    rotate = pyqtSignal(float, float)
    pan = pyqtSignal(float, float)
    zoom = pyqtSignal(float)
    released = pyqtSignal()
    resized = pyqtSignal()
    resetRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self._frame: Optional[QImage] = None
        self._last: Optional[QPointF] = None
        self._mode: Optional[str] = None

    def set_frame(self, image: QImage) -> None:
        self._frame = image
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(8, 8, 12))
        if self._frame is not None:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(QRectF(self.rect()), self._frame)
        p.end()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.resized.emit()

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        self._last = event.position()
        b, mods = event.button(), event.modifiers()
        if b == Qt.MouseButton.MiddleButton or (
                b == Qt.MouseButton.LeftButton and mods & Qt.KeyboardModifier.ShiftModifier):
            self._mode = "pan"
        elif b == Qt.MouseButton.LeftButton:
            self._mode = "rotate"
        elif b == Qt.MouseButton.RightButton:
            self._mode = "zoom"

    def mouseMoveEvent(self, event) -> None:
        if self._mode is None or self._last is None:
            return
        pos = event.position()
        d = pos - self._last
        self._last = pos
        if self._mode == "rotate":
            self.rotate.emit(d.x(), d.y())
        elif self._mode == "pan":
            self.pan.emit(d.x(), d.y())
        elif self._mode == "zoom":
            self.zoom.emit(1.01 ** (-d.y()))

    def mouseReleaseEvent(self, event) -> None:
        if self._mode is not None:
            self._mode = None
            self.released.emit()

    def mouseDoubleClickEvent(self, event) -> None:
        self.resetRequested.emit()

    def wheelEvent(self, event) -> None:
        dy = event.angleDelta().y() or event.pixelDelta().y()
        if dy:
            self.zoom.emit(1.0015 ** dy)
        event.accept()

    def event(self, ev) -> bool:
        if ev.type() == QEvent.Type.NativeGesture and \
                ev.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            self.zoom.emit(1.0 + ev.value())        # trackpad pinch
            return True
        return super().event(ev)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_R:
            self.resetRequested.emit()
        else:
            super().keyPressEvent(event)


class Viewer3D(QWidget):
    """Interactive 3D anatomy viewer, shown next to the 2D canvas.

    * GPU ray-cast volume rendering; each tissue layer is its own channel with
      an independent colour/opacity transfer function.
    * "Peel" slider removes everything shallower than N mm below the skin.
    * Layer list: show/hide, colour, opacity, solo — volume layers and surface
      structures (bright regions, AI segmentation labels).
    * Cut-away plane (sagittal / coronal / axial), current 2D slice as a
      textured plane, measurements as red tubes, orientation cube (L/R/A/P/S/I).
    Mouse: left-drag rotate · right-drag or scroll zoom · middle/shift-drag pan.
    """

    message = pyqtSignal(str)
    brightRequested = pyqtSignal(float)   # top-% for the bright-region surface

    MODES = ("Atlas layers", "MIP · grayscale", "MIP · hot (PET)")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.model: Optional[AnatomyModel] = None
        self.volume: Optional[DicomVolume] = None
        self.available = False          # True once the VTK UI has been built successfully
        self.enabled = True             # False when disabled by --no-3d or a failed start
        self._initialized = False
        self._rows: list[LayerRow] = []
        self._solo_saved: Optional[dict] = None
        self._structures: dict[str, AnatomyLayer] = {}
        self._surface_actors: dict[str, tuple] = {}
        self._meas_actors: list = []
        self._pending_slice: Optional[tuple[int, np.ndarray]] = None
        self._pending_measurements: list = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._outer = outer
        self._placeholder = QLabel("3D view not loaded yet")
        self._placeholder.setObjectName("emptyHint")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        outer.addWidget(self._placeholder)

    # Settings key that survives a crash/hang so the next launch can skip 3D.
    PENDING_KEY = "viewer3d/start_pending"

    def activate(self) -> bool:
        """Load VTK and build the 3D UI (once). Returns False if 3D is unavailable."""
        if self.available:
            return True
        if not self.enabled:
            return False
        self._placeholder.setText("Loading the 3D engine (VTK)…\n\n"
                                  "The first launch after installing can take up to a minute.")
        QApplication.processEvents()
        settings = QSettings()
        settings.setValue(self.PENDING_KEY, True)       # cleared after the first successful render
        settings.sync()
        log("3D: importing VTK…")
        faulthandler.dump_traceback_later(90, exit=False)   # prints where it hangs, if it does
        try:
            ok = _load_vtk()
            log("3D: VTK import " + ("ok" if ok else f"failed — {VTK_IMPORT_ERROR}"))
            if ok:
                self._build_vtk_ui()
                log("3D: viewer widgets created")
        except Exception as exc:
            traceback.print_exc()
            ok = False
            self._fail(f"{type(exc).__name__}: {exc}")
        finally:
            faulthandler.cancel_dump_traceback_later()
        if not ok:
            settings.setValue(self.PENDING_KEY, False)
            if VTK_IMPORT_ERROR:
                self._fail(VTK_IMPORT_ERROR)
            return False
        self._placeholder.hide()
        self.available = True
        # The panel may already be on screen (its showEvent fired before VTK
        # existed), so connect OpenGL now instead of waiting for another show.
        if self.isVisible():
            QTimer.singleShot(50, self._init_gl)
        return True

    def _fail(self, reason: str) -> None:
        self.enabled = False
        self._placeholder.setText("The 3D view could not start.\n\n"
                                  "Install / repair it in Terminal (venv active):\n"
                                  "pip install --upgrade vtk scikit-image\n\n"
                                  f"Details: {reason}")
        self._placeholder.show()

    def _build_vtk_ui(self) -> None:
        outer = self._outer
        # -- toolbar --
        bar = QWidget()
        bar.setObjectName("toolbar3d")
        hb = QHBoxLayout(bar)
        hb.setContentsMargins(8, 5, 8, 5)
        hb.setSpacing(6)
        title = QLabel("3D")
        title.setObjectName("title3d")
        hb.addWidget(title)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.MODES)
        self.mode_combo.setToolTip("Atlas layers: coloured tissue layers · MIP: maximum-intensity projection")
        hb.addWidget(self.mode_combo)
        for text, key in (("Front", "front"), ("Left", "left"), ("Right", "right"),
                          ("Top", "top"), ("Back", "back")):
            b = QToolButton()
            b.setText(text)
            b.clicked.connect(lambda _=False, k=key: self.set_view(k))
            hb.addWidget(b)
        hb.addStretch(1)
        self.slice_check = QCheckBox("Slice")
        self.slice_check.setChecked(True)
        self.slice_check.setToolTip("Show the current 2D slice inside the 3D view")
        self.meas_check = QCheckBox("Lines")
        self.meas_check.setChecked(True)
        self.meas_check.setToolTip("Show measurements as red tubes")
        hb.addWidget(self.slice_check)
        hb.addWidget(self.meas_check)
        outer.addWidget(bar)

        # -- VTK render window --
        self.vtk_widget = VtkImageView(self)
        outer.addWidget(self.vtk_widget, 1)
        self.status = QLabel("Open a series to build the 3D view")
        self.status.setObjectName("status3d")
        outer.addWidget(self.status)

        # -- controls --
        controls = QWidget()
        cl = QVBoxLayout(controls)
        cl.setContentsMargins(10, 8, 10, 8)
        cl.setSpacing(6)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        peel_row = QHBoxLayout()
        self.peel_slider = QSlider(Qt.Orientation.Horizontal)
        self.peel_slider.setRange(0, 0)
        self.peel_slider.setToolTip("Remove everything shallower than this depth below the skin")
        self.peel_label = QLabel("0.0 mm")
        self.peel_label.setMinimumWidth(62)
        peel_row.addWidget(self.peel_slider, 1)
        peel_row.addWidget(self.peel_label)
        form.addRow("Peel depth", peel_row)

        depth_row = QHBoxLayout()
        self.b1_spin = QDoubleSpinBox()
        self.b2_spin = QDoubleSpinBox()
        for sp, val in ((self.b1_spin, 4.0), (self.b2_spin, 12.0)):
            sp.setRange(0.5, 150.0)
            sp.setDecimals(1)
            sp.setSingleStep(0.5)
            sp.setSuffix(" mm")
            sp.setValue(val)
            sp.setKeyboardTracking(False)
        self.b1_spin.setToolTip("Outer layer thickness (skin / scalp)")
        self.b2_spin.setToolTip("Depth where the inner layer starts")
        depth_row.addWidget(QLabel("outer <"))
        depth_row.addWidget(self.b1_spin)
        depth_row.addWidget(QLabel("< middle <"))
        depth_row.addWidget(self.b2_spin)
        depth_row.addWidget(QLabel("< inner"))
        form.addRow("Layer depths", depth_row)

        cut_row = QHBoxLayout()
        self.cut_combo = QComboBox()
        self.cut_combo.addItems(["Off", "Sagittal (L ↔ R)", "Coronal (A ↔ P)", "Axial (S ↔ I)"])
        self.cut_slider = QSlider(Qt.Orientation.Horizontal)
        self.cut_slider.setRange(0, 1000)
        self.cut_slider.setValue(500)
        self.cut_flip = QCheckBox("Flip")
        self.cut_flip.setToolTip("Keep the other half")
        cut_row.addWidget(self.cut_combo)
        cut_row.addWidget(self.cut_slider, 1)
        cut_row.addWidget(self.cut_flip)
        form.addRow("Cut-away", cut_row)

        bright_row = QHBoxLayout()
        self.bright_spin = QDoubleSpinBox()
        self.bright_spin.setRange(0.1, 30.0)
        self.bright_spin.setDecimals(1)
        self.bright_spin.setSingleStep(0.5)
        self.bright_spin.setValue(2.0)
        self.bright_spin.setSuffix(" %")
        self.bright_spin.setKeyboardTracking(False)
        self.bright_spin.setToolTip("Brightest part of the inner layer shown as a yellow surface "
                                    "(CSF on T2 MRI, hot spots on PET)")
        bright_row.addWidget(QLabel("brightest"))
        bright_row.addWidget(self.bright_spin)
        bright_row.addWidget(QLabel("of inner tissue"))
        bright_row.addStretch(1)
        form.addRow("Highlight", bright_row)
        cl.addLayout(form)

        layers_box = QGroupBox("Layers & structures")
        self.layers_layout = QVBoxLayout(layers_box)
        self.layers_layout.setSpacing(2)
        cl.addWidget(layers_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(controls)
        scroll.setMinimumHeight(150)
        scroll.setMaximumHeight(330)
        outer.addWidget(scroll)
        self._controls = controls

        # -- VTK scene --
        self.renderer = vtkRenderer()
        self.renderer.GradientBackgroundOn()
        self.renderer.SetBackground(0.03, 0.03, 0.05)
        self.renderer.SetBackground2(0.17, 0.18, 0.23)
        self.renderer.SetLayer(0)
        self.render_window = vtkRenderWindow()
        self.render_window.SetOffScreenRendering(1)      # never opens a native window
        self.render_window.SetNumberOfLayers(2)
        self.render_window.SetSize(400, 400)
        self.render_window.AddRenderer(self.renderer)
        self._grabber = vtkWindowToImageFilter()
        self._grabber.SetInput(self.render_window)
        self._grabber.SetInputBufferTypeToRGB()
        self._grabber.ReadFrontBufferOff()
        self._grabber.ShouldRerenderOff()
        self._fast_frame = False
        self._render_timer = QTimer(self, singleShot=True, interval=0)
        self._render_timer.timeout.connect(self._do_render)
        self._settle_timer = QTimer(self, singleShot=True, interval=160)  # full-res after scrolling
        self._settle_timer.timeout.connect(lambda: self.render())
        v = self.vtk_widget
        v.rotate.connect(self._on_rotate)
        v.pan.connect(self._on_pan)
        v.zoom.connect(self._on_zoom)
        v.released.connect(lambda: self.render())
        v.resized.connect(lambda: self.render())
        v.resetRequested.connect(lambda: self.set_view("front"))

        cube = vtkAnnotatedCubeActor()     # LPS: +x Left, +y Posterior, +z Superior
        cube.SetXPlusFaceText("L")
        cube.SetXMinusFaceText("R")
        cube.SetYPlusFaceText("P")
        cube.SetYMinusFaceText("A")
        cube.SetZPlusFaceText("S")
        cube.SetZMinusFaceText("I")
        cube.SetFaceTextScale(0.6)
        cube.GetCubeProperty().SetColor(0.22, 0.24, 0.30)
        cube.GetTextEdgesProperty().SetColor(0.6, 0.6, 0.65)
        for prop in (cube.GetXPlusFaceProperty(), cube.GetXMinusFaceProperty(),
                     cube.GetYPlusFaceProperty(), cube.GetYMinusFaceProperty(),
                     cube.GetZPlusFaceProperty(), cube.GetZMinusFaceProperty()):
            prop.SetColor(1.0, 0.72, 0.6)
        # Orientation cube: a second renderer layered in the corner whose camera
        # follows the main camera's direction.
        self.marker_ren = vtkRenderer()
        self.marker_ren.SetLayer(1)
        self.marker_ren.SetViewport(0.0, 0.0, 0.18, 0.18)
        self.marker_ren.InteractiveOff()
        self.marker_ren.AddActor(cube)
        self.render_window.AddRenderer(self.marker_ren)

        # Volume pipeline (inputs are attached in set_model)
        self.vol_mapper = vtkSmartVolumeMapper()
        self.vol_mapper.SetAutoAdjustSampleDistances(True)
        self.volume_actor = vtkVolume()
        self.volume_actor.SetMapper(self.vol_mapper)
        self.atlas_prop = vtkVolumeProperty()
        self.atlas_prop.IndependentComponentsOn()
        self.atlas_prop.SetInterpolationTypeToLinear()
        self._ctf = [vtkColorTransferFunction() for _ in range(3)]
        self._otf = [vtkPiecewiseFunction() for _ in range(3)]
        for c in range(3):
            self.atlas_prop.SetColor(c, self._ctf[c])
            self.atlas_prop.SetScalarOpacity(c, self._otf[c])
            self.atlas_prop.SetShade(c, 1)
        for c in range(3):   # the non-indexed setters only touch channel 0
            self.atlas_prop.SetAmbient(c, 0.28)
            self.atlas_prop.SetDiffuse(c, 0.75)
            self.atlas_prop.SetSpecular(c, 0.22)
            self.atlas_prop.SetSpecularPower(c, 18.0)
        self.mip_prop = vtkVolumeProperty()
        self.mip_prop.SetInterpolationTypeToLinear()
        self._mip_ctf = vtkColorTransferFunction()
        self._mip_otf = vtkPiecewiseFunction()
        self.mip_prop.SetColor(self._mip_ctf)
        self.mip_prop.SetScalarOpacity(self._mip_otf)
        self.mip_prop.ShadeOff()

        # Slice plane
        self._plane_src = vtkPlaneSource()
        self._slice_img = vtkImageData()
        self._texture = vtkTexture()
        self._texture.SetInputData(self._slice_img)
        self._texture.InterpolateOn()
        plane_mapper = vtkPolyDataMapper()
        plane_mapper.SetInputConnection(self._plane_src.GetOutputPort())
        self.slice_actor = vtkActor()
        self.slice_actor.SetMapper(plane_mapper)
        self.slice_actor.SetTexture(self._texture)
        self.slice_actor.GetProperty().LightingOff()
        self.slice_actor.GetProperty().SetOpacity(0.92)
        self.slice_actor.VisibilityOff()

        self._clip_plane = vtkPlane()
        self._keep: list = []          # numpy buffers shared with VTK arrays

        # timers coalesce rapid UI changes into one re-render
        self._peel_timer = QTimer(self, singleShot=True, interval=45)
        self._peel_timer.timeout.connect(self._apply_peel)
        self._slice_timer = QTimer(self, singleShot=True, interval=30)
        self._slice_timer.timeout.connect(self._apply_slice)

        self.mode_combo.currentIndexChanged.connect(self._apply_mode)
        self.slice_check.toggled.connect(self._toggle_slice)
        self.meas_check.toggled.connect(lambda _: self._apply_measurements())
        self.peel_slider.valueChanged.connect(self._peel_changed)
        self.b1_spin.valueChanged.connect(self._boundaries_changed)
        self.b2_spin.valueChanged.connect(self._boundaries_changed)
        self.cut_combo.currentIndexChanged.connect(lambda _: self._apply_cut())
        self.cut_slider.valueChanged.connect(lambda _: self._apply_cut())
        self.cut_flip.toggled.connect(lambda _: self._apply_cut())
        self._bright_timer = QTimer(self, singleShot=True, interval=400)
        self._bright_timer.timeout.connect(lambda: self.brightRequested.emit(float(self.bright_spin.value())))
        self.bright_spin.valueChanged.connect(lambda _v: self._bright_timer.start())
        self._controls.setEnabled(False)

    # ---- camera interaction (driven by VtkImageView signals) ---------------------------
    def _on_rotate(self, dx: float, dy: float) -> None:
        cam = self.renderer.GetActiveCamera()
        k = 220.0 / max(200, self.vtk_widget.height())           # degrees per pixel
        cam.Azimuth(-dx * k)
        cam.Elevation(dy * k)
        cam.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()
        self.render(fast=True)

    def _on_pan(self, dx: float, dy: float) -> None:
        cam = self.renderer.GetActiveCamera()
        fp, pos = np.array(cam.GetFocalPoint()), np.array(cam.GetPosition())
        up = np.array(cam.GetViewUp())
        dop = (fp - pos) / max(np.linalg.norm(fp - pos), 1e-9)
        right = np.cross(dop, up)
        per_px = 2.0 * cam.GetDistance() * math.tan(math.radians(cam.GetViewAngle() / 2)) \
            / max(1, self.vtk_widget.height())
        shift = (-dx * right + dy * up) * per_px
        cam.SetFocalPoint(*(fp + shift))
        cam.SetPosition(*(pos + shift))
        self.render(fast=True)

    def _on_zoom(self, factor: float) -> None:
        cam = self.renderer.GetActiveCamera()
        cam.Dolly(max(0.5, min(2.0, factor)))
        self.renderer.ResetCameraClippingRange()
        self.render(fast=True)
        self._settle_timer.start()

    # ---- lifecycle -----------------------------------------------------------------
    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.available and not self._initialized:
            # Defer: initialise OpenGL only once the native window is really on screen.
            QTimer.singleShot(50, self._init_gl)
        else:
            self.render()

    def _init_gl(self) -> None:
        if self._initialized or not self.available or not self.isVisible():
            return
        log("3D: first off-screen render…")
        faulthandler.dump_traceback_later(60, exit=False)
        try:
            self._initialized = True
            self._do_render()
        except Exception as exc:
            traceback.print_exc()
            self.available = False
            self.vtk_widget.hide()
            self._fail(f"OpenGL initialisation failed — {type(exc).__name__}: {exc}")
            return
        finally:
            faulthandler.cancel_dump_traceback_later()
        QSettings().setValue(self.PENDING_KEY, False)   # 3D works on this machine
        log("3D: ready")

    def shutdown(self) -> None:
        """Release the OpenGL context before the window closes (avoids macOS crashes)."""
        if self.available:
            try:
                self._render_timer.stop()
                self.render_window.Finalize()
            except Exception:
                pass

    def render(self, fast: bool = False) -> None:
        """Request a frame. Many requests in a row collapse into one render."""
        if not (self.available and self.isVisible()):
            return
        if not self._initialized:
            QTimer.singleShot(0, self._init_gl)   # safety net: never stay blank
            return
        if not self._render_timer.isActive():
            self._fast_frame = fast
        elif not fast:
            self._fast_frame = False          # a full-quality request wins
        self._render_timer.start()

    def _do_render(self) -> None:
        """Render off-screen and hand the pixels to the Qt widget."""
        view = self.vtk_widget
        w, h = max(16, view.width()), max(16, view.height())
        if self._fast_frame:
            scale = 0.5                                 # quick frames while dragging
        else:
            scale = min(view.devicePixelRatioF(), 1800.0 / max(w, h))
        W, H = max(16, int(w * scale)), max(16, int(h * scale))
        if tuple(self.render_window.GetSize()) != (W, H):
            self.render_window.SetSize(W, H)
        # keep the orientation cube looking the same way as the main camera
        cam = self.renderer.GetActiveCamera()
        d = np.array(cam.GetPosition()) - np.array(cam.GetFocalPoint())
        d = d / max(np.linalg.norm(d), 1e-9)
        mc = self.marker_ren.GetActiveCamera()
        mc.SetFocalPoint(0.0, 0.0, 0.0)
        mc.SetPosition(*(d * 4.2))
        mc.SetViewUp(*cam.GetViewUp())
        self.marker_ren.ResetCameraClippingRange()

        self.render_window.Render()
        self._grabber.Modified()
        self._grabber.Update()
        out = self._grabber.GetOutput()
        gw, gh, _ = out.GetDimensions()
        pixels = vtk_to_numpy(out.GetPointData().GetScalars()).reshape(gh, gw, -1)[::-1]
        rgb = np.ascontiguousarray(pixels[..., :3], dtype=np.uint8)
        image = QImage(rgb.data, gw, gh, 3 * gw, QImage.Format.Format_RGB888).copy()
        view.set_frame(image)

    def set_busy(self, text: str) -> None:
        if self.available:
            self.status.setText(text)

    def clear(self) -> None:
        if not self.available:
            return
        self.model = None
        self.volume = None
        self.renderer.RemoveAllViewProps()
        self._surface_actors.clear()
        self._structures.clear()
        self._meas_actors.clear()
        self._keep.clear()
        self._rebuild_rows()
        self._controls.setEnabled(False)
        self.status.setText("Open a series to build the 3D view")
        self.render()

    # ---- model -----------------------------------------------------------------------
    def _image_from(self, arr: np.ndarray, components: int):
        """vtkImageData that SHARES memory with `arr` (Z, Y, X[, C]) — in-place edits show up."""
        img = vtkImageData()
        z, y, x = arr.shape[:3]
        img.SetDimensions(x, y, z)
        img.SetSpacing(1.0, 1.0, 1.0)
        img.SetOrigin(0.0, 0.0, 0.0)
        flat = arr.reshape(-1, components) if components > 1 else arr.reshape(-1)
        vtk_arr = numpy_to_vtk(flat, deep=False)
        vtk_arr.SetName("scalars")
        img.GetPointData().SetScalars(vtk_arr)
        self._keep.append(flat)
        return img, vtk_arr

    def set_model(self, model: AnatomyModel, volume: DicomVolume) -> None:
        if not self.available:
            return
        self.clear()
        self.model, self.volume = model, volume
        self._comp_img, self._comp_arr = self._image_from(model.components, 3)
        self._mip_img, self._mip_arr = self._image_from(model.mip, 1)
        m = vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                m.SetElement(r, c, float(model.grid_to_lps[r, c]))
        self.volume_actor.SetUserMatrix(m)
        unit = float(min(model.voxel_mm))
        for c in range(3):
            self.atlas_prop.SetScalarOpacityUnitDistance(c, unit)
        self.mip_prop.SetScalarOpacityUnitDistance(unit)
        self.renderer.AddVolume(self.volume_actor)
        self.renderer.AddActor(self.slice_actor)

        self.peel_slider.blockSignals(True)
        self.peel_slider.setRange(0, int(model.max_depth * 10))
        self.peel_slider.setValue(0)
        self.peel_slider.blockSignals(False)
        self.peel_label.setText("0.0 mm")
        for sp, val in ((self.b1_spin, model.boundaries[0]), (self.b2_spin, model.boundaries[1])):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        self._controls.setEnabled(True)
        self._rebuild_rows()
        if volume.modality in ("PT", "NM"):          # PET/SPECT: MIP with hot colours by default
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(2)
            self.mode_combo.blockSignals(False)
        self._apply_mode()
        self._apply_cut()
        if self._pending_slice is not None:
            self._apply_slice()
        self._apply_measurements()
        lo, hi = model.bounds_lps
        size = hi - lo
        self.status.setText(f"{model.shape[2]}×{model.shape[1]}×{model.shape[0]} voxels · "
                            f"{size[0]:.0f}×{size[1]:.0f}×{size[2]:.0f} mm · drag to rotate, scroll to zoom")
        self.set_view("front")

    def set_structure(self, layer: AnatomyLayer) -> None:
        """Add or replace a surface structure (bright regions, segmentation labels)."""
        if not self.available or self.model is None:
            return
        self.remove_structure(layer.key)
        self._structures[layer.key] = layer
        pts = vtkPoints()
        pts.SetData(numpy_to_vtk(np.ascontiguousarray(layer.vertices, np.float32), deep=True))
        conn = np.ascontiguousarray(np.asarray(layer.faces).reshape(-1), dtype=np.int64)
        offsets = np.arange(0, len(conn) + 1, 3, dtype=np.int64)
        cells = vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True), numpy_to_vtkIdTypeArray(conn, deep=True))
        poly = vtkPolyData()
        poly.SetPoints(pts)
        poly.SetPolys(cells)
        normals = numpy_to_vtk(np.ascontiguousarray(layer.normals, np.float32), deep=True)
        normals.SetName("Normals")
        poly.GetPointData().SetNormals(normals)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOff()
        actor = vtkActor()
        actor.SetMapper(mapper)
        p = actor.GetProperty()
        p.SetSpecular(0.35)
        p.SetSpecularPower(25)
        self.renderer.AddActor(actor)
        self._surface_actors[layer.key] = (actor, mapper)
        self._rebuild_rows()
        self._apply_cut()

    def remove_structure(self, key: str) -> None:
        if key in self._surface_actors:
            actor, _m = self._surface_actors.pop(key)
            self.renderer.RemoveActor(actor)
        self._structures.pop(key, None)

    def remove_structures(self, prefix: str) -> None:
        if not self.available:
            return
        for key in [k for k in self._structures if k.startswith(prefix)]:
            self.remove_structure(key)
        self._rebuild_rows()
        self.render()

    # ---- layer list ------------------------------------------------------------------
    def _all_layers(self) -> list[AnatomyLayer]:
        vol_layers = self.model.layers if self.model else []
        return list(vol_layers) + list(self._structures.values())

    def _rebuild_rows(self) -> None:
        while self.layers_layout.count():
            item = self.layers_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows = []
        self._solo_saved = None
        for layer in self._all_layers():
            row = LayerRow(layer)
            row.changed.connect(self._apply_appearance)
            row.soloToggled.connect(self._solo)
            self.layers_layout.addWidget(row)
            self._rows.append(row)
        if not self._rows:
            hint = QLabel("No 3D model yet")
            hint.setObjectName("metaKey")
            self.layers_layout.addWidget(hint)
        self._apply_appearance()

    def _solo(self, row: LayerRow, on: bool) -> None:
        if on:
            if self._solo_saved is None:
                self._solo_saved = {id(r.layer): r.layer.visible for r in self._rows}
            for r in self._rows:
                if r is not row:
                    r.solo.blockSignals(True)
                    r.solo.setChecked(False)
                    r.solo.blockSignals(False)
                r.set_visible_silently(r is row)
        elif self._solo_saved is not None:
            for r in self._rows:
                r.set_visible_silently(self._solo_saved.get(id(r.layer), True))
            self._solo_saved = None
        self._apply_appearance()

    def _apply_appearance(self) -> None:
        if not self.available or self.model is None:
            return
        for layer in self.model.layers:
            colors, alphas = tissue_transfer_function(layer.color, layer.opacity, layer.visible)
            ctf, otf = self._ctf[layer.channel], self._otf[layer.channel]
            ctf.RemoveAllPoints()
            otf.RemoveAllPoints()
            for x, r, g, b in colors:
                ctf.AddRGBPoint(x, r, g, b)
            for x, a in alphas:
                otf.AddPoint(x, a)
        for key, layer in self._structures.items():
            actor, _m = self._surface_actors[key]
            actor.SetVisibility(bool(layer.visible and layer.opacity > 0))
            actor.GetProperty().SetColor(*layer.color)
            actor.GetProperty().SetOpacity(max(0.0, min(1.0, layer.opacity)))
        self.render()

    # ---- interaction ---------------------------------------------------------------------
    def _apply_mode(self) -> None:
        if self.model is None:
            return
        idx = self.mode_combo.currentIndex()
        if idx == 0:
            self.vol_mapper.SetInputData(self._comp_img)
            self.vol_mapper.SetBlendModeToComposite()
            self.volume_actor.SetProperty(self.atlas_prop)
        else:
            colors, alphas = mip_transfer_function("hot" if idx == 2 else "gray")
            self._mip_ctf.RemoveAllPoints()
            self._mip_otf.RemoveAllPoints()
            for x, r, g, b in colors:
                self._mip_ctf.AddRGBPoint(x, r, g, b)
            for x, a in alphas:
                self._mip_otf.AddPoint(x, a)
            self.vol_mapper.SetInputData(self._mip_img)
            self.vol_mapper.SetBlendModeToMaximumIntensity()
            self.volume_actor.SetProperty(self.mip_prop)
        self.render()

    def _peel_changed(self, value: int) -> None:
        self.peel_label.setText(f"{value / 10:.1f} mm")
        self._peel_timer.start()

    def _apply_peel(self) -> None:
        if self.model is None:
            return
        self.model.update_scalars(self.peel_slider.value() / 10.0)
        self._mark_modified()

    def _boundaries_changed(self) -> None:
        if self.model is None:
            return
        self.model.set_boundaries(self.b1_spin.value(), self.b2_spin.value())
        self._mark_modified()

    def _mark_modified(self) -> None:
        for arr, img in ((self._comp_arr, self._comp_img), (self._mip_arr, self._mip_img)):
            arr.Modified()
            img.Modified()
        self.render()

    def _apply_cut(self) -> None:
        if not self.available or self.model is None:
            return
        mappers = [self.vol_mapper] + [m for (_a, m) in self._surface_actors.values()]
        for m in mappers:
            m.RemoveAllClippingPlanes()
        mode = self.cut_combo.currentIndex()
        self.cut_slider.setEnabled(mode > 0)
        self.cut_flip.setEnabled(mode > 0)
        if mode > 0:
            axis = mode - 1                                   # 0 = x (L-R), 1 = y (A-P), 2 = z (S-I)
            lo, hi = self.model.bounds_lps
            origin = (lo + hi) / 2.0
            origin[axis] = lo[axis] + (hi[axis] - lo[axis]) * self.cut_slider.value() / 1000.0
            normal = np.zeros(3)
            normal[axis] = -1.0 if self.cut_flip.isChecked() else 1.0
            self._clip_plane.SetOrigin(*origin)
            self._clip_plane.SetNormal(*normal)
            for m in mappers:
                m.AddClippingPlane(self._clip_plane)
        self.render()

    def set_view(self, key: str) -> None:
        """Camera presets in patient space (LPS): front = looking at the face."""
        if not self.available or self.model is None:
            return
        lo, hi = self.model.bounds_lps
        c = (lo + hi) / 2.0
        direction, up = {
            "front": ((0, -1, 0), (0, 0, 1)), "back": ((0, 1, 0), (0, 0, 1)),
            "left": ((1, 0, 0), (0, 0, 1)), "right": ((-1, 0, 0), (0, 0, 1)),
            "top": ((0, 0, 1), (0, -1, 0)),
        }[key]
        cam = self.renderer.GetActiveCamera()
        cam.SetFocalPoint(*c)
        cam.SetPosition(*(c + np.array(direction, dtype=float) * 1000.0))
        cam.SetViewUp(*up)
        self.renderer.ResetCamera()
        cam.Zoom(1.15)
        self.renderer.ResetCameraClippingRange()
        self.render()

    # ---- 2D sync -----------------------------------------------------------------------------
    def update_slice(self, index: int, display_u8: np.ndarray) -> None:
        """Called by the main window whenever the 2D slice or window/level changes."""
        self._pending_slice = (index, display_u8)
        if self.available and self.model is not None:
            self._slice_timer.start()

    def _toggle_slice(self, on: bool) -> None:
        self.slice_actor.SetVisibility(bool(on and self.model is not None and self._pending_slice is not None))
        self.render()

    def _apply_slice(self) -> None:
        if self.model is None or self.volume is None or self._pending_slice is None:
            return
        z, u8 = self._pending_slice
        rows, cols = u8.shape
        a = self.volume.affine
        corner = lambda i, j: (a @ np.array([i, j, z, 1.0]))[:3]   # noqa: E731
        self._plane_src.SetOrigin(*corner(-0.5, -0.5))
        self._plane_src.SetPoint1(*corner(cols - 0.5, -0.5))
        self._plane_src.SetPoint2(*corner(-0.5, rows - 0.5))
        self._plane_src.Modified()
        self._slice_img.SetDimensions(cols, rows, 1)
        arr = numpy_to_vtk(np.ascontiguousarray(u8).reshape(-1), deep=True)
        self._slice_img.GetPointData().SetScalars(arr)
        self._slice_img.Modified()
        self._texture.Modified()
        self.slice_actor.SetVisibility(self.slice_check.isChecked())
        self.render()

    def update_measurements(self, measurements: Sequence[Measurement]) -> None:
        self._pending_measurements = list(measurements)
        self._apply_measurements()

    def _apply_measurements(self) -> None:
        if not self.available:
            return
        for actor in self._meas_actors:
            self.renderer.RemoveActor(actor)
        self._meas_actors = []
        if self.model is None or self.volume is None or not self.meas_check.isChecked():
            self.render()
            return
        lo, hi = self.model.bounds_lps
        radius = max(0.6, float(np.linalg.norm(hi - lo)) * 0.003)
        a = self.volume.affine
        for m in self._pending_measurements:
            p1 = (a @ np.array([m.p1[0] - 0.5, m.p1[1] - 0.5, m.slice_index, 1.0]))[:3]
            p2 = (a @ np.array([m.p2[0] - 0.5, m.p2[1] - 0.5, m.slice_index, 1.0]))[:3]
            line = vtkLineSource()
            line.SetPoint1(*p1)
            line.SetPoint2(*p2)
            tube = vtkTubeFilter()
            tube.SetInputConnection(line.GetOutputPort())
            tube.SetRadius(radius)
            tube.SetNumberOfSides(14)
            tube.CappingOn()
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(tube.GetOutputPort())
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.16, 0.16)
            actor.GetProperty().SetAmbient(0.35)
            self.renderer.AddActor(actor)
            self._meas_actors.append(actor)
        self.render()


class _WorkerSignals(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)


class Worker(QRunnable):
    """Runs fn(progress_callback) on the global thread pool (loading, inference)."""

    def __init__(self, fn: Callable[[Callable[[int, int], None]], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            result = self.fn(self.signals.progress.emit)
        except Exception as exc:  # report every failure to the UI thread
            traceback.print_exc()
            self.signals.failed.emit(str(exc))
        else:
            self.signals.finished.emit(result)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DICOM Viewer")
        self.resize(1320, 860)
        self.setAcceptDrops(True)
        self.setUnifiedTitleAndToolBarOnMac(True)

        self.volume: Optional[DicomVolume] = None
        self.series_list: list[SeriesInfo] = []
        self._otsu_level = 0.0
        self.segmentation: Optional[np.ndarray] = None
        self.backends: list[SegmentationBackend] = []
        self.slice_index = 0
        self.window_center = 0.0
        self.window_width = 1.0
        self.invert = False
        self._busy = False
        self._workers: set[Worker] = set()  # keep references alive while running
        self._printer: Optional[QPrinter] = None  # reused so paper/orientation choices stick

        self.canvas = ImageCanvas()
        self.panel = ControlPanel()
        self.viewer3d = Viewer3D()
        self._model3d: Optional[AnatomyModel] = None
        self._model3d_token = 0

        # [ sidebar | [ 2D canvas | 3D viewer ] ]
        self.view_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.view_splitter.addWidget(self.canvas)
        self.view_splitter.addWidget(self.viewer3d)
        self.view_splitter.setHandleWidth(2)
        self.view_splitter.setSizes([700, 700])
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.panel)
        splitter.addWidget(self.view_splitter)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 1300])
        splitter.setHandleWidth(1)
        self.setCentralWidget(splitter)


        self._build_menus()
        self._connect()
        self.register_backend(ThresholdBackend())
        self.set_layout("2d")             # 3D is started by start_3d() once the window is up
        self.statusBar().showMessage("Ready")

    # ---- extensibility -----------------------------------------------------------
    def register_backend(self, backend: SegmentationBackend) -> None:
        """Add a segmentation model to the Model drop-down."""
        self.backends.append(backend)
        self.panel.backend_combo.addItem(backend.name)

    # ---- setup -----------------------------------------------------------------------
    def _build_menus(self) -> None:
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        self._add_action(file_menu, "Open File…", self.open_file_dialog, QKeySequence.StandardKey.Open)
        self._add_action(file_menu, "Open Folder…", self.open_folder_dialog, "Ctrl+Shift+O")
        file_menu.addSeparator()
        self._add_action(file_menu, "Export Volume (.npz)…", self.export_volume)
        self._add_action(file_menu, "Export Measurements (.json)…", self.export_measurements)
        file_menu.addSeparator()
        self._add_action(file_menu, "Print…", self.print_report, QKeySequence.StandardKey.Print)
        self._add_action(file_menu, "Print Preview…", self.print_preview, "Ctrl+Alt+P")
        self._add_action(file_menu, "Save Report as PDF…", self.save_report_pdf)
        file_menu.addSeparator()
        self._add_action(file_menu, "Export 3D Model (.glb)…", self.export_3d_model, "Ctrl+E")

        view = mb.addMenu("&View")
        self._add_action(view, "Fit to Window", self.canvas.fit, "F")
        self._add_action(view, "Zoom In", lambda: self.canvas.zoom(1.25), QKeySequence.StandardKey.ZoomIn)
        self._add_action(view, "Zoom Out", lambda: self.canvas.zoom(0.8), QKeySequence.StandardKey.ZoomOut)
        view.addSeparator()
        smooth = self._add_action(view, "Smooth Interpolation", self.canvas.set_smooth, "S")
        smooth.setCheckable(True)
        smooth.setChecked(True)
        overlay = self._add_action(view, "Show Image Annotations", self._toggle_annotations, "I")
        overlay.setCheckable(True)
        overlay.setChecked(True)
        view.addSeparator()
        self._add_action(view, "2D Only", lambda: self.set_layout("2d"), "Ctrl+1")
        self._add_action(view, "2D + 3D Side by Side", lambda: self.set_layout("both"), "Ctrl+2")
        self._add_action(view, "3D Only", lambda: self.set_layout("3d"), "Ctrl+3")
        self._add_action(view, "Rebuild 3D Model", self.build_3d_model, "Ctrl+Shift+R")
        view.addSeparator()
        self._add_action(view, "Next Slice", lambda: self.step_slice(1), ["Down", "Right"])
        self._add_action(view, "Previous Slice", lambda: self.step_slice(-1), ["Up", "Left"])

        tools = mb.addMenu("&Tools")
        self._add_action(tools, "Measure", lambda: self._select_tool(Tool.MEASURE), "M")
        self._add_action(tools, "Pan", lambda: self._select_tool(Tool.PAN), "P")
        self._add_action(tools, "Window / Level", lambda: self._select_tool(Tool.WINDOW), "W")
        tools.addSeparator()
        self._add_action(tools, "Cancel Measurement", self.canvas.cancel_measurement, "Esc")
        self._add_action(tools, "Delete Selected Measurement", self.delete_selected_measurements,
                         ["Backspace", "Delete"])

    def _add_action(self, menu, text, slot, shortcut=None) -> QAction:
        act = QAction(text, self)
        if isinstance(shortcut, (list, tuple)):
            act.setShortcuts([QKeySequence(k) for k in shortcut])
        elif shortcut is not None:
            act.setShortcut(QKeySequence(shortcut))
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    def _connect(self) -> None:
        p, c = self.panel, self.canvas
        p.open_file_btn.clicked.connect(self.open_file_dialog)
        p.open_folder_btn.clicked.connect(self.open_folder_dialog)
        for tool, btn in p.tool_buttons.items():
            btn.clicked.connect(lambda _=False, t=tool: self._select_tool(t))
        p.series_combo.currentIndexChanged.connect(self.load_series)
        p.slice_slider.valueChanged.connect(self.set_slice)
        p.preset_combo.currentTextChanged.connect(self.apply_preset)
        p.center_spin.valueChanged.connect(lambda v: self.set_window(v, self.window_width))
        p.width_spin.valueChanged.connect(lambda v: self.set_window(self.window_center, v))
        p.center_slider.valueChanged.connect(self._center_slider_moved)
        p.width_slider.valueChanged.connect(self._width_slider_moved)
        p.invert_check.toggled.connect(self._set_invert)
        p.measure_list.itemSelectionChanged.connect(self._measurement_selected)
        p.delete_measure_btn.clicked.connect(self.delete_selected_measurements)
        p.clear_measure_btn.clicked.connect(self.clear_measurements)
        p.export_measure_btn.clicked.connect(self.export_measurements)
        p.run_seg_btn.clicked.connect(self.run_segmentation)
        p.print_btn.clicked.connect(self.print_report)
        p.preview_btn.clicked.connect(self.print_preview)
        p.pdf_btn.clicked.connect(self.save_report_pdf)
        p.export_glb_btn.clicked.connect(self.export_3d_model)
        p.iso_auto_check.toggled.connect(self._iso_auto_toggled)
        p.glb_style_combo.currentIndexChanged.connect(self._glb_style_changed)
        self._glb_style_changed()
        self.viewer3d.message.connect(lambda s: self.statusBar().showMessage(s, 6000))
        self.viewer3d.brightRequested.connect(self._compute_bright_regions)
        p.overlay_check.toggled.connect(lambda _: self.render())
        p.opacity_slider.valueChanged.connect(lambda _: self.render())

        c.sliceChangeRequested.connect(self.step_slice)
        c.windowDragged.connect(self._window_dragged)
        c.measurementAdded.connect(self._measurement_added)
        c.cursorInfo.connect(lambda s: self.statusBar().showMessage(s))
        c.set_tool(Tool.MEASURE)

    # ---- loading --------------------------------------------------------------------
    def open_file_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open DICOM file(s)", os.path.expanduser("~"),
            "DICOM (*.dcm *.DCM *.dicom *.ima *.IMA);;All files (*)")
        if paths:
            self.load_paths(paths)

    def open_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open DICOM folder", os.path.expanduser("~"))
        if folder:
            self.load_paths([folder])

    def load_paths(self, paths: list[str]) -> None:
        if self._busy:
            return
        files: list[str] = []
        for p in paths:
            files.extend(DicomLoader.collect_files(p) if os.path.isdir(p) else [p])
        if not files:
            QMessageBox.warning(self, "Nothing to open", "No candidate DICOM files were found.")
            return
        self.statusBar().showMessage(f"Scanning {len(files)} file(s)…")
        self._run_in_background(lambda prog: DicomLoader.scan(files, prog),
                                self._on_scanned, "Could not read DICOM")

    def _on_scanned(self, series: list[SeriesInfo]) -> None:
        """List every series found; load the one with the most images."""
        if not series:
            QMessageBox.warning(self, "No images", "No DICOM images were found in the selection.")
            return
        self.series_list = series
        combo = self.panel.series_combo
        combo.blockSignals(True)
        combo.clear()
        for info in series:
            combo.addItem(info.label)
        best = max(range(len(series)), key=lambda i: series[i].n_images)
        combo.setCurrentIndex(best)
        combo.setEnabled(len(series) > 1)
        combo.blockSignals(False)
        self.load_series(best)

    def load_series(self, index: int) -> None:
        if self._busy or not (0 <= index < len(self.series_list)):
            return
        info = self.series_list[index]
        self.statusBar().showMessage(f"Loading {info.label}…")
        self._run_in_background(lambda prog: DicomLoader.load_series(info, prog),
                                self._on_volume_loaded, "Could not load series")

    def _on_volume_loaded(self, volume: DicomVolume) -> None:
        self.volume = volume
        self.segmentation = None
        self._model3d = None
        self._model3d_token += 1
        self.viewer3d.clear()
        self.viewer3d.update_measurements([])
        self.canvas.clear()
        self.panel.measure_list.clear()
        self.canvas.hint.hide()

        lo, hi = volume.value_range
        span = max(hi - lo, 1e-6)
        decimals = 0 if span > 200 else (2 if span > 2 else 4)
        p = self.panel
        for spin in (p.center_spin, p.width_spin):
            spin.blockSignals(True)
            spin.setDecimals(decimals)
            spin.setSingleStep(10.0 ** -decimals if decimals else max(1.0, round(span / 200.0)))
        p.center_spin.setRange(lo - span, hi + span)
        p.width_spin.setRange(10 ** -decimals if decimals else 1.0, span * 4)
        for spin in (p.center_spin, p.width_spin):
            spin.blockSignals(False)

        p.slice_slider.blockSignals(True)
        p.slice_slider.setRange(0, volume.num_slices - 1)
        p.slice_slider.setEnabled(volume.num_slices > 1)
        p.slice_slider.setValue(volume.num_slices // 2)
        p.slice_slider.blockSignals(False)
        self.slice_index = volume.num_slices // 2

        self.invert = volume.invert
        p.invert_check.blockSignals(True)
        p.invert_check.setChecked(self.invert)
        p.invert_check.blockSignals(False)

        p.calib_warning.setVisible(not volume.spacing_calibrated)
        p.run_seg_btn.setEnabled(bool(self.backends))
        for b in (p.print_btn, p.preview_btn, p.pdf_btn):
            b.setEnabled(True)
        # 3D: default iso level = Otsu threshold of this series.
        self._otsu_level = otsu_threshold(volume.voxels)
        p.iso_spin.blockSignals(True)
        p.iso_spin.setDecimals(decimals)
        p.iso_spin.setRange(lo, hi)
        p.iso_spin.setSingleStep(max(span / 200.0, 10.0 ** -decimals))
        p.iso_spin.setValue(self._otsu_level)
        p.iso_spin.blockSignals(False)
        p.iso_auto_check.setChecked(True)
        can_3d = volume.num_slices >= 2
        p.export_glb_btn.setEnabled(can_3d)
        p.export_glb_btn.setToolTip("" if can_3d else
                                    "Needs a series with ≥ 2 slices — use Open Folder… on the series")
        p.set_metadata(volume.metadata)

        # Default window: DICOM tag if present, otherwise robust auto window.
        p.preset_combo.blockSignals(True)
        p.preset_combo.setCurrentIndex(0 if volume.default_window else 1)
        p.preset_combo.blockSignals(False)
        self.apply_preset(p.preset_combo.currentText())

        if self.viewer3d.available and self.viewer3d.isVisible():
            self.build_3d_model()
        name = volume.metadata.get("Series", "") or ""
        self.setWindowTitle(f"DICOM Viewer — {volume.modality} {name}".strip(" —"))
        self.statusBar().showMessage(
            f"Loaded {volume.num_slices} slice(s), {volume.shape[2]}×{volume.shape[1]} px, "
            f"spacing {volume.pixel_spacing[0]:.3g}×{volume.pixel_spacing[1]:.3g} mm", 8000)

    # ---- rendering ------------------------------------------------------------------
    def render(self) -> None:
        """Re-draw the current slice with the current window/level and overlay."""
        v = self.volume
        if v is None:
            return
        values = v.voxels[self.slice_index]
        display = apply_window(values, self.window_center, self.window_width, self.invert)
        self.canvas.set_image(display, values, self.slice_index, v.pixel_spacing, v.units)

        if self.segmentation is not None and self.panel.overlay_check.isChecked():
            rgba = label_overlay_rgba(self.segmentation[self.slice_index],
                                      self.panel.opacity_slider.value() / 100.0)
            self.canvas.set_overlay(rgba)
        else:
            self.canvas.set_overlay(None)
        self._update_annotations()
        self.viewer3d.update_slice(self.slice_index, display)
        self.panel.slice_label.setText(
            f"{self.slice_index + 1} / {v.num_slices}    "
            f"z = {self.slice_index * v.slice_spacing:.1f} mm")

    def _update_annotations(self) -> None:
        v = self.volume
        if v is None:
            return
        md = v.metadata

        def val(k: str) -> str:
            return md.get(k, "—")

        tl = [val("Manufacturer"), f"Se: {val('Series number')}",
              f"Im: {self.slice_index + 1}/{v.num_slices}", val("Series")]
        tr = [val("Patient"), val("Patient ID"), val("Study date"), val("Institution")]
        bl = [f"TR / TE: {val('TR / TE')}" if val("TR / TE") != "—" else "",
              f"Thickness: {val('Slice thickness')}" if val("Slice thickness") != "—" else "",
              f"W: {self.window_width:.6g}  L: {self.window_center:.6g}"]  # zoom line added live by canvas
        br = [f"{v.modality}  {v.shape[2]}×{v.shape[1]}",
              f"{v.pixel_spacing[0]:.3g}×{v.pixel_spacing[1]:.3g} mm/px"]

        def clean(lines: list[str]) -> list[str]:
            return [ln for ln in lines if ln and ln != "—"]

        self.canvas.set_annotations(
            {"tl": clean(tl), "tr": clean(tr), "bl": clean(bl), "br": clean(br)},
            orientation_labels(v.orientation))

    def set_slice(self, index: int) -> None:
        if self.volume is None:
            return
        index = int(np.clip(index, 0, self.volume.num_slices - 1))
        if index == self.slice_index and self.canvas._values is not None:
            return
        self.slice_index = index
        if self.panel.slice_slider.value() != index:
            self.panel.slice_slider.blockSignals(True)
            self.panel.slice_slider.setValue(index)
            self.panel.slice_slider.blockSignals(False)
        self.render()

    def _toggle_annotations(self, on: bool) -> None:
        self.canvas.show_annotations = on
        self.canvas.viewport().update()

    def step_slice(self, step: int) -> None:
        self.set_slice(self.slice_index + step)

    # ---- window-level controls ----------------------------------------------------------------
    def apply_preset(self, name: str) -> None:
        v = self.volume
        if v is None:
            return
        preset = ControlPanel.WINDOW_PRESETS.get(name)
        if preset is None:
            c, w = v.default_window or auto_window(v.voxels)
        elif preset == "auto":
            c, w = auto_window(v.voxels)
        elif preset == "full":
            lo, hi = v.value_range
            c, w = (lo + hi) / 2.0, max(hi - lo, 1e-6)
        else:
            c, w = preset
        self.set_window(c, w)

    def set_window(self, center: float, width: float) -> None:
        width = max(float(width), 1e-6)
        self.window_center, self.window_width = float(center), width
        self._sync_window_widgets()
        self.render()

    def _sync_window_widgets(self) -> None:
        p, v = self.panel, self.volume
        if v is None:
            return
        lo, hi = v.value_range
        span = max(hi - lo, 1e-6)
        steps = ControlPanel.SLIDER_STEPS
        for w, val in ((p.center_spin, self.window_center), (p.width_spin, self.window_width)):
            w.blockSignals(True)
            w.setValue(val)
            w.blockSignals(False)
        for w, val in ((p.center_slider, (self.window_center - lo) / span),
                       (p.width_slider, self.window_width / span)):
            w.blockSignals(True)
            w.setValue(int(np.clip(val, 0, 1) * steps))
            w.blockSignals(False)

    def _center_slider_moved(self, pos: int) -> None:
        if self.volume is None:
            return
        lo, hi = self.volume.value_range
        self.set_window(lo + (hi - lo) * pos / ControlPanel.SLIDER_STEPS, self.window_width)

    def _width_slider_moved(self, pos: int) -> None:
        if self.volume is None:
            return
        lo, hi = self.volume.value_range
        self.set_window(self.window_center, max((hi - lo) * pos / ControlPanel.SLIDER_STEPS, 1e-6))

    def _window_dragged(self, d_center_px: float, d_width_px: float) -> None:
        if self.volume is None:
            return
        lo, hi = self.volume.value_range
        k = max(hi - lo, 1e-6) / 600.0   # data units per screen pixel
        self.set_window(self.window_center + d_center_px * k,
                        self.window_width + d_width_px * k)

    def _set_invert(self, on: bool) -> None:
        self.invert = on
        self.render()

    # ---- tools & measurements --------------------------------------------------------
    def _select_tool(self, tool: str) -> None:
        self.panel.tool_buttons[tool].setChecked(True)
        self.canvas.set_tool(tool)

    def _measurement_added(self, m: Measurement) -> None:
        unit = "mm" if self.volume and self.volume.spacing_calibrated else "px"
        item = QListWidgetItem(f"#{m.id}   {m.distance_mm:.2f} {unit}    slice {m.slice_index + 1}")
        item.setData(Qt.ItemDataRole.UserRole, m.id)
        self.panel.measure_list.addItem(item)
        self._sync_3d_measurements()
        self.statusBar().showMessage(
            f"Distance #{m.id}: {m.distance_mm:.3f} mm  ({m.distance_px:.1f} px)", 10000)

    def _measurement_selected(self) -> None:
        items = self.panel.measure_list.selectedItems()
        if not items:
            self.canvas.highlight_measurement(None)
            return
        mid = items[-1].data(Qt.ItemDataRole.UserRole)
        m = next((x for x in self.canvas.measurements() if x.id == mid), None)
        if m is not None:
            self.set_slice(m.slice_index)
        self.canvas.highlight_measurement(mid)

    def delete_selected_measurements(self) -> None:
        lst = self.panel.measure_list
        for item in lst.selectedItems():
            self.canvas.remove_measurement(item.data(Qt.ItemDataRole.UserRole))
            lst.takeItem(lst.row(item))
        self._sync_3d_measurements()

    def clear_measurements(self) -> None:
        self.canvas.clear_measurements()
        self.panel.measure_list.clear()
        self._sync_3d_measurements()

    def _sync_3d_measurements(self) -> None:
        self.viewer3d.update_measurements(self.canvas.measurements())

    def export_measurements(self) -> None:
        ms = self.canvas.measurements()
        if not ms:
            QMessageBox.information(self, "Export", "There are no measurements to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export measurements", "measurements.json", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"pixel_spacing_mm": self.volume.pixel_spacing if self.volume else None,
                           "measurements": [m.as_dict() for m in ms]}, fh, indent=2)
            self.statusBar().showMessage(f"Saved {len(ms)} measurement(s) to {path}", 6000)

    def export_volume(self) -> None:
        if self.volume is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export volume", "volume.npz", "NumPy archive (*.npz)")
        if path:
            self.volume.save_npz(path)
            if self.segmentation is not None:
                np.save(os.path.splitext(path)[0] + "_labels.npy", self.segmentation)
            self.statusBar().showMessage(f"Exported volume to {path}", 6000)

    # ---- printing -------------------------------------------------------------------------
    def _get_printer(self) -> QPrinter:
        if self._printer is None:
            self._printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            self._printer.setDocName("DICOM measurement report")
        return self._printer

    def _report_ready(self) -> bool:
        if self.volume is None:
            QMessageBox.information(self, "Print", "Open an image first.")
            return False
        self.canvas.cancel_measurement()  # don't print a half-drawn line
        return True

    def _render_report(self, printer: QPrinter) -> None:
        """Paint the report onto any QPrinter (real printer, PDF, or preview)."""
        p = self.panel
        labels = self.segmentation if (self.segmentation is not None and p.overlay_check.isChecked()) else None
        measurements = self.canvas.measurements()
        renderer = ReportRenderer(self.volume, self.window_center, self.window_width, self.invert,
                                  measurements, labels, p.opacity_slider.value() / 100.0,
                                  include_patient=p.print_patient_check.isChecked())
        pages = slices_to_print(measurements, self.slice_index, p.print_all_check.isChecked())
        try:
            renderer.print_to(printer, pages)
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "Print failed", str(exc))

    def print_report(self) -> None:
        if not self._report_ready():
            return
        printer = self._get_printer()
        dlg = QPrintDialog(printer, self)
        dlg.setWindowTitle("Print measurement report")
        if dlg.exec():
            self._render_report(printer)
            self.statusBar().showMessage("Sent report to printer", 5000)

    def print_preview(self) -> None:
        if not self._report_ready():
            return
        dlg = QPrintPreviewDialog(self._get_printer(), self)
        dlg.setWindowTitle("Print preview")
        dlg.resize(980, 1100)
        dlg.paintRequested.connect(self._render_report)
        dlg.exec()

    def save_report_pdf(self) -> None:
        if not self._report_ready():
            return
        series = (self.volume.metadata.get("Series", "") or "report").replace("/", "-").replace(" ", "_")
        default = os.path.join(default_save_dir(), f"{series}_slice{self.slice_index + 1}.pdf")
        path, _ = QFileDialog.getSaveFileName(self, "Save report as PDF", default, "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        pdf = QPrinter(QPrinter.PrinterMode.HighResolution)
        pdf.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        pdf.setOutputFileName(path)
        pdf.setPageLayout(self._get_printer().pageLayout())  # same paper size/orientation as printing
        self._render_report(pdf)
        self.statusBar().showMessage(f"Saved {path}", 6000)

    # ---- 3D export ------------------------------------------------------------------------
    def _glb_style_changed(self, *_args) -> None:
        iso = self.panel.glb_style_combo.currentData() == "iso"
        self.panel.iso_auto_check.setEnabled(iso)
        self.panel.iso_spin.setEnabled(iso and not self.panel.iso_auto_check.isChecked())

    def _iso_auto_toggled(self, auto: bool) -> None:
        self.panel.iso_spin.setEnabled(not auto)
        if auto and self.volume is not None:
            self.panel.iso_spin.setValue(self._otsu_level)

    def export_3d_model(self) -> None:
        v, p = self.volume, self.panel
        if v is None or self._busy:
            return
        if v.num_slices < 2:
            QMessageBox.information(self, "3D export", "A 3D model needs a series with at least 2 slices.\n"
                                    "Use Open Folder… on the series folder.")
            return
        self.canvas.cancel_measurement()
        series = (v.metadata.get("Series", "") or "dicom").replace("/", "-").replace(" ", "_")
        default = os.path.join(default_save_dir(), f"{series}_3d.glb")
        path, _ = QFileDialog.getSaveFileName(self, "Export 3D model", default, "glTF binary (*.glb)")
        if not path:
            return
        if not path.lower().endswith(".glb"):
            path += ".glb"
        v3 = self.viewer3d
        opts = Export3DOptions(
            scheme=p.glb_style_combo.currentData(),
            boundaries=(v3.b1_spin.value(), v3.b2_spin.value()) if v3.available else (4.0, 12.0),
            highlight_percent=v3.bright_spin.value() if v3.available else 2.0,
            iso_level=None if p.iso_auto_check.isChecked() else p.iso_spin.value(),
            detail=int(p.detail_combo.currentData()),
            surface_opacity=p.surface_opacity_slider.value() / 100.0,
            include_slice=p.glb_slice_check.isChecked(),
            slice_index=self.slice_index,
            window=(self.window_center, self.window_width),
            invert=self.invert,
            include_measurements=p.glb_measure_check.isChecked(),
            measurements=list(self.canvas.measurements()),
            labels=self.segmentation if p.glb_seg_check.isChecked() else None,
        )
        open_after = p.glb_open_check.isChecked()
        self.statusBar().showMessage("Building 3D model…")

        def done(result: dict) -> None:
            mb = result["bytes"] / 1e6
            iso = f", iso level {result['iso_level']:.4g}" if result.get("iso_level") is not None else ""
            self.statusBar().showMessage(
                f"Saved {os.path.basename(path)} — {result['triangles']:,} triangles, {mb:.1f} MB{iso}", 12000)
            if open_after:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))

        self._run_in_background(lambda prog: export_glb(v, path, opts, prog), done, "3D export failed")

    # ---- segmentation -----------------------------------------------------------------
    def run_segmentation(self) -> None:
        if self.volume is None or not self.backends or self._busy:
            return
        backend = self.backends[self.panel.backend_combo.currentIndex()]
        vol = self.volume
        self.statusBar().showMessage(f"Running {backend.name}…")

        def job(progress):
            mask = backend.predict(vol, progress)
            if mask.shape != vol.shape:
                raise ValueError(f"Model returned shape {mask.shape}, expected {vol.shape}")
            return vol, mask

        self._run_in_background(job, self._on_segmentation_done, "Segmentation failed")

    def _on_segmentation_done(self, result) -> None:
        vol, mask = result
        if vol is not self.volume:  # a different series was opened meanwhile
            return
        self.segmentation = mask
        self.panel.overlay_check.setChecked(True)
        self.render()
        n = int(np.count_nonzero(mask))
        voxel_ml = float(np.prod(self.volume.spacing_zyx)) / 1000.0 if self.volume else 0.0
        self.statusBar().showMessage(f"Segmentation done: {n} voxels ≈ {n * voxel_ml:.1f} ml", 10000)
        self._compute_label_surfaces()

    # ---- 3D view ---------------------------------------------------------------------------
    def set_layout(self, mode: str) -> None:
        """'2d', 'both' or '3d'."""
        if mode != "2d":
            self.viewer3d.enabled = True      # an explicit menu choice retries a disabled 3D view
            self.viewer3d.setVisible(True)
            if not self.viewer3d.activate():
                self.statusBar().showMessage("3D view unavailable — see the message in the 3D panel", 10000)
                mode = "both"                 # keep the panel visible so the reason is readable
        self.canvas.setVisible(mode != "3d")
        self.viewer3d.setVisible(mode != "2d")
        if mode == "both":
            total = max(2, sum(self.view_splitter.sizes()))
            self.view_splitter.setSizes([total // 2, total - total // 2])
        if mode != "2d" and self.volume is not None and self._model3d is None:
            self.build_3d_model()

    def start_3d(self, disabled_by_flag: bool = False) -> None:
        """Called shortly after the window is shown, so 3D can never block the 2D app."""
        settings = QSettings()
        if disabled_by_flag:
            self.statusBar().showMessage("3D view off (--no-3d). View → 2D + 3D to turn it on.", 10000)
            return
        if settings.value(Viewer3D.PENDING_KEY, False, type=bool):
            settings.setValue(Viewer3D.PENDING_KEY, False)
            log("3D: skipped — it did not finish starting last time")
            QMessageBox.warning(
                self, "3D view turned off",
                "The 3D view did not finish starting last time, so it is off for now.\n\n"
                "You can retry with View → 2D + 3D Side by Side (" + _sc("⌘2", "Ctrl+2") + ").\n"
                "If it fails again: pip install --upgrade vtk  — and send me the Terminal output.")
            return
        self.set_layout("both")

    def _run_side_job(self, fn, on_done, label: str) -> None:
        """Background job that doesn't block the rest of the UI (3D builds)."""
        worker = Worker(fn)
        self._workers.add(worker)

        def ok(result):
            self._workers.discard(worker)
            on_done(result)

        def fail(msg):
            self._workers.discard(worker)
            self.viewer3d.set_busy(f"{label} failed: {msg}")
            self.statusBar().showMessage(f"{label} failed: {msg}", 10000)

        queued = Qt.ConnectionType.QueuedConnection
        worker.signals.finished.connect(ok, queued)
        worker.signals.failed.connect(fail, queued)
        QThreadPool.globalInstance().start(worker)

    def build_3d_model(self) -> None:
        if self.volume is None or not self.viewer3d.available:
            return
        if self.volume.num_slices < 2:
            self.viewer3d.set_busy("3D needs a series with ≥ 2 slices — use Open Folder… on the series")
            return
        self._model3d_token += 1
        token, vol, labels = self._model3d_token, self.volume, self.segmentation
        top = self.viewer3d.bright_spin.value() if self.viewer3d.available else 2.0
        self.viewer3d.set_busy("Building 3D model… (a few seconds)")

        def job(_progress):
            model = AnatomyModel(vol)
            structures = []
            bright = model.bright_regions_layer(top)
            if bright is not None:
                structures.append(bright)
            if labels is not None:
                structures.extend(model.label_layers(labels))
            return token, vol, model, structures

        self._run_side_job(job, self._on_3d_built, "3D build")

    def _on_3d_built(self, result) -> None:
        token, vol, model, structures = result
        if token != self._model3d_token or vol is not self.volume:
            return                     # stale: another series was opened meanwhile
        self._model3d = model
        self.viewer3d.set_model(model, vol)
        for layer in structures:
            self.viewer3d.set_structure(layer)
        self.render()                  # pushes the current slice texture
        self._sync_3d_measurements()

    def _compute_bright_regions(self, top_percent: float) -> None:
        model, token = self._model3d, self._model3d_token
        if model is None:
            return
        self.viewer3d.set_busy(f"Finding the brightest {top_percent:g}% of inner tissue…")

        def done(result):
            tok, layer = result
            if tok != self._model3d_token:
                return
            if layer is None:
                self.viewer3d.remove_structures("bright")
            else:
                self.viewer3d.set_structure(layer)
            self.viewer3d.set_busy("Highlight updated")

        self._run_side_job(lambda _p: (token, model.bright_regions_layer(top_percent)), done, "Highlight")

    def _compute_label_surfaces(self) -> None:
        model, token, labels = self._model3d, self._model3d_token, self.segmentation
        if model is None or labels is None:
            return

        def done(result):
            tok, layers = result
            if tok != self._model3d_token:
                return
            self.viewer3d.remove_structures("seg")
            for layer in layers:
                self.viewer3d.set_structure(layer)

        self._run_side_job(lambda _p: (token, model.label_layers(labels)), done, "Segmentation surfaces")

    def closeEvent(self, event) -> None:
        QSettings().setValue(Viewer3D.PENDING_KEY, False)   # a normal quit is not a failed 3D start
        self.viewer3d.shutdown()
        super().closeEvent(event)

    # ---- threading helper ---------------------------------------------------------------
    def _run_in_background(self, fn, on_done, error_title: str) -> None:
        self._busy = True
        p = self.panel
        p.progress.setRange(0, 0)
        p.progress.setVisible(True)
        worker = Worker(fn)
        self._workers.add(worker)

        def progress(i, n):
            p.progress.setRange(0, n)
            p.progress.setValue(i)

        def finish():
            self._busy = False
            p.progress.setVisible(False)
            self._workers.discard(worker)

        def ok(result):
            finish()
            on_done(result)

        def fail(msg):
            finish()
            self.statusBar().showMessage(error_title, 5000)
            QMessageBox.critical(self, error_title, msg)

        # Queued: the callbacks touch widgets, so they must run on the UI thread.
        queued = Qt.ConnectionType.QueuedConnection
        worker.signals.progress.connect(progress, queued)
        worker.signals.finished.connect(ok, queued)
        worker.signals.failed.connect(fail, queued)
        QThreadPool.globalInstance().start(worker)

    # ---- Qt events ----------------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.load_paths(paths)


STYLESHEET = """
QMainWindow, QWidget { background: #1c1c1f; color: #e6e6e9; font-size: 13px; }
QLabel#title { font-size: 18px; font-weight: 600; color: #ffffff; padding-bottom: 4px; }
QLabel#emptyHint { color: #6f6f78; font-size: 16px; background: transparent; }
QLabel#metaKey { color: #8d8d96; }
QLabel#warning { color: #ffb347; }
QGroupBox { border: 1px solid #2f2f35; border-radius: 8px; margin-top: 14px; padding: 10px 8px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #a0a0aa;
                   font-size: 11px; font-weight: 600; text-transform: uppercase; }
QPushButton, QToolButton { background: #2c2c32; border: 1px solid #3a3a42; border-radius: 6px;
                           padding: 6px 10px; color: #e6e6e9; }
QPushButton:hover, QToolButton:hover { background: #36363d; }
QPushButton:pressed, QToolButton:pressed { background: #232327; }
QPushButton:disabled { color: #66666e; }
QCheckBox { spacing: 6px; }
QToolButton:checked { background: #0a84ff; border-color: #0a84ff; color: white; }
QComboBox, QDoubleSpinBox { background: #26262b; border: 1px solid #3a3a42; border-radius: 6px; padding: 4px 6px; }
QListWidget { background: #151517; border: 1px solid #2f2f35; border-radius: 6px; }
QListWidget::item:selected { background: #0a84ff; color: white; }
QSlider::groove:horizontal { height: 4px; background: #3a3a42; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #0a84ff; border-radius: 2px; }
QSlider::handle:horizontal { background: #f2f2f5; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
QProgressBar { background: #2c2c32; border: none; border-radius: 3px; }
QProgressBar::chunk { background: #0a84ff; border-radius: 3px; }
QStatusBar { background: #151517; color: #a0a0aa; }
QSplitter::handle { background: #2f2f35; }
QWidget#toolbar3d { background: #151517; border-bottom: 1px solid #2f2f35; }
QLabel#title3d { font-weight: 700; color: #ffffff; padding-right: 4px; }
QLabel#status3d { background: #151517; color: #8d8d96; padding: 3px 8px; font-size: 11px; }
QScrollArea { background: transparent; }
QMenuBar { background: #1c1c1f; }
"""


def self_test() -> int:
    """`python3 dicom_viewer.py --selftest`: environment report without opening a window."""
    import platform
    log(f"Python {sys.version.split()[0]} ({platform.machine()}) on macOS {platform.mac_ver()[0] or '?'}"
        f" · {sys.executable}")
    for name in ("numpy", "pydicom", "PyQt6.QtCore", "scipy", "skimage", "vtk"):
        t = time.perf_counter()
        try:
            mod = __import__(name, fromlist=["_"])
            ver = getattr(mod, "__version__", None) or getattr(mod, "PYQT_VERSION_STR", "") \
                or getattr(mod, "VTK_VERSION", "")
            log(f"  {name:13s} {str(ver):10s} import {time.perf_counter() - t:5.2f}s")
        except Exception as exc:
            log(f"  {name:13s} MISSING / BROKEN: {exc}")
    if _load_vtk():
        from vtkmodules.vtkRenderingCore import vtkRenderWindow
        log("  VTK import ok; trying an off-screen OpenGL render (what the 3D view uses)…")
        rw = vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        rw.AddRenderer(vtkRenderer())
        rw.SetSize(64, 64)
        rw.Render()
        log(f"  OpenGL ok: {rw.ReportCapabilities().splitlines()[0] if rw.ReportCapabilities() else 'rendered'}")
        rw.Finalize()
    else:
        log(f"  VTK bridge unavailable: {VTK_IMPORT_ERROR}")
    log("self-test finished")
    return 0


# =============================================================================
# 5. APP LAUNCHERS  (python dicom_viewer.py --install-launcher)  macOS · Windows · Linux
# =============================================================================
#
# Creates ~/Applications/DICOM Viewer.app — a tiny app bundle whose executable
# is a shell script that starts THIS script with THIS Python (your venv), so no
# Terminal or venv activation is needed — plus a Desktop shortcut to it.
# Output is logged to ~/Library/Logs/DICOM Viewer.log.
# (PyInstaller-style freezing is avoided on purpose: VTK + Qt bundles are huge
# and fragile on macOS.)

APP_NAME = "DICOM Viewer"
APP_BUNDLE_ID = "local.dicomviewer.app"
APP_ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#2b3448"/><stop offset="1" stop-color="#0b0e15"/>
    </linearGradient>
    <radialGradient id="scalp" cx="0.5" cy="0.42" r="0.62">
      <stop offset="0" stop-color="#f1eee8"/><stop offset="0.75" stop-color="#b9b3aa"/><stop offset="1" stop-color="#6f6a64"/>
    </radialGradient>
    <radialGradient id="brain" cx="0.5" cy="0.45" r="0.6">
      <stop offset="0" stop-color="#d9d3cb"/><stop offset="1" stop-color="#8f8981"/>
    </radialGradient>
  </defs>
  <rect x="100" y="100" width="824" height="824" rx="185" fill="url(#bg)"/>
  <rect x="100" y="100" width="824" height="824" rx="185" fill="none" stroke="#ffffff" stroke-opacity="0.08" stroke-width="6"/>
  <!-- axial head slice -->
  <ellipse cx="512" cy="522" rx="262" ry="318" fill="url(#scalp)"/>
  <ellipse cx="512" cy="522" rx="236" ry="292" fill="#2a2826"/>
  <ellipse cx="512" cy="522" rx="214" ry="270" fill="url(#brain)"/>
  <!-- gyri -->
  <g fill="none" stroke="#5f5a54" stroke-width="12" stroke-linecap="round" stroke-opacity="0.75">
    <path d="M330 400 q40 -30 70 10 t70 0"/>
    <path d="M555 410 q40 -40 80 0 t60 10"/>
    <path d="M318 560 q35 30 70 -5 t65 10"/>
    <path d="M570 565 q30 35 65 0 t70 -5"/>
    <path d="M360 700 q40 -25 75 5"/>
    <path d="M590 705 q40 -25 75 5"/>
  </g>
  <line x1="512" y1="262" x2="512" y2="786" stroke="#3b3835" stroke-width="12"/>
  <!-- ventricles (bright, like T2) -->
  <path d="M470 470 q-40 60 -10 130 q25 -40 30 -120 z" fill="#ffffff"/>
  <path d="M554 470 q40 60 10 130 q-25 -40 -30 -120 z" fill="#ffffff"/>
  <!-- measurement -->
  <line x1="372" y1="398" x2="648" y2="640" stroke="#ff3b30" stroke-width="20" stroke-linecap="round"/>
  <circle cx="372" cy="398" r="30" fill="#ff3b30" stroke="#ffffff" stroke-width="9"/>
  <circle cx="648" cy="640" r="30" fill="#ff3b30" stroke="#ffffff" stroke-width="9"/>
  <rect x="566" y="282" width="236" height="96" rx="22" fill="#000000" fill-opacity="0.72"/>
  <text x="684" y="349" font-family="Helvetica, Arial, sans-serif" font-weight="700" font-size="58" fill="#ffffff" text-anchor="middle">21 mm</text>
  <text x="512" y="893" font-family="Helvetica Neue, Helvetica, Arial, sans-serif" font-size="40" font-style="italic" letter-spacing="1" fill="#c9d1de" fill-opacity="0.85" text-anchor="middle">created by meş</text>
</svg>"""


def make_app_icon_image(size: int) -> QImage:
    """Render the app icon (SVG) to a square QImage; simple fallback without QtSvg."""
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    try:
        from PyQt6.QtCore import QByteArray
        from PyQt6.QtSvg import QSvgRenderer
        QSvgRenderer(QByteArray(APP_ICON_SVG.encode("utf-8"))).render(p, QRectF(0, 0, size, size))
    except Exception:                                   # QtSvg missing: draw a simple version
        u = size / 1024.0
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(27, 33, 46))
        p.drawRoundedRect(QRectF(100 * u, 100 * u, 824 * u, 824 * u), 185 * u, 185 * u)
        p.setBrush(QColor(190, 184, 176))
        p.drawEllipse(QPointF(512 * u, 522 * u), 262 * u, 318 * u)
        pen = QPen(QColor(255, 59, 48), 20 * u)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(372 * u, 398 * u), QPointF(648 * u, 640 * u))
        f = QFont()
        f.setPixelSize(max(1, int(40 * u)))
        f.setItalic(True)
        p.setFont(f)
        p.setPen(QColor(201, 209, 222))
        p.drawText(QRectF(100 * u, 850 * u, 824 * u, 60 * u),
                   int(Qt.AlignmentFlag.AlignCenter.value), "created by meş")
    p.end()
    return img


def _png_bytes(image: QImage) -> bytes:
    from PyQt6.QtCore import QBuffer, QIODevice
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buf, "PNG")
    return bytes(buf.data())


def write_ico(path: str, pngs: Sequence[tuple[int, bytes]]) -> None:
    """Windows .ico with PNG-compressed images (supported since Windows Vista)."""
    header = struct.pack("<HHH", 0, 1, len(pngs))
    offset = 6 + 16 * len(pngs)
    entries, blobs = b"", b""
    for size, data in pngs:
        entries += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    with open(path, "wb") as fh:
        fh.write(header + entries + blobs)


def _windows_paths() -> tuple[str, str]:
    """(pythonw.exe or python.exe, folder for the icon/log next to the script)."""
    exe_dir = os.path.dirname(sys.executable)
    pyw = os.path.join(exe_dir, "pythonw.exe")      # no console window
    return (pyw if os.path.exists(pyw) else sys.executable), os.path.dirname(os.path.abspath(__file__))


_WIN_SHORTCUT_PS = r"""
$ws = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
  $path = Join-Path $dir ($env:DV_NAME + '.lnk')
  if ($env:DV_REMOVE -eq '1') { if (Test-Path $path) { Remove-Item $path -Force }; continue }
  $lnk = $ws.CreateShortcut($path)
  $lnk.TargetPath = $env:DV_TARGET
  $lnk.Arguments = '"' + $env:DV_SCRIPT + '"'
  $lnk.WorkingDirectory = $env:DV_DIR
  $lnk.IconLocation = $env:DV_ICON + ',0'
  $lnk.Description = 'DICOM viewer and measurement tool'
  $lnk.Save()
  Write-Output $path
}
"""


def _run_powershell(script: str, extra_env: dict) -> "subprocess.CompletedProcess":
    import subprocess
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                          env=dict(os.environ, **extra_env), capture_output=True, text=True,
                          timeout=120, creationflags=flags)


def _install_launcher_windows() -> int:
    """Start-menu + Desktop shortcuts (with icon) that run this script with pythonw.exe."""
    target, folder = _windows_paths()
    ico = os.path.join(folder, f"{APP_NAME}.ico")
    write_ico(ico, [(n, _png_bytes(make_app_icon_image(n))) for n in (16, 24, 32, 48, 64, 128, 256)])
    r = _run_powershell(_WIN_SHORTCUT_PS, {"DV_NAME": APP_NAME, "DV_TARGET": target,
                                           "DV_SCRIPT": os.path.abspath(__file__), "DV_DIR": folder,
                                           "DV_ICON": ico, "DV_REMOVE": "0"})
    if r.returncode != 0:
        log(f"Could not create shortcuts: {r.stderr.strip()}")
        return 1
    log("Shortcuts created:\n  " + "\n  ".join(p for p in r.stdout.splitlines() if p.strip()))
    log(f"Uses Python {target}")
    log(f"Log file:   {os.path.join(folder, 'dicom_viewer.log')}")
    return 0


def _install_launcher_linux() -> int:
    """~/.local/share/applications/dicom-viewer.desktop (menu entry) + Desktop copy."""
    script = os.path.abspath(__file__)
    icon_dir = os.path.expanduser("~/.local/share/icons/hicolor/256x256/apps")
    os.makedirs(icon_dir, exist_ok=True)
    icon = os.path.join(icon_dir, "dicom-viewer.png")
    make_app_icon_image(256).save(icon, "PNG")
    entry = (f"[Desktop Entry]\nType=Application\nName={APP_NAME}\nComment=DICOM viewer and measurement tool\n"
             f'Exec="{sys.executable}" "{script}" %F\nIcon={icon}\nTerminal=false\nCategories=Graphics;Science;MedicalSoftware;\n')
    apps = os.path.expanduser("~/.local/share/applications")
    os.makedirs(apps, exist_ok=True)
    paths = [os.path.join(apps, "dicom-viewer.desktop")]
    desk = default_save_dir()
    if os.path.isdir(desk) and desk != os.path.expanduser("~"):
        paths.append(os.path.join(desk, "dicom-viewer.desktop"))
    for p in paths:
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(entry)
        os.chmod(p, 0o755)
    log("Menu entry created:\n  " + "\n  ".join(paths))
    return 0


def install_launcher() -> int:
    """macOS: .app in ~/Applications + Desktop link · Windows: Start menu + Desktop
    shortcuts · Linux: application-menu entry."""
    if IS_WINDOWS:
        return _install_launcher_windows()
    if not IS_MAC:
        return _install_launcher_linux()
    return _install_launcher_macos()


def _install_launcher_macos() -> int:
    """Create the .app bundle in ~/Applications and a Desktop shortcut."""
    import plistlib
    import shutil
    import subprocess
    import tempfile
    script = os.path.abspath(__file__)
    python = sys.executable                    # venv python (keep the venv path, don't resolve symlinks)
    apps_dir = os.path.expanduser("~/Applications")
    app = os.path.join(apps_dir, f"{APP_NAME}.app")
    contents = os.path.join(app, "Contents")
    if os.path.exists(app):
        shutil.rmtree(app)
    os.makedirs(os.path.join(contents, "MacOS"))
    os.makedirs(os.path.join(contents, "Resources"))

    # 1) executable: a shell script that runs this file with this Python
    exe = os.path.join(contents, "MacOS", APP_NAME)
    with open(exe, "w", encoding="utf-8") as fh:
        fh.write(f"""#!/bin/bash
# Generated by: {python} {script} --install-launcher
# Re-run that command if you move dicom_viewer.py or recreate the venv.
LOG="$HOME/Library/Logs/{APP_NAME}.log"
PY="{python}"
SCRIPT="{script}"
if [ ! -x "$PY" ] || [ ! -f "$SCRIPT" ]; then
  osascript -e 'display alert "{APP_NAME}" message "Python or dicom_viewer.py was moved or deleted.\n\nRun again in Terminal:\npython3 dicom_viewer.py --install-launcher"'
  exit 1
fi
mkdir -p "$HOME/Library/Logs"
cd "$(dirname "$SCRIPT")"
echo "=== $(date) ===" >> "$LOG"
exec "$PY" "$SCRIPT" "$@" >> "$LOG" 2>&1
""")
    os.chmod(exe, 0o755)

    # 2) icon: render PNGs at every size macOS wants, pack with iconutil
    icon_ok = False
    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, "AppIcon.iconset")
        os.makedirs(iconset)
        for base in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                name = f"icon_{base}x{base}{'@2x' if scale == 2 else ''}.png"
                make_app_icon_image(base * scale).save(os.path.join(iconset, name), "PNG")
        icns = os.path.join(contents, "Resources", "AppIcon.icns")
        try:
            subprocess.run(["iconutil", "-c", "icns", iconset, "-o", icns], check=True,
                           capture_output=True, timeout=60)
            icon_ok = True
        except Exception as exc:
            log(f"icon not created ({exc}); the app will use a generic icon")

    # 3) Info.plist
    info = {
        "CFBundleName": APP_NAME, "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": APP_BUNDLE_ID, "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL", "CFBundleShortVersionString": "1.0", "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "11.0", "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "DICOM Viewer — research use only, not for diagnosis",
    }
    if icon_ok:
        info["CFBundleIconFile"] = "AppIcon"
    with open(os.path.join(contents, "Info.plist"), "wb") as fh:
        plistlib.dump(info, fh)

    # 4) register with Launch Services (icon + Spotlight/Launchpad) and add a Desktop shortcut
    lsregister = ("/System/Library/Frameworks/CoreServices.framework/Frameworks/"
                  "LaunchServices.framework/Support/lsregister")
    for cmd in (["touch", app], [lsregister, "-f", app]):
        try:
            subprocess.run(cmd, capture_output=True, timeout=30)
        except Exception:
            pass
    desktop_link = os.path.expanduser(f"~/Desktop/{APP_NAME}")
    try:
        if os.path.islink(desktop_link) or os.path.exists(desktop_link):
            os.remove(desktop_link)
        os.symlink(app, desktop_link)
        desktop_msg = f"Desktop shortcut:  {desktop_link}"
    except OSError as exc:
        desktop_msg = f"Desktop shortcut not created ({exc}) — drag the app from ~/Applications instead"

    log(f"Installed:  {app}")
    log(desktop_msg)
    log(f"Uses Python {python}")
    log(f"Log file:   ~/Library/Logs/{APP_NAME}.log")
    log("Tip: drag the app from ~/Applications to the Dock to keep it there.")
    return 0


def uninstall_launcher() -> int:
    import shutil
    if IS_WINDOWS:
        target, folder = _windows_paths()
        _run_powershell(_WIN_SHORTCUT_PS, {"DV_NAME": APP_NAME, "DV_TARGET": target, "DV_SCRIPT": "",
                                           "DV_DIR": folder, "DV_ICON": "", "DV_REMOVE": "1"})
        log("Shortcuts removed")
        return 0
    if not IS_MAC:
        for p in (os.path.expanduser("~/.local/share/applications/dicom-viewer.desktop"),
                  os.path.join(default_save_dir(), "dicom-viewer.desktop")):
            if os.path.exists(p):
                os.remove(p)
        log("Menu entry removed")
        return 0
    app = os.path.expanduser(f"~/Applications/{APP_NAME}.app")
    link = os.path.expanduser(f"~/Desktop/{APP_NAME}")
    if os.path.islink(link):
        os.remove(link)
    if os.path.exists(app):
        shutil.rmtree(app)
    log("Launcher removed")
    return 0


def _ensure_output_streams() -> None:
    """pythonw.exe (Windows shortcuts) has no console: sys.stdout/stderr are None,
    which would break faulthandler and tracebacks. Send them to a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    folder = os.path.dirname(os.path.abspath(__file__))
    try:
        fh = open(os.path.join(folder, "dicom_viewer.log"), "a", encoding="utf-8", buffering=1)
    except OSError:
        fh = open(os.devnull, "w")
    sys.stdout = sys.stdout or fh
    sys.stderr = sys.stderr or fh
    print(f"=== {datetime.now():%Y-%m-%d %H:%M:%S} ===", file=sys.stderr)


def main() -> int:
    global DEBUG
    _ensure_output_streams()
    faulthandler.enable()                      # print a traceback on hard crashes (segfaults)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    DEBUG = "--debug" in flags
    if "--selftest" in flags:
        return self_test()
    if "--uninstall-launcher" in flags:
        return uninstall_launcher()
    if "--install-launcher" in flags:
        _app = QApplication([sys.argv[0]])     # needed to render the icon
        return install_launcher()
    no_3d = "--no-3d" in flags or os.environ.get("DICOM_VIEWER_NO_3D") == "1"
    if pydicom is None:
        print("pydicom is required:  pip install pydicom", file=sys.stderr)
        return 1

    # If start-up hangs, dump every thread's stack to Terminal after 45 s.
    faulthandler.dump_traceback_later(45, exit=False)
    if IS_WINDOWS:   # own taskbar group + icon instead of Python's
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_BUNDLE_ID)
        except Exception:
            pass
    log("starting Qt…")
    app = QApplication([sys.argv[0]])
    app.setApplicationName("DICOM Viewer")
    app.setOrganizationName("DicomViewer")
    app.setStyle("Fusion")          # consistent dark styling on macOS
    app.setStyleSheet(STYLESHEET)
    try:
        app.setWindowIcon(QIcon(QPixmap.fromImage(make_app_icon_image(512))))   # Dock / taskbar icon
    except Exception:
        pass
    win = MainWindow()
    screen = app.primaryScreen().availableGeometry()
    win.resize(int(min(1600, screen.width() * 0.96)), int(min(960, screen.height() * 0.94)))
    win.move(screen.center() - win.rect().center())
    win.show()
    win.raise_()                    # macOS: bring the window in front of Terminal
    win.activateWindow()
    log("window shown")

    def after_start() -> None:
        faulthandler.cancel_dump_traceback_later()
        log("event loop running")
        win.start_3d(disabled_by_flag=no_3d)
        if args:
            win.load_paths(args)

    QTimer.singleShot(300, after_start)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
