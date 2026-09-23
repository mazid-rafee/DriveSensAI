#!/usr/bin/env python3
"""Build ``*_v3.csv`` files from legacy or v2 Apple Vision CSVs.

Schema v3 keeps eye-validity flags and drops pupil coordinates.
Prefer converting from ``*_v2.csv`` (already eye-local) when available;
otherwise alias from legacy Absolute gap / EAR fields.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _SCRIPT_DIR.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from feature_contract import CSV_SUFFIX, DROWSINESS_FEATURE_NAMES  # noqa: E402

LEGACY_SUFFIX = "_rgb_face.apple_drowsiness.csv"
V2_SUFFIX = "_rgb_face.apple_drowsiness_v2.csv"

LEGACY_ALIASES = {
    "face_detected": "face_detected",
    "yaw": "yaw",
    "pitch": "pitch",
    "roll": "roll",
    "left_eye_valid": "left_eye_valid",
    "right_eye_valid": "right_eye_valid",
    "left_eye_aspect_ratio": "left_eye_aspect_ratio",
    "right_eye_aspect_ratio": "right_eye_aspect_ratio",
    "left_eyelid_gap": "left_eyelid_gap_ratio",
    "right_eyelid_gap": "right_eyelid_gap_ratio",
}


def _finite(raw: str | None, default: float = 0.0) -> float:
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).strip())
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return value


def _finalize_row(out: dict[str, float]) -> dict[str, float]:
    face = 1.0 if out["face_detected"] >= 0.5 else 0.0
    out["face_detected"] = face
    if face < 0.5:
        for name in DROWSINESS_FEATURE_NAMES:
            out[name] = 0.0
        return out
    for side in ("left", "right"):
        valid_key = f"{side}_eye_valid"
        valid = 1.0 if out[valid_key] >= 0.5 else 0.0
        out[valid_key] = valid
        if valid < 0.5:
            out[f"{side}_eye_aspect_ratio"] = 0.0
            out[f"{side}_eyelid_gap_ratio"] = 0.0
        else:
            out[f"{side}_eye_aspect_ratio"] = max(0.0, out[f"{side}_eye_aspect_ratio"])
            out[f"{side}_eyelid_gap_ratio"] = max(0.0, out[f"{side}_eyelid_gap_ratio"])
    return out


def convert_from_v2(src: Path, dst: Path) -> int:
    with src.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{src.name}: empty header")
        rows_out = []
        for row in reader:
            out = {"frame_index": int(str(row["frame_index"]).strip())}
            for name in DROWSINESS_FEATURE_NAMES:
                out[name] = _finite(row.get(name))
            rows_out.append(_finalize_row(out))

    fieldnames = ["frame_index", *DROWSINESS_FEATURE_NAMES]
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    return len(rows_out)


def convert_from_legacy(src: Path, dst: Path) -> int:
    with src.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{src.name}: empty header")
        rows_out = []
        for row in reader:
            out = {"frame_index": int(str(row["frame_index"]).strip())}
            for legacy, v3_name in LEGACY_ALIASES.items():
                out[v3_name] = _finite(row.get(legacy))
            rows_out.append(_finalize_row(out))

    fieldnames = ["frame_index", *DROWSINESS_FEATURE_NAMES]
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    return len(rows_out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--landmarks-dir",
        type=Path,
        default=None,
        help="Source directory (legacy or v2 CSVs). Default: package data/landmarks_v2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination for v3 CSVs (default: <package>/data/landmarks_v3).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    landmarks_dir = (
        args.landmarks_dir.expanduser().resolve()
        if args.landmarks_dir is not None
        else (_PACKAGE_ROOT / "data" / "landmarks_v2")
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (_PACKAGE_ROOT / "data" / "landmarks_v3")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    v2_files = sorted(landmarks_dir.glob(f"*{V2_SUFFIX}"))
    legacy_files = sorted(landmarks_dir.glob(f"*{LEGACY_SUFFIX}"))
    converted = 0

    if v2_files:
        for src in v2_files:
            stem = src.name[: -len(V2_SUFFIX)]
            dst = output_dir / f"{stem}{CSV_SUFFIX}"
            if dst.exists() and not args.force:
                print(f"skip existing {dst.name}")
                continue
            n = convert_from_v2(src, dst)
            print(f"wrote {dst.name} rows={n} (from v2)")
            converted += 1
    elif legacy_files:
        for src in legacy_files:
            stem = src.name[: -len(LEGACY_SUFFIX)]
            dst = output_dir / f"{stem}{CSV_SUFFIX}"
            if dst.exists() and not args.force:
                print(f"skip existing {dst.name}")
                continue
            n = convert_from_legacy(src, dst)
            print(f"wrote {dst.name} rows={n} (from legacy)")
            converted += 1
        print(
            "WARNING: eyelid_gap_ratio aliased from legacy absolute gap columns; "
            "prefer re-extraction when possible."
        )
    else:
        print(f"no v2/legacy CSVs found in {landmarks_dir}", file=sys.stderr)
        return 1

    print(f"converted={converted} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
