#!/usr/bin/env python3
"""Validate VTUAV-ST layout and sparse-annotation frame alignment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Set

import numpy as np


ANNOTATION_STRIDE = 10
EXPECTED_SEQUENCE_COUNTS = {"train": 207, "test": 176}
BUNDLED_MAPPING = (
    Path(__file__).resolve().parents[1]
    / "trackit/datasets/MMOT/datasets/vtuav_st_init_frame.npy"
)


def load_boxes(path: Path) -> np.ndarray:
    boxes = np.loadtxt(path, dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise RuntimeError(f"Expected an Nx4 annotation file, got {boxes.shape}: {path}")
    if not np.isfinite(boxes).all():
        raise RuntimeError(f"Annotation contains NaN or Inf: {path}")
    return boxes


def numeric_frame_ids(directory: Path) -> Set[int]:
    valid_suffixes = {".jpg", ".jpeg", ".png"}
    frame_ids = set()
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in valid_suffixes:
            continue
        if path.name.startswith("._"):
            continue
        try:
            frame_id = int(path.stem)
        except ValueError as exc:
            raise RuntimeError(f"Non-numeric frame name: {path}") from exc
        if frame_id in frame_ids:
            raise RuntimeError(f"Duplicate numeric frame id {frame_id}: {directory}")
        frame_ids.add(frame_id)
    if not frame_ids:
        raise RuntimeError(f"No image frames found: {directory}")
    return frame_ids


def load_mapping(path: Path) -> Dict[str, int]:
    if not path.is_file():
        raise RuntimeError(f"Init-frame mapping does not exist: {path}")
    mapping = np.load(path, allow_pickle=True).item()
    if not isinstance(mapping, dict):
        raise RuntimeError(f"Init-frame mapping must contain a dict: {path}")
    result = {str(name): int(offset) for name, offset in mapping.items()}
    invalid = {name: offset for name, offset in result.items() if offset < 0}
    if invalid:
        raise RuntimeError(f"Negative init-frame offsets: {invalid}")
    return result


def sequence_names(split_root: Path) -> Iterable[str]:
    return sorted(
        path.name
        for path in split_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def validate_split(root: Path, split: str, mapping: Dict[str, int], strict_count: bool) -> None:
    directory_name = "trainingset" if split == "train" else "testingset"
    split_root = root / directory_name
    if not split_root.is_dir():
        raise RuntimeError(f"Missing split directory: {split_root}")

    names = list(sequence_names(split_root))
    expected_count = EXPECTED_SEQUENCE_COUNTS[split]
    if strict_count and len(names) != expected_count:
        raise RuntimeError(
            f"Unexpected {split} sequence count: {len(names)}; expected {expected_count}"
        )

    total_frames = 0
    total_annotations = 0
    invalid_annotations = 0
    nonzero_offsets = 0

    for name in names:
        sequence_root = split_root / name
        rgb_dir = sequence_root / "rgb"
        ir_dir = sequence_root / "ir"
        rgb_gt_path = sequence_root / "rgb.txt"
        ir_gt_path = sequence_root / "ir.txt"
        for path in (rgb_dir, ir_dir, rgb_gt_path, ir_gt_path):
            if not path.exists():
                raise RuntimeError(f"Missing sequence data: {path}")

        rgb_ids = numeric_frame_ids(rgb_dir)
        ir_ids = numeric_frame_ids(ir_dir)
        if rgb_ids != ir_ids:
            raise RuntimeError(
                f"RGB/TIR frame mismatch in {name}: "
                f"RGB-only={sorted(rgb_ids - ir_ids)[:10]}, "
                f"TIR-only={sorted(ir_ids - rgb_ids)[:10]}"
            )

        rgb_boxes = load_boxes(rgb_gt_path)
        ir_boxes = load_boxes(ir_gt_path)
        if len(rgb_boxes) != len(ir_boxes):
            raise RuntimeError(
                f"RGB/TIR annotation length mismatch in {name}: "
                f"RGB={len(rgb_boxes)}, TIR={len(ir_boxes)}"
            )

        offset = mapping.get(name, 0) if split == "train" else min(rgb_ids)
        nonzero_offsets += int(offset != 0)
        expected_ids = {
            offset + ANNOTATION_STRIDE * index for index in range(len(rgb_boxes))
        }
        missing_ids = sorted(expected_ids - rgb_ids)
        if missing_ids:
            raise RuntimeError(
                f"Annotation mapping points to missing frames in {name}; "
                f"offset={offset}, examples={missing_ids[:10]}"
            )

        paired_valid = (
            (rgb_boxes[:, 2] > 0)
            & (rgb_boxes[:, 3] > 0)
            & (ir_boxes[:, 2] > 0)
            & (ir_boxes[:, 3] > 0)
        )
        invalid_annotations += int((~paired_valid).sum())
        total_frames += len(rgb_ids)
        total_annotations += len(rgb_boxes)

    print(
        f"{split}: {len(names)} sequences, {total_frames} paired frames, "
        f"{total_annotations} sparse annotations, "
        f"{invalid_annotations} invalid paired annotations, "
        f"{nonzero_offsets} nonzero start offsets"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="VTUAV-ST dataset root")
    parser.add_argument(
        "--split", choices=("train", "test", "all"), default="all",
        help="Dataset split to validate",
    )
    parser.add_argument(
        "--mapping", type=Path, default=BUNDLED_MAPPING,
        help="init_frame.npy used for sparse training annotations",
    )
    parser.add_argument(
        "--allow-count-mismatch", action="store_true",
        help="Do not require exactly 207 train and 176 test sequences",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"VTUAV-ST root does not exist: {root}")

    splits = ("train", "test") if args.split == "all" else (args.split,)
    mapping = load_mapping(args.mapping.expanduser().resolve()) if "train" in splits else {}
    for split in splits:
        validate_split(root, split, mapping, not args.allow_count_mismatch)
    print("VTUAV-ST validation passed")


if __name__ == "__main__":
    main()
