"""VTUAV short-term (VTUAV-ST) dataset support for DSRTrack.

Training uses the sparse VTUAV annotations. Annotation row ``k`` is mapped to
video frame ``offset + 10 * k``. Sequence-specific offsets are loaded from an
explicit ``init_frame.npy``, the dataset directory, or the mapping distributed
with the public HMFT implementation and bundled with this loader.

Testing always runs the tracker on the full-rate RGB/TIR frame stream.  Sparse
annotations are attached only to their annotated frames and are not used to
skip frames during inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image

from trackit.datasets.common.seed import BaseSeed
from trackit.datasets.MMOT.constructor import MultiModalObjectTrackingDatasetConstructor


class VTUAV_ST_Seed(BaseSeed):
    ANNOTATION_STRIDE = 10
    EXPECTED_SEQUENCE_COUNTS = {
        "train": 207,
        "test": 176,
    }

    def __init__(self, root_path: str = None, data_split: str = ("train", "test"),
                 init_frame_path: str = None, strict_sequence_count: bool = True):
        if root_path is None:
            root_path = self.get_path_from_config("VTUAV_ST_PATH")

        self.init_frame_path = init_frame_path
        self.strict_sequence_count = bool(strict_sequence_count)

        super(VTUAV_ST_Seed, self).__init__(
            "VTUAV_ST",
            root_path,
            supported_data_splits=("train", "test"),
            data_split=data_split,
            # Version 2 adds the official train offsets and first-frame-based
            # initialization for full-rate test sequences.
            version=2,
        )

    @staticmethod
    def _load_boxes(path: Path) -> np.ndarray:
        boxes = np.loadtxt(path, dtype=np.float64)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise RuntimeError(f"VTUAV-ST annotation must be Nx4, got {boxes.shape}: {path}")
        if not np.isfinite(boxes).all():
            raise RuntimeError(f"VTUAV-ST annotation contains NaN/Inf: {path}")
        return boxes

    @staticmethod
    def _numeric_image_map(directory: Path) -> Dict[int, Path]:
        image_map: Dict[int, Path] = {}
        valid_suffixes = {".jpg", ".jpeg", ".png"}
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in valid_suffixes:
                continue
            # Ignore macOS AppleDouble metadata files if one is copied in later.
            if path.name.startswith("._"):
                continue
            try:
                frame_id = int(path.stem)
            except ValueError as exc:
                raise RuntimeError(f"Non-numeric VTUAV-ST frame name: {path}") from exc
            if frame_id in image_map:
                raise RuntimeError(f"Duplicate VTUAV-ST frame id {frame_id}: {directory}")
            image_map[frame_id] = path
        if not image_map:
            raise RuntimeError(f"No image frames found in {directory}")
        return image_map

    def _load_init_frame_mapping(self) -> Tuple[Dict[str, int], Optional[Path]]:
        candidates = []
        if self.init_frame_path:
            explicit_path = Path(self.init_frame_path)
            if not explicit_path.is_file():
                raise RuntimeError(
                    f"Configured VTUAV-ST init-frame mapping does not exist: {explicit_path}"
                )
            candidates.append(explicit_path)
        root = Path(self.root_path)
        candidates.extend((
            root / "init_frame.npy",
            root.parent / "init_frame.npy",
            Path(__file__).with_name("vtuav_st_init_frame.npy"),
        ))

        visited_paths = set()
        for path in candidates:
            path = path.expanduser().resolve()
            if path in visited_paths:
                continue
            visited_paths.add(path)
            if not path.is_file():
                continue
            try:
                mapping = np.load(path, allow_pickle=True).item()
            except Exception as exc:
                raise RuntimeError(f"Cannot load VTUAV-ST init-frame mapping: {path}") from exc
            if not isinstance(mapping, dict):
                raise RuntimeError(f"init_frame.npy must contain a dict: {path}")
            result = {}
            for sequence_name, offset in mapping.items():
                offset = int(offset)
                if offset < 0:
                    raise RuntimeError(
                        f"Negative VTUAV-ST init offset for {sequence_name}: {offset}"
                    )
                result[str(sequence_name)] = offset
            return result, path

        raise RuntimeError(
            "VTUAV-ST init-frame mapping was not found. Training on the original "
            "full-frame archive requires sequence-specific offsets. Place the "
            "official init_frame.npy in VTUAV_ST_PATH or restore the bundled "
            "vtuav_st_init_frame.npy file."
        )

    @staticmethod
    def _read_sequence_image_size(rgb_path: Path, ir_path: Path) -> Tuple[int, int]:
        # VTUAV RGB/TIR frames are spatially aligned.  Supplying the size here
        # prevents the generic dataset constructor from opening every one of the
        # >1M full-rate test frames merely to query image dimensions.
        with Image.open(rgb_path) as image_rgb:
            rgb_size = tuple(image_rgb.size)
        with Image.open(ir_path) as image_ir:
            ir_size = tuple(image_ir.size)
        if rgb_size != ir_size:
            raise RuntimeError(
                f"VTUAV-ST RGB/TIR image-size mismatch: RGB={rgb_size}, "
                f"IR={ir_size}, files=({rgb_path}, {ir_path})"
            )
        return rgb_size

    @staticmethod
    def _paired_box_is_valid(rgb_box: np.ndarray, ir_box: np.ndarray) -> bool:
        return bool(
            rgb_box[2] > 0 and rgb_box[3] > 0 and
            ir_box[2] > 0 and ir_box[3] > 0
        )

    def _get_split_root(self) -> Tuple[str, Path]:
        if len(self.data_split) != 1:
            raise RuntimeError(f"VTUAV-ST expects exactly one split, got {self.data_split}")
        split = self.data_split[0]
        if split == "train":
            split_root = Path(self.root_path) / "trainingset"
        elif split == "test":
            split_root = Path(self.root_path) / "testingset"
        else:
            raise NotImplementedError(f"Unsupported VTUAV-ST split: {split}")
        if not split_root.is_dir():
            raise RuntimeError(f"VTUAV-ST split directory does not exist: {split_root}")
        return split, split_root

    def _get_sequence_names(self, split: str, split_root: Path) -> Iterable[str]:
        sequence_names = sorted(
            path.name for path in split_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        expected_count = self.EXPECTED_SEQUENCE_COUNTS[split]
        if self.strict_sequence_count and len(sequence_names) != expected_count:
            raise RuntimeError(
                f"Unexpected VTUAV-ST {split} sequence count: {len(sequence_names)}; "
                f"expected {expected_count}. Check that only the short-term subset "
                f"was extracted into {split_root}."
            )
        return sequence_names

    def construct(self, constructor: MultiModalObjectTrackingDatasetConstructor):
        split, split_root = self._get_split_root()
        sequence_names = list(self._get_sequence_names(split, split_root))
        if split == "train":
            init_frame_mapping, init_frame_source = self._load_init_frame_mapping()
            print(f"VTUAV-ST: using init-frame mapping from {init_frame_source}")
        else:
            init_frame_mapping = {}
            print("VTUAV-ST: initializing each test sequence from its first available frame")

        constructor.set_total_number_of_sequences(len(sequence_names))
        constructor.set_bounding_box_format("XYWH")

        for sequence_name in sequence_names:
            sequence_path = split_root / sequence_name
            rgb_dir = sequence_path / "rgb"
            ir_dir = sequence_path / "ir"
            rgb_gt_path = sequence_path / "rgb.txt"
            ir_gt_path = sequence_path / "ir.txt"

            for required_path in (rgb_dir, ir_dir, rgb_gt_path, ir_gt_path):
                if not required_path.exists():
                    raise RuntimeError(f"Missing VTUAV-ST data: {required_path}")

            rgb_boxes = self._load_boxes(rgb_gt_path)
            ir_boxes = self._load_boxes(ir_gt_path)
            if len(rgb_boxes) != len(ir_boxes):
                raise RuntimeError(
                    f"VTUAV-ST RGB/IR GT length mismatch in {sequence_name}: "
                    f"RGB={len(rgb_boxes)}, IR={len(ir_boxes)}"
                )

            rgb_frames = self._numeric_image_map(rgb_dir)
            ir_frames = self._numeric_image_map(ir_dir)
            if set(rgb_frames) != set(ir_frames):
                rgb_only = sorted(set(rgb_frames) - set(ir_frames))[:10]
                ir_only = sorted(set(ir_frames) - set(rgb_frames))[:10]
                raise RuntimeError(
                    f"VTUAV-ST RGB/IR frame mismatch in {sequence_name}: "
                    f"RGB-only={rgb_only}, IR-only={ir_only}"
                )

            if split == "train":
                offset = int(init_frame_mapping.get(sequence_name, 0))
            else:
                # The official test runner initializes from the first frame in
                # each sequence. Do not apply train-only init-frame offsets here.
                offset = min(rgb_frames)
            annotated_frame_ids = [
                offset + self.ANNOTATION_STRIDE * annotation_index
                for annotation_index in range(len(rgb_boxes))
            ]
            missing_annotated_frames = [
                frame_id for frame_id in annotated_frame_ids
                if frame_id not in rgb_frames
            ]
            if missing_annotated_frames:
                raise RuntimeError(
                    f"VTUAV-ST sparse annotation mapping points to missing frames in "
                    f"{sequence_name}; offset={offset}, examples={missing_annotated_frames[:10]}"
                )

            with constructor.new_sequence() as sequence_constructor:
                sequence_constructor.set_name(sequence_name)

                if split == "train":
                    # The DSRTrack MMOT pipeline uses one common target box for the
                    # aligned RGB/TIR pair.  Following the public HMFT RGBT loader,
                    # rgb.txt is used as the common box; ir.txt is used to reject
                    # modality-invalid samples.
                    first_frame_id = annotated_frame_ids[0]
                    image_size = self._read_sequence_image_size(
                        rgb_frames[first_frame_id], ir_frames[first_frame_id]
                    )

                    for annotation_index, frame_id in enumerate(annotated_frame_ids):
                        rgb_box = rgb_boxes[annotation_index]
                        ir_box = ir_boxes[annotation_index]
                        valid = self._paired_box_is_valid(rgb_box, ir_box)

                        with sequence_constructor.new_frame() as frame_constructor:
                            frame_constructor.set_path(
                                (str(rgb_frames[frame_id]), str(ir_frames[frame_id])),
                                image_size=image_size,
                            )
                            frame_constructor.set_bounding_box(rgb_box, validity=valid)

                else:
                    # Test on every original video frame from the sequence's
                    # first available frame, which corresponds to the first GT.
                    source_frame_ids = sorted(rgb_frames)

                    first_frame_id = source_frame_ids[0]
                    assert first_frame_id == offset
                    if not self._paired_box_is_valid(rgb_boxes[0], ir_boxes[0]):
                        raise RuntimeError(
                            f"VTUAV-ST test sequence has invalid first GT and cannot be initialized: "
                            f"{sequence_name}"
                        )

                    image_size = self._read_sequence_image_size(
                        rgb_frames[first_frame_id], ir_frames[first_frame_id]
                    )
                    annotation_by_source_frame = {
                        frame_id: annotation_index
                        for annotation_index, frame_id in enumerate(annotated_frame_ids)
                    }

                    for frame_id in source_frame_ids:
                        with sequence_constructor.new_frame() as frame_constructor:
                            frame_constructor.set_path(
                                (str(rgb_frames[frame_id]), str(ir_frames[frame_id])),
                                image_size=image_size,
                            )
                            annotation_index = annotation_by_source_frame.get(frame_id)
                            if annotation_index is not None:
                                rgb_box = rgb_boxes[annotation_index]
                                ir_box = ir_boxes[annotation_index]
                                valid = self._paired_box_is_valid(rgb_box, ir_box)
                                frame_constructor.set_bounding_box(rgb_box, validity=valid)
