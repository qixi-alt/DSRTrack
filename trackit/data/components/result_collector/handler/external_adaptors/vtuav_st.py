"""Result writer compatible with the VTUAV short-term evaluation workflow.

The official HMFT VTUAV runner saves one full-frame ``sequence.txt`` file per
video.  This adaptor mirrors that convention: all predicted boxes are written
in XYWH format, one box per original test frame.  The generated zip can be
extracted and its text files placed in the official VTUAV ``BB_results``
location before running ``GenerateMat_ST.m`` / ``plot_ST.m``.
"""

import io
import os
import zipfile
from typing import Optional, Sequence

import numpy as np

from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT

from ...progress_tracer import EvaluationProgress
from .. import EvaluationResultHandler


class VTUAVSTTrackingResultWriter:
    def __init__(self, output_folder: str, file_name: str):
        os.makedirs(output_folder, exist_ok=True)
        self._zip_path = os.path.join(output_folder, file_name + ".zip")
        self._zip_file = zipfile.ZipFile(self._zip_path, "w", zipfile.ZIP_DEFLATED)
        self._duplication_check = set()

    def write(self, sequence_name: str, predicted_bboxes_xywh: np.ndarray):
        if sequence_name in self._duplication_check:
            raise RuntimeError(f"Duplicate VTUAV-ST result for sequence: {sequence_name}")
        self._duplication_check.add(sequence_name)

        with io.StringIO() as handle:
            # Match the simple whitespace-separated format used by the public
            # HMFT run_VTUAV.py result writer.
            np.savetxt(handle, predicted_bboxes_xywh, fmt="%.6f")
            self._zip_file.writestr(f"{sequence_name}.txt", handle.getvalue())

    def close(self):
        self._zip_file.close()


class VTUAVSTEvaluationToolAdaptor(EvaluationResultHandler):
    def __init__(self, tracker_name: str, output_folder: str, file_name: str,
                 rasterize_bbox: bool = False):
        del tracker_name  # Official VTUAV files are keyed by sequence name.
        self._writer = VTUAVSTTrackingResultWriter(output_folder, file_name)
        self._rasterize_bbox = rasterize_bbox

    def accept(self, evaluation_results: Sequence[SequenceEvaluationResult_SOT],
               evaluation_progresses: Sequence[EvaluationProgress]):
        for evaluation_result, evaluation_progress in zip(evaluation_results, evaluation_progresses):
            if evaluation_progress.this_dataset is not None and \
                    evaluation_progress.this_dataset.total_repeat_times != 1:
                raise RuntimeError("VTUAV-ST official result export supports one run per sequence")

            predicted_bboxes = evaluation_result.output_box
            if predicted_bboxes is None:
                raise RuntimeError(
                    f"No predicted boxes for VTUAV-ST sequence "
                    f"{evaluation_result.sequence_info.sequence_name}"
                )

            expected_indices = np.arange(len(predicted_bboxes), dtype=evaluation_result.evaluated_frame_indices.dtype)
            if not np.array_equal(evaluation_result.evaluated_frame_indices, expected_indices):
                raise RuntimeError(
                    f"VTUAV-ST predictions are not a contiguous full-frame stream for "
                    f"{evaluation_result.sequence_info.sequence_name}"
                )

            if self._rasterize_bbox:
                predicted_bboxes = bbox_rasterize(predicted_bboxes)
            predicted_bboxes = bbox_xyxy_to_xywh(predicted_bboxes)

            self._writer.write(
                evaluation_result.sequence_info.sequence_name,
                predicted_bboxes,
            )

    def close(self):
        self._writer.close()
