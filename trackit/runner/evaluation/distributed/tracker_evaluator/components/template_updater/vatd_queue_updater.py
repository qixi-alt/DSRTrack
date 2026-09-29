from typing import Tuple, Dict, Any, List
import torch
import numpy as np
import math
from collections import deque

from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.core.utils.siamfc_cropping import get_siamfc_cropping_params, apply_siamfc_cropping
from . import TemplateUpdater


class VATDQueueUpdater(TemplateUpdater):
    """
    Volatility-Aware Temporal Decay (VATD) Queue Updater
    """

    def __init__(
            self,
            update_interval: int,
            update_threshold: float,
            queue_size: int,
            template_area_factor: float,
            template_size: Tuple[int, int],
            norm_stats_dataset_name,
            interpolation_mode,
            interpolation_align_corners,
            device: torch.device,



            base_gamma: float = 0.98,
            volatility_window: int = 5,
            lambda_decay: float = 15.0
    ):
        super().__init__()

        self.update_interval = max(1, int(update_interval))
        self.update_threshold = float(update_threshold)
        self.queue_size = max(1, int(queue_size))

        self.template_area_factor = float(template_area_factor)
        self.template_size = np.array(template_size)
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners
        self.transforms = get_dataset_norm_stats_transform(norm_stats_dataset_name, inplace=True)
        self.device = device



        self.base_gamma = float(base_gamma)
        self.volatility_window = int(volatility_window)
        self.lambda_decay = float(lambda_decay)



        self.template_queues: Dict[Any, List[torch.Tensor]] = {}
        self.frame_counters: Dict[Any, int] = {}



        self.last_bboxes: Dict[Any, np.ndarray] = {}
        self.recent_shifts: Dict[Any, deque] = {}

    def start(self, max_batch_size: int, template_shape: Tuple[int, int, int]):
        self.template_queues = {}
        self.frame_counters = {}
        self.last_bboxes = {}
        self.recent_shifts = {}

    def stop(self):
        self.template_queues.clear()
        self.frame_counters.clear()
        self.last_bboxes.clear()
        self.recent_shifts.clear()

    def initialize(self, task_id, template: torch.Tensor, initial_bbox=None):
        self.template_queues[task_id] = [(template.clone(), 1.0) for _ in range(self.queue_size)]
        self.frame_counters[task_id] = 1



        self.recent_shifts[task_id] = deque(maxlen=self.volatility_window)
        if initial_bbox is not None:
            self.last_bboxes[task_id] = initial_bbox.copy()

    def delete(self, task_id):
        self.template_queues.pop(task_id, None)
        self.frame_counters.pop(task_id, None)
        self.last_bboxes.pop(task_id, None)
        self.recent_shifts.pop(task_id, None)

    def get(self, task_id) -> torch.Tensor:
        queue = self.template_queues[task_id]
        shifts = list(self.recent_shifts.get(task_id, []))



        if len(shifts) >= 2:
            volatility = float(np.var(shifts))
        else:
            volatility = 0.0



        dynamic_gamma = self.base_gamma * math.exp(-self.lambda_decay * volatility)

        best_score = -1.0
        best_template = None

        for i, (template_tensor, conf) in enumerate(queue):
            age = (self.queue_size - 1) - i


            decayed_score = conf * (dynamic_gamma ** age)

            if decayed_score > best_score:
                best_score = decayed_score
                best_template = template_tensor

        return best_template

    def get_batch(self, task_ids: list) -> torch.Tensor:
        batch_templates = [self.get(tid) for tid in task_ids]
        return torch.stack(batch_templates, dim=0)

    def update(self, task_id, confidence: float, x: torch.Tensor, bbox: np.ndarray, extra_info: dict = None):
        self.frame_counters[task_id] += 1
        current_frame = self.frame_counters[task_id]



        curr_cx, curr_cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        if task_id in self.last_bboxes:
            last_b = self.last_bboxes[task_id]
            last_cx, last_cy = last_b[0] + last_b[2] / 2, last_b[1] + last_b[3] / 2



            shift = math.hypot(curr_cx - last_cx, curr_cy - last_cy)
            target_sz = math.sqrt(bbox[2] * bbox[3])
            normalized_shift = shift / (target_sz + 1e-6)

            self.recent_shifts[task_id].append(normalized_shift)



        self.last_bboxes[task_id] = bbox.copy()



        if current_frame % self.update_interval == 0:
            if float(confidence) >= self.update_threshold:
                new_template = self._crop_template(x, bbox)
                queue = self.template_queues[task_id]
                queue.pop(0)
                queue.append((new_template, float(confidence)))

        # print("vatd")

    def _crop_template(self, x: torch.Tensor, bbox) -> torch.Tensor:
        template_curation_parameter = get_siamfc_cropping_params(
            bbox, self.template_area_factor, self.template_size
        )
        d, _, _ = apply_siamfc_cropping(
            x.to(torch.float32),
            self.template_size,
            template_curation_parameter,
            self.interpolation_mode,
            self.interpolation_align_corners,
        )
        return self.transforms(d.div_(255.0))