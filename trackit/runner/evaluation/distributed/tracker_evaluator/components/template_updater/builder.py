import torch

from trackit.miscellanies.pretty_format import pretty_format
from . import TemplateUpdater


def build_template_updater(
    template_update_config: dict,
    common_config: dict,
    device: torch.device,
) -> TemplateUpdater:
    """Build the online-template updater used by DSRTrack."""
    print("DSRTrack template update:\n" + pretty_format(template_update_config))

    template_update_type = template_update_config["type"]
    if template_update_type != "vatd_queue":
        raise NotImplementedError(
            f"Template updater '{template_update_type}' is not included in DSRTrack."
        )

    from .vatd_queue_updater import VATDQueueUpdater

    return VATDQueueUpdater(
        update_interval=template_update_config.get("update_interval", 5),
        update_threshold=template_update_config.get("update_threshold", 0.78),
        queue_size=template_update_config.get("queue_size", 2),
        base_gamma=template_update_config.get("base_gamma", 0.98),
        volatility_window=template_update_config.get("volatility_window", 5),
        lambda_decay=template_update_config.get("lambda_decay", 15.0),
        template_area_factor=template_update_config["template_area_factor"],
        template_size=common_config["template_size"],
        norm_stats_dataset_name=common_config["normalization"],
        interpolation_mode=common_config["interpolation_mode"],
        interpolation_align_corners=common_config[
            "interpolation_align_corners"
        ],
        device=device,
    )
