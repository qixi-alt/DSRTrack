from trackit.models import ModelBuildingContext, ModelImplSuggestions
from trackit.models.backbone.builder import build_backbone
from trackit.miscellanies.pretty_format import pretty_format

from .sample_data_generator import build_sample_input_data_generator


def get_DSRTrack_build_context(config: dict):
    print("DSRTrack model config:\n" + pretty_format(config["model"]))
    return ModelBuildingContext(
        lambda impl_advice: build_DSRTrack_model(config, impl_advice),
        lambda impl_advice: get_DSRTrack_build_string(
            config["model"]["type"], impl_advice
        ),
        build_sample_input_data_generator(config),
    )


def build_DSRTrack_model(
    config: dict,
    model_impl_suggestions: ModelImplSuggestions,
):
    """Build the inference-only DSRTrack model."""
    model_config = config["model"]
    common_config = config["common"]
    model_type = model_config["type"]

    if model_type != "dinov2":
        raise NotImplementedError(
            f"Model type '{model_type}' is not included in DSRTrack."
        )

    if not model_impl_suggestions.optimize_for_inference:
        raise RuntimeError(
            "This public release provides DSRTrack inference and evaluation "
            "code only. Method-specific training code is not included."
        )

    backbone = build_backbone(
        model_config["backbone"],
        torch_jit_trace_compatible=(
            model_impl_suggestions.torch_jit_trace_compatible
        ),
    )

    from .dsrtrack_full_finetune import DSRTrackBaseline_DINOv2

    model = DSRTrackBaseline_DINOv2(
        backbone,
        common_config["template_feat_size"],
        common_config["search_region_feat_size"],
    )

    if config["model"]["eval"]:
        for path in config["model"]["weight_path"]:
            model.load_state_dict_from_file(path)

    return model


def get_DSRTrack_build_string(
    model_type: str,
    model_impl_suggestions: ModelImplSuggestions,
):
    if model_type != "dinov2":
        raise NotImplementedError(
            f"Model type '{model_type}' is not included in DSRTrack."
        )

    build_string = "DSRTrack"
    if model_impl_suggestions.optimize_for_inference:
        build_string += "_merged"
    if model_impl_suggestions.torch_jit_trace_compatible:
        build_string += "_disable_flash_attn"
    return build_string
