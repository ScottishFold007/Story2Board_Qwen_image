from typing import TYPE_CHECKING

from diffusers.utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_additional_imports = {}
_import_structure = {"pipeline_output": ["QwenImagePipelineOutput", "QwenImagePriorReduxPipelineOutput"]}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from diffusers.utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["modeling_qwenimage"] = ["ReduxImageEncoder"]
    _import_structure["pipeline_qwenimage"] = ["QwenImagePipeline"]
    _import_structure["pipeline_qwenimage_story2board"] = ["Story2BoardQwenImagePipeline"]
    _import_structure["story2board_transformer"] = ["Story2BoardQwenTransformer2DModel"]
    _import_structure["story2board_attention_processor"] = ["Story2BoardAttentionProcessor"]
    _import_structure["attention_store"] = ["AttentionStore"]
    _import_structure["pipeline_qwenimage_controlnet"] = ["QwenImageControlNetPipeline"]
    _import_structure["pipeline_qwenimage_controlnet_inpaint"] = ["QwenImageControlNetInpaintPipeline"]
    _import_structure["pipeline_qwenimage_edit"] = ["QwenImageEditPipeline"]
    _import_structure["pipeline_qwenimage_edit_inpaint"] = ["QwenImageEditInpaintPipeline"]
    _import_structure["pipeline_qwenimage_edit_plus"] = ["QwenImageEditPlusPipeline"]
    _import_structure["pipeline_qwenimage_img2img"] = ["QwenImageImg2ImgPipeline"]
    _import_structure["pipeline_qwenimage_inpaint"] = ["QwenImageInpaintPipeline"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from diffusers.utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_story2board import Story2BoardQwenImagePipeline
        from diffusers.pipelines.qwenimage.story2board_transformer import Story2BoardQwenTransformer2DModel
        from diffusers.pipelines.qwenimage.story2board_attention_processor import Story2BoardAttentionProcessor
        from diffusers.pipelines.qwenimage.attention_store import AttentionStore
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_controlnet import QwenImageControlNetPipeline
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_controlnet_inpaint import (
            QwenImageControlNetInpaintPipeline,
        )
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_inpaint import QwenImageEditInpaintPipeline
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_img2img import QwenImageImg2ImgPipeline
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_inpaint import QwenImageInpaintPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
