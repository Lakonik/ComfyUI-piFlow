import comfy
from .modules.model_base import ModelSamplingPiFlow as _ModelSamplingPiFlow


class ModelSamplingPiFlow:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "shift": ("FLOAT", {"default": 3.2, "min": 0.0, "max": 100.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "piflow"

    def patch(self, model, shift):
        m = model.clone()
        multiplier = model.model.model_config.sampling_settings.get("multiplier", 1.0)

        sampling_base = _ModelSamplingPiFlow
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift, multiplier=multiplier)
        m.add_object_patch("model_sampling", model_sampling)
        return (m,)
