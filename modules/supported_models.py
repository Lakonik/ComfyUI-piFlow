import torch
import comfy
from comfy import latent_formats, supported_models_base
from . import model_base


class OklabPixels(latent_formats.LatentFormat):
    is_pixel_latent = True
    latent_channels = 3
    spacial_downscale_ratio = 1


class GMQwenImage(supported_models_base.BASE):
    unet_config = {
        "image_model": "gm_qwen_image",
    }

    policy_config = {
        "type": "GMFlow",
    }

    sampling_settings = {'multiplier': 1.0}

    memory_usage_factor = 1.8

    unet_extra_config = {}
    latent_format = latent_formats.Wan21

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.GMQwenImage(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.qwen_image.QwenImageTokenizer, comfy.text_encoders.qwen_image.te(**hunyuan_detect))


class QwenImage(supported_models_base.BASE):
    unet_config = {
        "image_model": "qwen_image",
    }

    policy_config = {}

    sampling_settings = {'multiplier': 1.0}

    memory_usage_factor = 1.8

    unet_extra_config = {}
    latent_format = latent_formats.Wan21

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.QwenImage(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.qwen_image.QwenImageTokenizer, comfy.text_encoders.qwen_image.te(**hunyuan_detect))


class GMFlux(supported_models_base.BASE):
    unet_config = {
        "image_model": "gm_flux",
        "guidance_embed": True,
    }

    policy_config = {
        "type": "GMFlow",
    }

    sampling_settings = {'multiplier': 1.0}

    unet_extra_config = {}
    latent_format = latent_formats.Flux

    memory_usage_factor = 3.1

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.GMFlux(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.flux.FluxTokenizer, comfy.text_encoders.flux.flux_clip(**t5_detect))


class Flux(supported_models_base.BASE):
    unet_config = {
        "image_model": "flux",
        "guidance_embed": True,
    }

    policy_config = {}

    sampling_settings = {'multiplier': 1.0}

    unet_extra_config = {}
    latent_format = latent_formats.Flux

    memory_usage_factor = 3.1

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Flux(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.flux.FluxTokenizer, comfy.text_encoders.flux.flux_clip(**t5_detect))


class GMFlux2(supported_models_base.BASE):
    unet_config = {
        "image_model": "gm_flux2",
    }

    policy_config = {
        "type": "GMFlow",
    }

    sampling_settings = {
        'multiplier': 1.0,
        'patch_size': (2, 2)
    }

    unet_extra_config = {}
    latent_format = latent_formats.Flux2

    memory_usage_factor = 3.1

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = self.memory_usage_factor * (2.0 * 2.0) * (unet_config["hidden_size"] / 2604)

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.GMFlux2(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return None


class Flux2(supported_models_base.BASE):
    unet_config = {
        "image_model": "flux2",
    }

    policy_config = {}

    sampling_settings = {
        'multiplier': 1.0,
        'patch_size': (2, 2)
    }

    unet_extra_config = {}
    latent_format = latent_formats.Flux2

    memory_usage_factor = 3.1

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = self.memory_usage_factor * (2.0 * 2.0) * (unet_config['hidden_size'] / 2604)

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Flux2(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return None


class AsymFlux2(supported_models_base.BASE):
    unet_config = {
        "image_model": "asym_flux2",
    }

    sampling_settings = {
        "shift": 17.0,
    }

    unet_extra_config = {}
    latent_format = OklabPixels

    memory_usage_factor = 3.1

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        hidden_size = unet_config["hidden_size"]
        patch_size = unet_config["patch_size"]
        self.memory_usage_factor = self.memory_usage_factor * (2.0 * 2.0) * (hidden_size / 2604) / ((patch_size // 2) ** 2)

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.AsymFlux2(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return None


models = [GMQwenImage, QwenImage, GMFlux, Flux, GMFlux2, Flux2, AsymFlux2]
