import comfy
import latent_preview
from .modules.sampler import sample


def piflow_sampler(
        model, seed, steps, substeps, final_step_size_scale,
        diffusion_coefficient, gm_temperature, manual_gm_temperature,
        conditioning, latent, denoise=1.0):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = sample(
        model, noise, steps, substeps, final_step_size_scale, diffusion_coefficient,
        gm_temperature, manual_gm_temperature,
        conditioning, latent_image, denoise=denoise,
        noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )


class PiFlowSampler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000, "tooltip": "The number of network steps used in the denoising process."}),
                "substeps": ("INT", {"default": 128, "min": 1, "max": 10000, "tooltip": "The number of policy sub-steps used in the denoising process."}),
                "final_step_size_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The size of the final step relative to other steps."}),
                "diffusion_coefficient": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000000.0, "step": 0.01, "tooltip": "The coefficient controlling the stochasticity of the sampling process. 0.0 is deterministic. 1.0 is standard DDPM stochasticity."}),
                "gm_temperature": (['auto', 'manual'], {"default": 'auto', "tooltip": "The GMFlow temperature setting."}),
                "manual_gm_temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The GMFlow temperature to use if gm_temperature is set to manual."}),
                "conditioning": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "piflow"
    DESCRIPTION = "Uses the provided model, positive conditioning to denoise the latent image."

    def sample(self, model, seed, steps, substeps, final_step_size_scale,
               diffusion_coefficient, gm_temperature, manual_gm_temperature,
               conditioning, latent_image, denoise=1.0):
        return piflow_sampler(
            model, seed, steps, substeps, final_step_size_scale,
            diffusion_coefficient, gm_temperature, manual_gm_temperature,
            conditioning, latent_image, denoise)
