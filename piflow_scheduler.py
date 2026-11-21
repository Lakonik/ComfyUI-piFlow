import torch


def get_piflow_schedule(model_sampling, num_steps: int, min_raw_t: float, max_raw_t: float,
                        final_step_size_scale: float) -> torch.Tensor:
    raw_t = torch.linspace(
        max_raw_t,
        (max_raw_t - min_raw_t) * final_step_size_scale / (num_steps - 1 + final_step_size_scale) + min_raw_t,
        num_steps)
    # the model's own time shift maps raw t to sigma
    sigmas = torch.tensor([model_sampling.percent_to_sigma(1.0 - t) for t in raw_t.tolist()])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


class PiFlowScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 4, "min": 1, "max": 4096}),
                "min_raw_t": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "max_raw_t": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "final_step_size_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "LakonLab"

    def get_sigmas(self, model, steps, min_raw_t, max_raw_t, final_step_size_scale):
        # MiniMax H3 derives the audio sigmas from these video sigmas with its audio shift
        model_sampling = model.get_model_object("model_sampling")
        return (get_piflow_schedule(model_sampling, steps, min_raw_t, max_raw_t, final_step_size_scale),)
