from .lakonlab_loaders import PiFlowLoader, PiFlowLoaderGGUF, AsymFlowLoader, AsymFlowLoaderGGUF, LakonLabAdapterLoader, DWDDecode
from .piflow_sampler import PiFlowSampler
from .model_sampling_piflow import ModelSamplingPiFlow
from .oklab_color_encoder import OklabColorEncoderNode
from .pixel_preview import PixelPreview, install_pixel_previewer
from .clamp_denoised import ClampDenoised
from .asymflux2_scheduler import AsymFlux2Scheduler
from .piflow_scheduler import PiFlowScheduler


install_pixel_previewer()


NODE_CLASS_MAPPINGS = {
    "Load pi-Flow Model": PiFlowLoader,
    "Load AsymFlow Model": AsymFlowLoader,
    "Load LakonLab Adapter": LakonLabAdapterLoader,
    "DWD Decode": DWDDecode,
    "Oklab Color Encoder": OklabColorEncoderNode,
    "Pixel Preview": PixelPreview,
    "Clamp Denoised": ClampDenoised,
    "AsymFlux2Scheduler": AsymFlux2Scheduler,
    "pi-Flow Scheduler": PiFlowScheduler,
    "pi-Flow Sampler": PiFlowSampler,
    "ModelSamplingPiFlow": ModelSamplingPiFlow,
}

if PiFlowLoaderGGUF is not None:
    NODE_CLASS_MAPPINGS["Load pi-Flow Model (GGUF)"] = PiFlowLoaderGGUF
if AsymFlowLoaderGGUF is not None:
    NODE_CLASS_MAPPINGS["Load AsymFlow Model (GGUF)"] = AsymFlowLoaderGGUF


__all__ = ['NODE_CLASS_MAPPINGS']
