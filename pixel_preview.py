import latent_preview
from comfy.cli_args import LatentPreviewMethod


class PixelPreviewLatentFormat:
    def __init__(self, latent_format, codec):
        self.latent_format = latent_format
        self.pixel_preview_codec = codec
        self.is_pixel_latent = True

    def __getattr__(self, name):
        return getattr(self.latent_format, name)

    def process_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_out(self, latent):
        return self.latent_format.process_out(latent)


class PixelPreviewer(latent_preview.LatentPreviewer):
    def __init__(self, codec):
        self.codec = codec

    def decode_latent_to_preview(self, x0):
        image = self.codec.decode(x0[:1])[0]
        return latent_preview.preview_to_image(image, do_scale=False)


def attach_pixel_previewer(model, codec):
    model.add_object_patch(
        "latent_format",
        PixelPreviewLatentFormat(model.get_model_object("latent_format"), codec),
    )


class PixelPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "vae": ("VAE",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "LakonLab"

    def patch(self, model, vae):
        model = model.clone()
        attach_pixel_previewer(model, vae)
        return (model,)


def install_pixel_previewer():
    if getattr(latent_preview, "_lakonlab_pixel_previewer_installed", False):
        return

    original_get_previewer = latent_preview.get_previewer
    original_prepare_callback = latent_preview.prepare_callback

    def get_previewer(device, latent_format):
        previewer = original_get_previewer(device, latent_format)
        if previewer is not None:
            return previewer

        method = latent_preview.args.preview_method
        if method not in (LatentPreviewMethod.Auto, LatentPreviewMethod.Latent2RGB):
            return None

        codec = getattr(latent_format, "pixel_preview_codec", None)
        if getattr(latent_format, "is_pixel_latent", False) and codec is not None:
            return PixelPreviewer(codec)

        return None

    def prepare_callback(model, steps, x0_output_dict=None):
        try:
            latent_format = model.get_model_object("latent_format")
        except Exception:
            return original_prepare_callback(model, steps, x0_output_dict=x0_output_dict)

        if not isinstance(latent_format, PixelPreviewLatentFormat):
            return original_prepare_callback(model, steps, x0_output_dict=x0_output_dict)

        previewer = get_previewer(model.load_device, latent_format)
        progress = latent_preview.comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            if x0_output_dict is not None:
                x0_output_dict["x0"] = x0

            preview_bytes = None
            if previewer:
                if getattr(x0, "is_nested", False):
                    x0 = x0.tensors[0]
                preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
            progress.update_absolute(step + 1, total_steps, preview_bytes)

        return callback

    latent_preview.get_previewer = get_previewer
    latent_preview.prepare_callback = prepare_callback
    latent_preview._lakonlab_pixel_previewer_installed = True
