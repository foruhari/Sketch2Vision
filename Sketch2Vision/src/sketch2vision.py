"""Sketch-conditioned image generation using frozen Stable Diffusion + ControlNet weights."""
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"

def prepare_sketch(image: Image.Image, size: int = 512, low: int = 45, high: int = 125):
    """Resize a photographed pencil sketch and derive a clean Canny condition."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size-image.width)//2, (size-image.height)//2))
    gray = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2GRAY)
    # Suppress paper grain and mild illumination variation while retaining pencil strokes.
    smooth = cv2.bilateralFilter(gray, 9, 55, 55)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(smooth)
    edges = cv2.Canny(enhanced, low, high)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    control = Image.fromarray(edges).convert("RGB")
    return canvas, control

class Sketch2Vision:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL, torch_dtype=dtype, use_safetensors=True
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL, controlnet=controlnet, torch_dtype=dtype,
            use_safetensors=True
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
        else:
            self.pipe.to(self.device)

    def generate(self, sketch: Image.Image, prompt: str, negative_prompt: str = "",
                 control_scale: float = 0.75, steps: int = 30, guidance: float = 7.5,
                 seed: int = 123, size: int = 512):
        _, control = prepare_sketch(sketch, size=size)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        result = self.pipe(
            prompt=prompt, negative_prompt=negative_prompt or None,
            image=control, num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            controlnet_conditioning_scale=float(control_scale),
            generator=generator, height=size, width=size
        ).images[0]
        return result, control
