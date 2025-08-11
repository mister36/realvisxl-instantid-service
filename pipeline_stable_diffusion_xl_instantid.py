import cv2
import torch
import math
import numpy as np
from PIL import Image
from typing import Optional, Union, List, Callable, Dict, Any

from diffusers import StableDiffusionXLPipeline
from diffusers.models import ControlNetModel
from diffusers.utils import PIL_INTERPOLATION


def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    """
    Draw keypoints on the image
    """
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


class StableDiffusionXLInstantIDPipeline(StableDiffusionXLPipeline):
    """
    Pipeline for Stable Diffusion XL with InstantID
    Based on the official InstantID implementation
    """
    
    def __init__(
        self, 
        vae,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        unet,
        controlnet,
        scheduler,
        feature_extractor=None,
        image_encoder=None,
        force_zeros_for_empty_prompt=True,
        requires_safety_checker=False,
    ):
        """
        Initialize the InstantID pipeline using explicit components
        """
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            requires_safety_checker=requires_safety_checker,
        )
        
        # Register controlnet
        self.register_modules(controlnet=controlnet)
        self.ip_adapter_scale = 1.0

    def load_ip_adapter_instantid(self, model_path: str):
        """Load the InstantID IP adapter"""
        import os
        # Load IP adapter for image prompt embeddings
        if os.path.isfile(model_path):
            # If model_path is a file, use the directory and filename
            directory = os.path.dirname(model_path)
            filename = os.path.basename(model_path)
            self.load_ip_adapter(directory, subfolder="", weight_name=filename)
        else:
            # If model_path is a directory, assume the standard filename
            self.load_ip_adapter(model_path, subfolder="", weight_name="ip-adapter.bin")
        
    def set_ip_adapter_scale(self, scale: float):
        """Set the IP adapter scale"""
        self.ip_adapter_scale = scale
        
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_mask: Optional[Union[torch.FloatTensor, Image.Image]] = None,
        image: Optional[Union[torch.FloatTensor, Image.Image]] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        """
        Generate images using InstantID
        """
        # Set default dimensions
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # Prepare the image for ControlNet (face keypoints)
        control_image = None
        if image is not None:
            if isinstance(image, Image.Image):
                control_image = image.resize((width, height), PIL_INTERPOLATION["lanczos"])
            else:
                control_image = image
        
        # Prepare cross attention kwargs for IP adapter
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        
        # Set IP adapter scale in cross attention kwargs
        cross_attention_kwargs["ip_adapter_scale"] = self.ip_adapter_scale
        
        # Use the parent pipeline with InstantID features
        result = super().__call__(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            cross_attention_kwargs=cross_attention_kwargs,
            # ControlNet parameters
            image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            # IP adapter parameters
            ip_adapter_image_embeds=[image_embeds] if image_embeds is not None else None,
            **kwargs,
        )
        
        return result
