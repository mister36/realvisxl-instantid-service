import os
import cv2
import torch
import numpy as np
import tempfile
import gc
from typing import Optional
from PIL import Image
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
from diffusers.models import ControlNetModel
from pipeline_stable_diffusion_xl_instantid import draw_kps
import diffusers


class InstantIDService:
    def __init__(
        self,
        base_model: str = "wangqixun/YamerMIX_v8",  # Using YamerMIX v8 as requested
        device: str = "cuda",
        dtype=torch.float16,
        checkpoints_dir: str = "./checkpoints",
        models_dir: str = "./models",
    ):
        # Check CUDA compatibility
        if device == "cuda" and torch.cuda.is_available():
            try:
                # Test basic CUDA operations
                test_tensor = torch.tensor([1.0], device="cuda")
                test_result = test_tensor * 2
                print("CUDA compatibility test passed")
                self.device = device
                self.dtype = dtype
            except RuntimeError as e:
                if "no kernel image is available" in str(e):
                    print("WARNING: CUDA kernel compatibility issue detected (RTX 5090 needs newer PyTorch)")
                    print("Falling back to CPU for tensor operations")
                    self.device = "cpu"
                    self.dtype = torch.float32
                else:
                    print(f"CUDA error: {e}")
                    self.device = device
                    self.dtype = dtype
        else:
            self.device = device
            self.dtype = dtype
        
        self.checkpoints_dir = checkpoints_dir
        self.models_dir = models_dir
        
        print(f"Loading InstantID models with device: {self.device}, dtype: {self.dtype}")
        
        # Ensure directories exist
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Download InstantID models if not present
        self._download_models()
        
        # Initialize face analysis - root should point to directory containing models/antelopev2
        self.app = FaceAnalysis(
            name='antelopev2', 
            root='./', 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load ControlNet
        controlnet_path = os.path.join(checkpoints_dir, "ControlNetModel")
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
        
        # Load pipeline using YamerMIX v8 (following the pattern from your example)
        from diffusers import StableDiffusionXLPipeline
        from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
        
        # Load the YamerMIX v8 model and create InstantID pipeline
        if base_model.endswith(".ckpt") or base_model.endswith(".safetensors"):
            # Handle local checkpoint files
            scheduler_kwargs = hf_hub_download(
                repo_id="wangqixun/YamerMIX_v8",
                subfolder="scheduler",
                filename="scheduler_config.json",
            )
            
            # This would require model_util which might not be available
            # For now, we'll assume we're loading from HuggingFace Hub
            raise NotImplementedError("Local checkpoint loading not implemented yet")
        else:
            # Load from HuggingFace Hub using the exact pattern from your example
            self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                base_model,
                controlnet=self.controlnet,
                torch_dtype=dtype,
                safety_checker=None,
                feature_extractor=None,
            )
            
            # Set EulerDiscreteScheduler as used in YamerMIX examples
            self.pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        
        # Keep everything on GPU for maximum performance
        self.pipe.to(device)
        
        # Load IP adapter for InstantID AFTER moving to device
        face_adapter_path = os.path.join(checkpoints_dir, "ip-adapter.bin")
        self.ip_adapter_loaded = False
        
        try:
            # Try different methods to load the IP adapter
            if os.path.exists(face_adapter_path):
                # Method 1: Try the InstantID specific loading method
                try:
                    self.pipe.load_ip_adapter_instantid(face_adapter_path)
                    self.ip_adapter_loaded = True
                    print("InstantID IP adapter loaded successfully with load_ip_adapter_instantid")
                except Exception as e1:
                    print(f"load_ip_adapter_instantid failed: {e1}")
                    # Method 2: Try standard IP adapter loading
                    try:
                        # Some versions use standard IP adapter loading
                        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
                        self.ip_adapter_loaded = True
                        print("IP adapter loaded successfully with standard method")
                    except Exception as e2:
                        print(f"Standard IP adapter loading also failed: {e2}")
                        self.ip_adapter_loaded = False
            else:
                print(f"IP adapter file not found at {face_adapter_path}")
                self.ip_adapter_loaded = False
        except Exception as e:
            print(f"Note: Could not load IP adapter: {e}")
            self.ip_adapter_loaded = False
            
        # Load and disable LCM LoRA weights (as shown in YamerMIX example)
        try:
            self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
            self.pipe.disable_lora()  # Disabled by default
            print("LCM LoRA weights loaded and disabled")
        except Exception as e:
            print(f"Could not load LCM LoRA weights: {e}")
            
        # Enable VAE tiling for lower memory usage, but DO NOT use attention slicing
        # as it's incompatible with IP adapters (causes SlicedAttnProcessor warning)
        if hasattr(self.pipe, 'vae'):
            self.pipe.vae.enable_tiling()
        
        # For InstantID with face embeddings, we need to keep models on GPU for proper functionality
        # Use memory efficient attention if available, but avoid CPU offloading
        try:
            # Try to use memory efficient attention backends if available
            if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                self.pipe.enable_xformers_memory_efficient_attention()
                print("Enabled xFormers memory efficient attention")
            elif hasattr(self.pipe.unet, 'set_attn_processor'):
                # Use AttnProcessor2_0 for memory efficiency while maintaining IP adapter compatibility
                from diffusers.models.attention_processor import AttnProcessor2_0
                self.pipe.unet.set_attn_processor(AttnProcessor2_0())
                print("Set AttnProcessor2_0 for memory efficiency")
        except Exception as e:
            print(f"Could not enable memory efficient attention: {e}")
            print("Proceeding with default attention processors")
        
        print("InstantID pipeline initialized successfully with YamerMIX v8")
        
    def clear_memory(self):
        """Clear GPU memory cache and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Cleared GPU memory cache")
        
    def _download_models(self):
        """Download InstantID models from HuggingFace Hub"""
        try:
            # Download ControlNet model
            hf_hub_download(
                repo_id="InstantX/InstantID", 
                filename="ControlNetModel/config.json", 
                local_dir=self.checkpoints_dir
            )
            hf_hub_download(
                repo_id="InstantX/InstantID", 
                filename="ControlNetModel/diffusion_pytorch_model.safetensors", 
                local_dir=self.checkpoints_dir
            )
            
            # Download IP adapter
            hf_hub_download(
                repo_id="InstantX/InstantID", 
                filename="ip-adapter.bin", 
                local_dir=self.checkpoints_dir
            )
            
            # Download InsightFace AntelopeV2 models from official source
            self._download_antelopev2_models()
            
            print("InstantID models downloaded successfully")
            
        except Exception as e:
            print(f"Error downloading models: {e}")
            raise

    def _download_antelopev2_models(self):
        """Download AntelopeV2 models from official InsightFace releases"""
        import urllib.request
        import zipfile
        
        # Place models in ./models/antelopev2 as expected by InsightFace with root='./'
        models_base_dir = "./models"
        antelopev2_dir = os.path.join(models_base_dir, "antelopev2")
        
        # Check if models already exist
        if os.path.exists(antelopev2_dir) and len(os.listdir(antelopev2_dir)) >= 5:
            print("AntelopeV2 models already exist, skipping download")
            return
            
        os.makedirs(models_base_dir, exist_ok=True)
        
        # Download from official InsightFace releases
        antelopev2_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
        zip_path = os.path.join(models_base_dir, "antelopev2.zip")
        
        print("Downloading AntelopeV2 models from official InsightFace releases...")
        urllib.request.urlretrieve(antelopev2_url, zip_path)
        
        print("Extracting AntelopeV2 models...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_base_dir)
        
        # Clean up zip file
        os.remove(zip_path)
        
        print("AntelopeV2 models downloaded and extracted successfully")
    
    def generate_consistent_image(
        self,
        face_image: Image.Image,
        prompt: str,
        negative_prompt: str = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        ip_adapter_scale: float = 0.8,
        controlnet_conditioning_scale: float = 0.8,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate a consistent image given a face image and prompt
        
        Args:
            face_image: PIL Image containing the face to preserve
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt for image generation
            width: Output image width
            height: Output image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            ip_adapter_scale: Scale for IP adapter influence
            controlnet_conditioning_scale: Scale for ControlNet conditioning
            seed: Random seed for reproducibility
            
        Returns:
            Generated PIL Image
        """
        # Prepare face embedding
        face_info = self.app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        
        if not face_info:
            raise ValueError("No face detected in the input image")
        
        # Use the largest face
        face_info = sorted(
            face_info, 
            key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])
        )[-1]
        
        face_emb = face_info['embedding']
        face_kps = draw_kps(face_image, face_info['kps'])
        
        # Convert face embedding to proper tensor format with CUDA error handling
        # InsightFace returns 1D embedding [512], InstantID expects [1, 512]
        try:
            face_emb_tensor = torch.tensor(face_emb, dtype=self.dtype, device=self.device)
            if face_emb_tensor.dim() == 1:
                face_emb_tensor = face_emb_tensor.unsqueeze(0)  # Shape: [1, 512]
            
            # For IP adapter (if loaded), create combined embeddings
            # Reshape to 4D: [1, embedding_dim, 1, 1] for IP adapter
            ip_face_emb_tensor = face_emb_tensor.unsqueeze(-1).unsqueeze(-1)  # Shape: [1, 512, 1, 1]
            
            # Create negative embedding (zeros) for IP adapter
            # The IP adapter expects concatenated [negative, positive] embeddings
            negative_emb_tensor = torch.zeros_like(ip_face_emb_tensor)
            
            # Concatenate negative and positive embeddings along batch dimension
            # Shape: [2, 512, 1, 1] where first is negative, second is positive
            combined_emb_tensor = torch.cat([negative_emb_tensor, ip_face_emb_tensor], dim=0)
            
        except RuntimeError as e:
            if "CUDA error" in str(e):
                print(f"CUDA error detected, falling back to CPU for face embeddings: {e}")
                # Fallback to CPU for tensor operations
                face_emb_tensor = torch.tensor(face_emb, dtype=torch.float32, device="cpu")
                if face_emb_tensor.dim() == 1:
                    face_emb_tensor = face_emb_tensor.unsqueeze(0)
                
                ip_face_emb_tensor = face_emb_tensor.unsqueeze(-1).unsqueeze(-1)
                negative_emb_tensor = torch.zeros_like(ip_face_emb_tensor)
                combined_emb_tensor = torch.cat([negative_emb_tensor, ip_face_emb_tensor], dim=0)
                
                # Move to GPU if possible, otherwise keep on CPU
                try:
                    face_emb_tensor = face_emb_tensor.to(self.device)
                    combined_emb_tensor = combined_emb_tensor.to(self.device)
                except:
                    print("Keeping face embeddings on CPU due to CUDA issues")
            else:
                raise e
        
        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Prepare generation parameters
        generation_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": face_kps,  # Face keypoints for pose/structure control
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "generator": generator,
        }
        
        # For InstantID, face identity is preserved through IP adapter
        # Add IP adapter parameters only if IP adapter is loaded
        if self.ip_adapter_loaded and hasattr(self.pipe, 'set_ip_adapter_scale'):
            self.pipe.set_ip_adapter_scale(ip_adapter_scale)
            generation_kwargs["ip_adapter_image_embeds"] = [combined_emb_tensor]
            print("Generating with InstantID + IP adapter for face identity")
        else:
            # If IP adapter failed to load, we need to inject face embeddings differently
            # This is a fallback that may not work as well but provides some face guidance
            print("WARNING: IP adapter not loaded. Face identity preservation may be limited.")
            print("Generating with ControlNet-only InstantID (pose/structure only)")
            # Pass face embeddings as image_embeds (may help with some pipelines)
            generation_kwargs["image_embeds"] = face_emb_tensor
        
        # Generate image using InstantID with face embeddings and keypoints
        result = self.pipe(**generation_kwargs)
        
        # Clear memory after generation
        self.clear_memory()
        
        return result.images[0]
    
    def generate_to_file(
        self,
        face_image: Image.Image,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Generate image and save to temporary file
        
        Returns:
            Path to the generated image file
        """
        generated_image = self.generate_consistent_image(face_image, prompt, **kwargs)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            generated_image.save(tmp_file.name)
            return tmp_file.name
