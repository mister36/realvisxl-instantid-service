import io, os
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
from PIL import Image
from instantid_service import InstantIDService

app = FastAPI(title="RealVisXL InstantID Service", description="Image generation service using InstantID with RealVisXL V5.0")
instantid_svc = None

@app.on_event("startup")
def _load():
    global instantid_svc
    
    # Load InstantID service
    instantid_base = os.getenv("INSTANTID_BASE_MODEL", "SG161222/RealVisXL_V5.0")
    print(f"Loading InstantID service with base model: {instantid_base}")
    
    instantid_svc = InstantIDService(
        base_model=instantid_base
    )

@app.post("/instantid")
async def instantid(
    face_image: UploadFile,
    prompt: str = Form(...),
    negative_prompt: str = Form("(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"),
    width: int = Form(1024),
    height: int = Form(1024),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(7.5),
    ip_adapter_scale: float = Form(0.8),
    controlnet_conditioning_scale: float = Form(0.8),
    seed: int = Form(None),
):
    """
    Generate consistent images using InstantID with RealVisXL V5.0
    
    Args:
        face_image: Input image containing the face to preserve
        prompt: Text prompt for image generation
        negative_prompt: Negative prompt to avoid unwanted elements
        width: Output image width (default: 1024)
        height: Output image height (default: 1024)
        num_inference_steps: Number of denoising steps (default: 30)
        guidance_scale: Guidance scale for generation (default: 7.5)
        ip_adapter_scale: Scale for IP adapter influence (default: 0.8)
        controlnet_conditioning_scale: Scale for ControlNet conditioning (default: 0.8)
        seed: Random seed for reproducibility (optional)
    """
    # Read and process the face image
    data = await face_image.read()
    face_img = Image.open(io.BytesIO(data)).convert("RGB")
    
    # Generate the image using InstantID
    image_path = instantid_svc.generate_to_file(
        face_image=face_img,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        ip_adapter_scale=ip_adapter_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        seed=seed,
    )
    
    # Stream the generated image
    def _stream():
        with open(image_path, "rb") as f:
            yield from f
        os.remove(image_path)
    
    return StreamingResponse(_stream(), media_type="image/png")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RealVisXL InstantID"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
