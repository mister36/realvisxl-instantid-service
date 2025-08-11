# RealVisXL InstantID Service

A standalone FastAPI service for generating consistent images using InstantID with RealVisXL V5.0.

## Features

-   InstantID face-consistent image generation
-   RealVisXL V5.0 base model for high-quality outputs
-   Memory optimization options (CPU offload)
-   RESTful API with FastAPI

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the service:

```bash
python app.py
```

The service will start on `http://localhost:8000`

## API Endpoints

### POST /instantid

Generate images with face consistency using InstantID.

**Parameters:**

-   `face_image`: Input image containing the face to preserve (file upload)
-   `prompt`: Text prompt for image generation
-   `negative_prompt`: Negative prompt (optional)
-   `width`: Output width (default: 1024)
-   `height`: Output height (default: 1024)
-   `num_inference_steps`: Number of denoising steps (default: 30)
-   `guidance_scale`: Guidance scale (default: 7.5)
-   `ip_adapter_scale`: IP adapter influence scale (default: 0.8)
-   `controlnet_conditioning_scale`: ControlNet conditioning scale (default: 0.8)
-   `seed`: Random seed for reproducibility (optional)

### GET /health

Health check endpoint.

## Environment Variables

-   `INSTANTID_BASE_MODEL`: Base model to use (default: "SG161222/RealVisXL_V5.0")
-   `INSTANTID_ENABLE_CPU_OFFLOAD`: Enable CPU offload for memory optimization (default: "true")
-   `INSTANTID_ENABLE_SEQUENTIAL_CPU_OFFLOAD`: Enable sequential CPU offload for maximum memory savings (default: "false")

## Memory Optimization

The service includes several memory optimization options:

1. **CPU Offload**: Moves inactive model components to CPU to save GPU memory
2. **Sequential CPU Offload**: Most aggressive memory saving - only keeps active components on GPU
3. **VAE Tiling**: Reduces memory usage during image encoding/decoding
4. **Attention Slicing**: Reduces memory usage during attention computation

Configure these via environment variables as needed for your hardware setup.
