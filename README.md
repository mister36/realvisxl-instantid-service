# RealVisXL InstantID Service

A standalone FastAPI service for generating consistent images using InstantID with RealVisXL V5.0.

## Features

-   InstantID face-consistent image generation
-   RealVisXL V5.0 base model for high-quality outputs
-   Optimized for full GPU utilization for maximum performance
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

## Performance Optimizations

The service includes several optimizations for efficient GPU utilization:

1. **Full GPU Utilization**: All model components remain on GPU for maximum performance
2. **VAE Tiling**: Reduces memory usage during image encoding/decoding
3. **Attention Slicing**: Reduces memory usage during attention computation

The service is optimized for high-performance GPU setups with sufficient VRAM.
