# InstantID Usage Guide

This service now includes InstantID functionality for generating consistent face images using RealVisXL V5.0 as the base model.

## New Route: `/instantid`

### Description

Generate consistent images preserving facial identity from an input face image using InstantID with RealVisXL V5.0.

### Endpoint

`POST /instantid`

### Parameters

| Parameter                       | Type    | Default                 | Description                                            |
| ------------------------------- | ------- | ----------------------- | ------------------------------------------------------ |
| `face_image`                    | File    | Required                | Input image containing the face to preserve            |
| `prompt`                        | String  | Required                | Text prompt for image generation                       |
| `negative_prompt`               | String  | Default negative prompt | Negative prompt to avoid unwanted elements             |
| `width`                         | Integer | 1024                    | Output image width                                     |
| `height`                        | Integer | 1024                    | Output image height                                    |
| `num_inference_steps`           | Integer | 30                      | Number of denoising steps                              |
| `guidance_scale`                | Float   | 7.5                     | Guidance scale for generation                          |
| `ip_adapter_scale`              | Float   | 0.8                     | Scale for IP adapter influence (identity preservation) |
| `controlnet_conditioning_scale` | Float   | 0.8                     | Scale for ControlNet conditioning                      |
| `seed`                          | Integer | None                    | Random seed for reproducibility                        |

### Example Usage

#### cURL

```bash
curl -X POST "http://localhost:8000/instantid" \
  -F "face_image=@path/to/face_image.jpg" \
  -F "prompt=analog film photo of a person in a vintage setting, highly detailed, masterpiece" \
  -F "width=1024" \
  -F "height=1024" \
  -F "ip_adapter_scale=0.8" \
  --output generated_image.png
```

#### Python with requests

```python
import requests

url = "http://localhost:8000/instantid"
files = {"face_image": open("path/to/face_image.jpg", "rb")}
data = {
    "prompt": "analog film photo of a person in a vintage setting, highly detailed, masterpiece",
    "width": 1024,
    "height": 1024,
    "ip_adapter_scale": 0.8,
    "seed": 42
}

response = requests.post(url, files=files, data=data)
with open("generated_image.png", "wb") as f:
    f.write(response.content)
```

### Parameter Tuning Tips

1. **Identity Preservation**:

    - Increase `ip_adapter_scale` (0.8-1.2) for stronger identity preservation
    - Increase `controlnet_conditioning_scale` (0.8-1.2) for better facial structure

2. **Image Quality**:

    - If saturation is too high, decrease `ip_adapter_scale` first
    - If text control is weak, decrease `ip_adapter_scale`
    - Adjust `guidance_scale` (5.0-10.0) for generation quality

3. **Style Control**:
    - Use detailed prompts for better style control
    - Include art style keywords (e.g., "analog film", "digital art", "oil painting")

### Model Information

-   **Base Model**: RealVisXL V5.0 (SG161222/RealVisXL_V5.0)
-   **Face Analysis**: InsightFace AntelopeV2
-   **Identity Preservation**: InstantX/InstantID
-   **Output Format**: PNG image

### First Run Setup

On the first run, the service will automatically:

1. Download InstantID models from HuggingFace
2. Download the AntelopeV2 face analysis model
3. Set up the RealVisXL V5.0 pipeline

This may take several minutes depending on your internet connection.

### Requirements

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

The service requires:

-   CUDA-compatible GPU (recommended)
-   At least 8GB VRAM
-   Python 3.8+

### Error Handling

-   **No face detected**: Ensure the input image contains a clear, visible face
-   **CUDA out of memory**: Reduce image dimensions or use CPU mode
-   **Model download fails**: Check internet connection and disk space

### Performance Notes

-   First generation may be slower due to model loading
-   Subsequent generations are faster due to caching
-   Generation time: ~30-60 seconds depending on hardware and settings
