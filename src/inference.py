"""Z-Image PyTorch Native Inference for RunPod Serverless."""

import base64
import io
import os
import runpod
import time
import warnings
from typing import Dict, Any, Optional

import torch

warnings.filterwarnings("ignore")
from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate

# Global variables for model components (loaded once)
_components = None
_device = None
_dtype = torch.bfloat16


def initialize_model(
    model_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    compile: bool = False,
    attn_backend: Optional[str] = None,
) -> None:
    """Initialize the model components globally (called once on cold start)."""
    global _components, _device, _dtype
    
    if _components is not None:
        print("Model already loaded, skipping initialization")
        return
    
    if model_path is None:
        model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)
    
    if attn_backend is None:
        attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
    
    # Device selection priority: cuda -> tpu -> mps -> cpu
    if torch.cuda.is_available():
        _device = "cuda"
        print("Chosen device: cuda")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            _device = xm.xla_device()
            print("Chosen device: tpu")
        except (ImportError, RuntimeError):
            if torch.backends.mps.is_available():
                _device = "mps"
                print("Chosen device: mps")
            else:
                _device = "cpu"
                print("Chosen device: cpu")
    
    print(f"Loading model from: {model_path}")
    _components = load_from_local_dir(model_path, device=_device, dtype=dtype, compile=compile)
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")
    _dtype = dtype
    print("Model initialization complete")


def image_to_base64(image) -> str:
    """Convert PIL Image to base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function.
    
    Expected event structure:
    {
        "input": {
            "prompt": str,  # Required: text prompt for image generation
            "height": int,  # Optional: image height (default: 1024)
            "width": int,   # Optional: image width (default: 1024)
            "num_inference_steps": int,  # Optional: number of steps (default: 8)
            "guidance_scale": float,  # Optional: guidance scale (default: 0.0)
            "seed": int,  # Optional: random seed (default: None, random)
        }
    }
    
    Returns:
    {
        "image": str,  # base64 encoded image if return_base64=True
        "output_path": str,  # path to saved image if output_path provided
        "generation_time": float,  # time taken in seconds
        "status": str,  # "success" or "error"
        "error": str,  # error message if status is "error"
    }
    """
    global _components, _device
    
    try:
        # Initialize model if not already loaded
        if _components is None:
            initialize_model()
        
        # Extract input parameters
        input_data = event.get("input", {})
        prompt = input_data.get("prompt")
        
        if not prompt:
            return {
                "status": "error",
                "error": "Missing required parameter: prompt"
            }
        
        height = input_data.get("height", 1024)
        width = input_data.get("width", 1024)
        num_inference_steps = input_data.get("num_inference_steps", 9)
        guidance_scale = input_data.get("guidance_scale", 0.0)
        seed = input_data.get("seed")
        return_base64 = input_data.get("return_base64", True)
        output_path = input_data.get("output_path")
        
        # Create generator with seed if provided
        if seed is not None:
            generator = torch.Generator(_device).manual_seed(seed)
        else:
            generator = torch.Generator(_device).manual_seed(torch.randint(0, 2**32, (1,)).item())
        
        # Generate image
        start_time = time.time()
        images = generate(
            prompt=prompt,
            **_components,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        end_time = time.time()
        generation_time = end_time - start_time
        
        image = images[0]
        result = {
            "status": "success",
            "generation_time": round(generation_time, 2),
        }
        
        # Return base64 encoded image
        if return_base64:
            result["image"] = image_to_base64(image)
        
        # Save image to file if output_path provided
        if output_path:
            image.save(output_path)
            result["output_path"] = output_path
        
        print(f"Generation completed in {generation_time:.2f} seconds")
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error during generation: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": error_msg
        }


if __name__ == '__main__':
    runpod.serverless.start({"handler": handler})
