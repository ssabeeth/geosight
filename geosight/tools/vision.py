"""
Vision tool — describes a land/site photograph.
Uses Groq when GROQ_API_KEY is set, otherwise LLaVA via a local Ollama
(which must be running with llava:7b pulled).
"""

import base64
import os
import re
from pathlib import Path

import requests
from pydantic import BaseModel

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

VISION_PROMPT = """You are a land and environmental analyst examining a site photograph.
Describe what you observe in structured terms relevant to land use and planning:

1. **Terrain & Topography**: flat, sloped, valley, hillside, coastal, etc.
2. **Vegetation & Ecology**: woodland, grassland, crops, hedgerows, scrub, wetland, etc.
3. **Land Use Indicators**: agricultural equipment, buildings, infrastructure, tracks, fencing
4. **Water Features**: streams, ditches, standing water, flooding signs
5. **Condition**: well-managed, overgrown, degraded, signs of recent activity
6. **Estimated Scale & Setting**: approximate area visible, rural/peri-urban/urban

Describe only what is visible in the photograph. If something cannot be judged
from the image, say so rather than guessing.
Be concise but specific. This analysis will be combined with geospatial data."""


class VisionResult(BaseModel):
    description: str
    model_used: str
    image_provided: bool = True


def describe_land_image(
    image_path: str | Path | None,
    image_bytes: bytes | None = None
) -> VisionResult:
    if image_bytes is None and image_path is None:
        return VisionResult(
            description="No image provided.",
            model_used=vision_model_name(),
            image_provided=False,
        )

    if image_bytes is None:
        image_bytes = Path(image_path).read_bytes()

    # Use Groq vision if API key is present
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from groq import Groq
        client = Groq(api_key=groq_key)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{_mime_type(image_bytes)};base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": VISION_PROMPT
                        }
                    ]
                }
            ],
            max_tokens=1200,
        )
        description = _strip_thinking(response.choices[0].message.content or "")
        return VisionResult(
            description=description,
            model_used=f"{GROQ_VISION_MODEL} (Groq)",
            image_provided=True,
        )

    # Fall back to local Ollama
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": VISION_MODEL,
        "prompt": VISION_PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 400},
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    description = resp.json().get("response", "No description returned.").strip()
    return VisionResult(
        description=description,
        model_used=f"{VISION_MODEL} (Ollama)",
        image_provided=True,
    )


def vision_model_name() -> str:
    if os.getenv("GROQ_API_KEY"):
        return f"{GROQ_VISION_MODEL} (Groq)"
    return f"{VISION_MODEL} (Ollama)"


def _mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _strip_thinking(text: str) -> str:
    """Some Groq models prefix their answer with a <think>...</think> block."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()