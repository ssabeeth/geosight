"""
Vision tool — describes a land/site photograph using LLaVA via Ollama.
Free, runs locally. No API key required.
Ollama must be running with llava:7b pulled.
"""

import base64
import os
from pathlib import Path

import requests
from pydantic import BaseModel

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")

VISION_PROMPT = """You are a land and environmental analyst examining a site photograph.
Describe what you observe in structured terms relevant to land use and planning:

1. **Terrain & Topography**: flat, sloped, valley, hillside, coastal, etc.
2. **Vegetation & Ecology**: woodland, grassland, crops, hedgerows, scrub, wetland, etc.
3. **Land Use Indicators**: agricultural equipment, buildings, infrastructure, tracks, fencing
4. **Water Features**: streams, ditches, standing water, flooding signs
5. **Condition**: well-managed, overgrown, degraded, signs of recent activity
6. **Estimated Scale & Setting**: approximate area visible, rural/peri-urban/urban

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
            model_used=VISION_MODEL,
            image_provided=False,
        )

    if image_bytes is None:
        image_bytes = Path(image_path).read_bytes()

    # Use Groq vision if API key is present
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        import base64 as b64
        from groq import Groq
        client = Groq(api_key=groq_key)
        image_b64 = b64.b64encode(image_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model="llava-v1.5-7b-4096-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": VISION_PROMPT
                        }
                    ]
                }
            ],
            max_tokens=400,
        )
        description = response.choices[0].message.content.strip()
        return VisionResult(
            description=description,
            model_used="llava-v1.5-7b-4096-preview",
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
        model_used=VISION_MODEL,
        image_provided=True,
    )