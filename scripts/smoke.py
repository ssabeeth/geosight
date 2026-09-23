"""
End-to-end run against the live APIs and the configured model.
Usage: python scripts/smoke.py "SP6 1EF" [--image path/to/photo.jpg]
"""

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from geosight.agent import run_agent  # noqa: E402  (needs .env loaded first)

parser = argparse.ArgumentParser()
parser.add_argument("postcode")
parser.add_argument("--image", type=Path)
args = parser.parse_args()

start = time.time()
result = run_agent(args.postcode, image_bytes=args.image.read_bytes() if args.image else None)
print(result["report"])
print(f"\n--- {time.time() - start:.0f}s")
for err in result["errors"]:
    print("WARNING:", err)
