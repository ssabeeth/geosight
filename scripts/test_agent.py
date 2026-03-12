from geosight.agent import run_agent
from pathlib import Path

print("Running GeoSight agent for TQ13 8HH (Dartmoor) with image...\n")

# Change this path to your image
image_bytes = Path("/Users/syedsabeeth/Downloads/dartmoor.jpg").read_bytes()

result = run_agent("TQ13 8HH", image_bytes=image_bytes)

if result["errors"]:
    print("Warnings:", result["errors"])

print(result["report"])