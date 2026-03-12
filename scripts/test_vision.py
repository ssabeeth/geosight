from geosight.tools.vision import describe_land_image
from pathlib import Path

# Change this path to any image on your Mac
image_path = "/Users/syedsabeeth/Downloads/dartmoor.jpg"

result = describe_land_image(image_path=image_path)
print(result.description)