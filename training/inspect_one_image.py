
from pathlib import Path
from PIL import Image

image_path = Path(
    r"D:\ISL\data\dataset\A\Image_1679027619.2595825.jpg"
)

if not image_path.exists():
    print("Image not found:", image_path)
else:
    with Image.open(image_path) as image:
        print("Image:", image_path.name)
        print("Dimensions:", image.size)
        print("Format:", image.format)
        print("Color mode:", image.mode)
        print("File size (KB):", round(image_path.stat().st_size / 1024, 2))

        # Save a copy for easy viewing; original remains unchanged.
        output = Path(r"D:\ISL\model\sample_original.png")
        image.convert("RGB").save(output)
        print("Preview saved to:", output)
