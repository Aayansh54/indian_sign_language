
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw

DATA_DIR = Path(r"D:\ISL\data\dataset")
OUTPUT_DIR = Path(r"D:\ISL\model")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not DATA_DIR.is_dir():
        print(f"Dataset not found: {DATA_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classes = sorted(
        folder for folder in DATA_DIR.iterdir()
        if folder.is_dir()
    )

    print("\nISL DATASET CLASS REPORT")
    print("-" * 55)

    total_images = 0
    counts = Counter()
    samples = []
    unreadable = 0

    for folder in classes:
        count = 0
        first_valid_image = None

        for path in folder.iterdir():
            if not path.is_file():
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            count += 1

            if first_valid_image is None:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    first_valid_image = path
                except Exception:
                    unreadable += 1

        counts[folder.name] = count
        total_images += count

        print(f"Class {folder.name:>3}: {count:>6} images")

        if first_valid_image is not None:
            samples.append((folder.name, first_valid_image))

    print("-" * 55)
    print(f"Total classes: {len(classes)}")
    print(f"Total images:  {total_images}")
    print(f"Unreadable sample images: {unreadable}")

    if counts:
        print(f"Smallest class: {min(counts, key=counts.get)} "
              f"({min(counts.values())} images)")
        print(f"Largest class:  {max(counts, key=counts.get)} "
              f"({max(counts.values())} images)")

    # Create a contact sheet with one sample per class.
    cell_width = 180
    cell_height = 190
    columns = 4
    rows = (len(samples) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (columns * cell_width, max(1, rows) * cell_height),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    for index, (label, path) in enumerate(samples):
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((160, 145))

                x = (index % columns) * cell_width
                y = (index // columns) * cell_height

                image_x = x + (cell_width - image.width) // 2
                sheet.paste(image, (image_x, y + 8))
                draw.text((x + 10, y + 160),
                          f"Label: {label}", fill="black")

        except Exception as error:
            print(f"Could not preview {path}: {error}")

    output_path = OUTPUT_DIR / "dataset_samples.jpg"
    sheet.save(output_path, quality=90)

    print(f"\nSample contact sheet saved to:\n{output_path}")


if __name__ == "__main__":
    main()
