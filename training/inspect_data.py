
from pathlib import Path
from collections import Counter

DATA_DIR = Path(r"D:\ISL\data\dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATA_EXTENSIONS = {".csv", ".json", ".txt", ".npy", ".npz"}


def inspect_dataset():
    print("=" * 55)
    print("       ISL DATASET INSPECTOR")
    print("=" * 55)

    if not DATA_DIR.is_dir():
        print(f"Dataset folder not found: {DATA_DIR}")
        return

    print(f"\nDataset: {DATA_DIR}")
    print("\nScanning one directory level at a time...")
    print("Press Ctrl+C if you need to stop.\n")

    total_files = 0
    total_images = 0
    total_data_files = 0
    extensions = Counter()
    directories_seen = 0
    max_directories = 5000

    # Only inspect the top-level directory first.
    try:
        top_items = list(DATA_DIR.iterdir())
    except OSError as error:
        print(f"Cannot read dataset directory: {error}")
        return

    print("TOP-LEVEL CONTENTS")
    print("-" * 55)

    for item in top_items[:100]:
        kind = "Folder" if item.is_dir() else "File"
        print(f"{kind}: {item.name}")

    if len(top_items) > 100:
        print(f"... {len(top_items) - 100} more entries")

    print("\nSUMMARY")
    print("-" * 55)

    # Walk the directory tree incrementally instead of using rglob.
    # Do not open or load image contents.
    pending = [DATA_DIR]

    try:
        while pending:
            current = pending.pop()
            directories_seen += 1

            if directories_seen > max_directories:
                print("\nSafety limit reached: 5000 directories.")
                print("Increase max_directories if needed.")
                break

            try:
                with __import__("os").scandir(current) as entries:
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                pending.append(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False):
                                total_files += 1
                                suffix = Path(entry.name).suffix.lower()
                                extensions[suffix or "[no extension]"] += 1

                                if suffix in IMAGE_EXTENSIONS:
                                    total_images += 1

                                if suffix in DATA_EXTENSIONS:
                                    total_data_files += 1

                        except OSError:
                            continue

            except OSError as error:
                print(f"Skipped unreadable folder: {current} ({error})")

            # Periodic progress for large datasets
            if directories_seen % 100 == 0:
                print(
                    f"Scanned {directories_seen} folders; "
                    f"found {total_files} files..."
                )

    except KeyboardInterrupt:
        print("\nScan interrupted by user. Partial results follow.")

    print(f"\nDirectories scanned: {directories_seen}")
    print(f"Files found: {total_files}")
    print(f"Images found: {total_images}")
    print(f"Data files found: {total_data_files}")

    print("\nFILE TYPES")
    print("-" * 55)

    for extension, count in extensions.most_common():
        print(f"{extension}: {count}")

    print("\nINSPECTION FINISHED")


if __name__ == "__main__":
    inspect_dataset()
