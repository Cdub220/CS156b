from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

root = Path("/resnick/groups/CS156b/from_central/data/train")

if not root.exists():
    print("Could not find training folder.")
    raise SystemExit

files = []
for path in root.rglob("*"):
    if path.is_file() and path.suffix.lower() in VALID_EXTS:
        files.append(path)

if len(files) == 0:
    print("No image files found under training folder.")
    raise SystemExit

size_counts = Counter()
mode_counts = Counter()
format_counts = Counter()
bad_files = []

min_width = None
max_width = None
min_height = None
max_height = None

for i, path in enumerate(files, 1):
    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            fmt = img.format

        size_counts[(width, height)] += 1
        mode_counts[mode] += 1
        format_counts[fmt] += 1

        if min_width is None or width < min_width:
            min_width = width
        if max_width is None or width > max_width:
            max_width = width
        if min_height is None or height < min_height:
            min_height = height
        if max_height is None or height > max_height:
            max_height = height

    except (UnidentifiedImageError, OSError, ValueError) as e:
        bad_files.append(f"{path} :: {type(e).__name__}: {e}")

    if i % 500 == 0 or i == len(files):
        print(f"Checked {i}/{len(files)}")

print("\n=== SUMMARY ===")
print("Total image files:", len(files))
print("Readable files:", len(files) - len(bad_files))
print("Bad files:", len(bad_files))

if min_width is not None:
    print("Width range:", min_width, "to", max_width)
    print("Height range:", min_height, "to", max_height)

print("\nTop image sizes:")
for size, count in size_counts.most_common(10):
    print(size, ":", count)

print("\nImage modes:")
for mode, count in mode_counts.items():
    print(mode, ":", count)

print("\nFormats:")
for fmt, count in format_counts.items():
    print(fmt, ":", count)

if len(bad_files) > 0:
    with open("bad_files.txt", "w") as f:
        for line in bad_files:
            f.write(line + "\n")
    print("\nWrote bad files to bad_files.txt")