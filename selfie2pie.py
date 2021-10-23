#!/usr/bin/env python3
"""This script converts images to grayscale images and resize them into
the same resolution as the CMU PIE image"""

from pathlib import Path

from PIL import Image

# PIE resolution
width = 32
height = 32

# Follow PIE naming scheme
cnt = 0

# Target directory: resized
Path("resized").mkdir(exist_ok=True)

# Source directory: selfies
# Accept any image format supported by Pillow
# Assume all files are images!
for filename in Path("selfies").glob("*"):
    cnt = cnt + 1
    im = Image.open(filename)
    # Convert to grayscale and resize to PIE resolution
    im_processed = im.convert("L").resize((width, height))
    im_processed.save(Path("resized") / (str(cnt) + ".jpg"))
