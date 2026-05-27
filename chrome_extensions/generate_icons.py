"""
Run this once to generate the icons/ folder.
Requires Pillow: pip install Pillow

Output: icons/icon16.png, icons/icon48.png, icons/icon128.png
"""
import os
from PIL import Image, ImageDraw, ImageFont

os.makedirs("icons", exist_ok=True)

def make_icon(size):
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background circle — dark blue
    padding = size // 10
    draw.ellipse(
        [padding, padding, size - padding, size - padding],
        fill=(30, 58, 95, 255)
    )

    # Globe emoji approximation — white circle outline
    inner = size // 4
    draw.ellipse(
        [inner, inner, size - inner, size - inner],
        outline=(122, 179, 212, 255),
        width=max(1, size // 16)
    )

    # Horizontal line through centre
    mid = size // 2
    w   = max(1, size // 16)
    draw.line([(padding + 2, mid), (size - padding - 2, mid)],
              fill=(122, 179, 212, 255), width=w)

    # Vertical line through centre
    draw.line([(mid, padding + 2), (mid, size - padding - 2)],
              fill=(122, 179, 212, 255), width=w)

    return img

for sz in [16, 48, 128]:
    icon = make_icon(sz)
    icon.save(f"icons/icon{sz}.png")
    print(f"Saved icons/icon{sz}.png")

print("Done. Icons ready for manifest.json.")