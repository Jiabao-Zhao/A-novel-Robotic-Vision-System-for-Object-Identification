from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE = Path(r"C:\Users\zhaoj\AppData\Local\Temp\codex-clipboard-d24718c5-5fba-4102-9ca1-7cc9c7f1ceb8.png")
OUTPUT = Path(r"D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification")


ASSETS = {
    "workflow_monitor.png": (33, 72, 147, 164),
    "icon_gear_program.png": (111, 130, 147, 165),
    "workflow_robot.png": (188, 77, 304, 150),
    "icon_product_a.png": (358, 112, 412, 154),
    "icon_new_product.png": (488, 111, 547, 155),
    "icon_check.png": (367, 79, 402, 114),
    "icon_not_adaptable.png": (498, 78, 534, 114),
    "icon_flexible_systems.png": (53, 187, 104, 233),
    "arrow_right_1.png": (157, 96, 183, 119),
    "arrow_right_2.png": (313, 96, 340, 119),
    "arrow_down.png": (250, 173, 274, 197),
}


def border_connected_background(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.int16)
    minimum = rgb.min(axis=2)
    spread = rgb.max(axis=2) - minimum
    candidate = (minimum >= 198) & (spread <= 78)

    height, width = candidate.shape
    connected = np.zeros_like(candidate, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        if candidate[0, x]:
            queue.append((0, x))
        if candidate[height - 1, x]:
            queue.append((height - 1, x))
    for y in range(height):
        if candidate[y, 0]:
            queue.append((y, 0))
        if candidate[y, width - 1]:
            queue.append((y, width - 1))

    while queue:
        y, x = queue.popleft()
        if connected[y, x] or not candidate[y, x]:
            continue
        connected[y, x] = True
        if y:
            queue.append((y - 1, x))
        if y + 1 < height:
            queue.append((y + 1, x))
        if x:
            queue.append((y, x - 1))
        if x + 1 < width:
            queue.append((y, x + 1))

    rgba[connected, 3] = 0
    return rgba


def extract(name: str, bounds: tuple[int, int, int, int], source: Image.Image) -> None:
    crop = source.crop(bounds).convert("RGBA")
    rgba = np.array(crop)

    if name == "workflow_monitor.png":
        # Remove rasterized program text while preserving the monitor bezel.
        x0, y0, _, _ = bounds
        left, top, right, bottom = 39 - x0, 79 - y0, 139 - x0, 133 - y0
        for y in range(top, bottom):
            shade = int(round(250 - 0.035 * (y - top)))
            rgba[y, left:right, :3] = (shade, shade, shade)
            rgba[y, left:right, 3] = 255

        # The gear badge is exported independently and sits above the monitor.
        yy, xx = np.ogrid[: rgba.shape[0], : rgba.shape[1]]
        cx, cy = 128 - x0, 147 - y0
        rgba[(xx - cx) ** 2 + (yy - cy) ** 2 <= 19**2, 3] = 0

    rgba = border_connected_background(rgba)
    Image.fromarray(rgba, mode="RGBA").save(OUTPUT / name)


def main() -> None:
    source = Image.open(SOURCE).convert("RGB")
    if source.size != (591, 233):
        raise ValueError(f"Unexpected source size: {source.size}")
    for name, bounds in ASSETS.items():
        extract(name, bounds, source)


if __name__ == "__main__":
    main()
