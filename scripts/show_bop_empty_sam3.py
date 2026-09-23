"""Illustrate an actual saved zero-proposal SAM3 response, not a returned blank mask."""

from PIL import Image, ImageDraw, ImageFont

from scripts.inspect_bop_mask_cad_oracles import ROOT, read_json


def main():
    output = ROOT / "outputs/bop_sam_depth_20260919"
    folder = output / "sam3/scene_000002_000003"
    response = read_json(folder / "response.json")
    assert response[0]["predictions"]["predictions"] == []
    assert read_json(folder / "proposals.json") == []
    rgb = Image.open(ROOT / "outputs/cache/bop_tless/data/tless/test_primesense/000002/rgb/000003.png").convert("RGB")
    panel = Image.new("RGB", (1440, 650), "#f4f6f8")
    draw = ImageDraw.Draw(panel)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    heading, text = [ImageFont.truetype(font_path, size) for size in (24, 19)]
    draw.text((18, 14), 'Saved SAM3 result: prompt "object", threshold 0.5', font=heading, fill="#172531")
    draw.text((18, 48), "Scene 2, frame 3 | Original RGB input", font=text, fill="#405263")
    draw.text((738, 48), "Returned masks: 0 | Empty output displayed below", font=text, fill="#405263")
    panel.paste(rgb, (0, 82))
    draw.rectangle((720, 82, 1439, 621), fill="black")
    draw.text((915, 310), '"predictions": []', font=heading, fill="white")
    draw.text((800, 351), "The API returned no masks, not a blank mask image.", font=text, fill="#c8d0d8")
    draw.text((18, 626), "Earlier text-prompted SAM3 run. Automatic original SAM later returned proposals for this frame.", font=text, fill="#405263")
    panel.save(output / "sam3_zero_masks_example.png")
    print(output / "sam3_zero_masks_example.png", flush=True)


if __name__ == "__main__":
    main()
