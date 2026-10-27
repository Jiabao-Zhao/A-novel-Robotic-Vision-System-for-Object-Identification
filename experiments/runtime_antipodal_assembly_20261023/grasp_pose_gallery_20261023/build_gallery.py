"""Build an image gallery from the saved 14-part pose-render manifest.

Usage: python build_gallery.py OUTPUT_DIRECTORY
Only index.html and README.md in that output directory are written. Rendering,
pose selection, source results and manifest records remain the caller's work.
"""

import html
import json
from pathlib import Path
import sys


TITLE = "Generated grasp positions for all 14 parts"


def _image_reference(output, value):
    reference = str(value).replace("\\", "/")
    candidate = Path(reference)
    if candidate.is_absolute() or ":" in reference.split("/")[0]:
        raise ValueError(f"Gallery images must use relative paths: {reference}")
    resolved = (output / candidate).resolve()
    if not resolved.is_relative_to(output) or not resolved.is_file():
        raise ValueError(f"Gallery image is missing or outside its output directory: {reference}")
    return reference


def _is_followup(record):
    return "follow" in str(record.get("source_kind", "")).lower()


def _badge(record):
    return "8 mm follow-up pose" if _is_followup(record) else ""


def _millimetres(value):
    return f"{float(value) * 1000:.1f}"


def build_gallery(output_dir):
    """Read manifest.json and return the generated local HTML path."""
    output = Path(output_dir).resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["parts"]
    if not isinstance(records, list) or len(records) != 14:
        raise ValueError("The gallery requires exactly 14 part records.")
    if len({record["part"] for record in records}) != 14:
        raise ValueError("Each of the 14 part records must identify a different part.")
    clean = []
    cards = []
    for index, record in enumerate(records):
        label = str(record["label"])
        images = {name: _image_reference(output, record["images"][name])
                  for name in ("overview", "detail", "card")}
        badge = _badge(record)
        width = _millimetres(record["grasp_opening_m"])
        span = _millimetres(record["touch_span_m"])
        clean.append({"label": label, "part": str(record["part"]),
                      "images": images, "badge": badge, "opening_mm": width, "span_mm": span})
        badge_html = f'<span class="badge">{html.escape(badge)}</span>' if badge else ""
        cards.append(
            f'<article class="card" id="part-{index + 1}">'
            '<div class="card-heading">'
            f'<span class="number">{index + 1:02d}</span><h2>{html.escape(label)}</h2>{badge_html}</div>'
            f'<button class="image-button" data-index="{index}" data-view="overview" '
            f'aria-label="Enlarge the grasp position for {html.escape(label, quote=True)}">'
            f'<img src="{html.escape(images["card"], quote=True)}" '
            f'alt="Generated gripper position on {html.escape(label, quote=True)}" '
            'loading="lazy" decoding="async"></button>'
            '<div class="card-footer">'
            f'<span>Saved opening setting: {width} mm</span><div class="card-actions">'
            f'<button data-index="{index}" data-view="overview">Whole part</button>'
            f'<button data-index="{index}" data-view="detail">Close-up</button>'
            '</div></div></article>'
        )
    data = json.dumps(clean, ensure_ascii=False).replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    document = HTML.replace("__TITLE__", html.escape(TITLE)).replace("__CARDS__", "\n".join(cards)).replace("__DATA__", data)
    html_path = output / "index.html"
    html_path.write_text(document, encoding="utf-8")
    readme = [f"# {TITLE}", "", "[Open the image gallery](index.html). Select a picture or **Close-up** to enlarge it.", "",
              "These are static pictures of generated grasp positions. They do not show an assembly result.", "",
              "Orange is the part. Transparent blue is the gripper, shown open before it closes. Green rings mark both planned finger touches, including the far side. The blue arrow shows the approach direction.", "",
              "The 8 mm round pin uses its saved follow-up pose; the other thirteen parts use their earlier saved poses.", "",
              "| Part | Saved pose | Opening setting | Pictures |", "| --- | --- | ---: | --- |"]
    for record, item in zip(records, clean):
        label = item["label"].replace("|", "\\|")
        source = "8 mm follow-up" if _is_followup(record) else "Earlier saved pose"
        readme.append(f'| {label} | {source} | {item["opening_mm"]} mm | '
                      f'[Whole part]({item["images"]["overview"]}) · [Close-up]({item["images"]["detail"]}) |')
    readme.extend(["", "The full saved poses, calculated touch points and source-result references are in [manifest.json](manifest.json).", ""])
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return html_path


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>__TITLE__</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#18242e;font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1280px;margin:auto;padding:32px 24px 48px}h1{font-size:clamp(25px,3vw,38px);line-height:1.15;margin:0 0 12px;letter-spacing:-.025em}
.intro{max-width:850px;color:#52616d;margin:0 0 10px}.notice{margin:0 0 20px;color:#52616d;font-size:14px}
.legend{display:flex;flex-wrap:wrap;gap:12px 26px;padding:14px 0 24px}.legend-item{display:inline-flex;align-items:center;gap:9px;font-size:14px}
.touch-dots{display:inline-flex;gap:5px}.touch-dots i{display:block;width:10px;height:10px;background:#20ac58;border:1px solid #107637;border-radius:50%}
.arrow{display:inline-block;width:24px;height:3px;background:#287ad9;position:relative;margin-right:3px}.arrow:after{content:"";position:absolute;right:-1px;top:-4px;border-left:8px solid #287ad9;border-top:5px solid transparent;border-bottom:5px solid transparent}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.card{min-width:0;border:1px solid #dbe2e8;border-radius:12px;overflow:hidden;background:#fff}
.card-heading{display:flex;align-items:center;flex-wrap:wrap;gap:9px;padding:14px 16px 10px;min-height:56px}.number{color:#7d8b95;font-size:12px;font-weight:650;font-variant-numeric:tabular-nums}
h2{font-size:18px;line-height:1.3;margin:0;flex:1;min-width:100px}.badge{font-size:11px;font-weight:650;background:#e8f3ff;color:#175996;border-radius:20px;padding:4px 9px;white-space:nowrap}
button,a{font:inherit}button{cursor:pointer}button:focus-visible,a:focus-visible{outline:3px solid #217bd5;outline-offset:3px}
.image-button{display:block;width:100%;border:0;background:#f4f7f8;padding:0;line-height:0}.image-button img{display:block;width:100%;height:auto}.image-button:hover{filter:brightness(.98)}
.card-footer{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;padding:11px 16px 14px;color:#596773;font-size:12px}.card-actions{display:flex;gap:7px}
.card-actions button,.tabs button,.dialog-tools button,.dialog-tools a{border:1px solid #ccd8e2;background:#fff;color:#245776;border-radius:7px;padding:6px 10px;text-decoration:none;font-size:13px;line-height:1.3}
.card-actions button:hover,.tabs button:hover,.dialog-tools button:hover{background:#edf5fb}.footer{margin:28px 0 0;color:#6a7883;font-size:13px}
dialog{width:min(1240px,calc(100vw - 28px));max-height:calc(100vh - 28px);border:1px solid #c7d2db;border-radius:14px;padding:0;background:white;color:#18242e}
dialog::backdrop{background:rgb(12 23 34/.75)}.dialog-heading{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:15px 18px 10px}.dialog-heading h2{font-size:21px}
.dialog-tools{display:flex;align-items:center;gap:8px;margin-left:auto}.position{font-size:12px;color:#667782}.tabs{display:flex;align-items:center;gap:8px;padding:0 18px 12px}.tabs button[aria-pressed=true]{background:#e8f3ff;border-color:#8bb9df;color:#184f80}
.image-stage{display:flex;align-items:center;justify-content:center;background:#f3f6f8;padding:6px;min-height:200px}.image-stage img{display:block;max-width:100%;max-height:calc(100vh - 235px);width:auto;height:auto;object-fit:contain}
.dialog-caption{padding:10px 18px 15px;font-size:13px;color:#52616d;margin:0}.dialog-caption .static{display:block;color:#778590;font-size:12px;margin-top:4px}
@media(max-width:700px){main{padding:22px 12px 32px}.grid{gap:12px}.card-heading{padding:10px 10px 8px;gap:6px}h2{font-size:15px;min-width:80px}.badge{font-size:10px;padding:3px 6px}.card-footer{padding:9px 10px 11px}.card-actions{gap:5px;flex-wrap:wrap}.card-actions button{font-size:11px;padding:5px 7px}.number{font-size:10px}.dialog-heading{padding:12px}.dialog-heading h2{font-size:18px}.dialog-tools{margin-left:0}.tabs{padding-left:12px}.legend{gap:10px 16px}.legend-item{font-size:12px}}
@media print{main{max-width:none;padding:0}.grid{gap:10px}.card{break-inside:avoid}.card-actions{display:none}.image-button{border:0}dialog{display:none}.footer{font-size:10px}}
</style>
</head>
<body>
<main>
<header>
<h1>__TITLE__</h1>
<p class="intro">See where the gripper is positioned on each part. Select a picture for a larger view, or choose <strong>Close-up</strong> to inspect the fingers and touch points.</p>
<p class="notice">The gripper is shown open, just before closing. Colors are added for clarity. The 8 mm round pin uses its saved follow-up pose. These pictures do not show assembly results.</p>
<div class="legend" aria-label="Image legend">
<span class="legend-item">Orange: part · Transparent blue: gripper</span>
<span class="legend-item"><span class="touch-dots" aria-hidden="true"><i></i><i></i></span>Green rings: planned finger touches on both sides</span>
<span class="legend-item"><span class="arrow" aria-hidden="true"></span>Blue arrow: approach direction</span>
</div>
</header>
<section class="grid" aria-label="All fourteen parts">__CARDS__</section>
<p class="footer">The pictures show saved generated poses. A pictured pose alone does not establish a successful pickup or assembly.</p>
</main>
<dialog id="zoom" aria-labelledby="zoom-title">
<div class="dialog-heading"><h2 id="zoom-title"></h2><span id="zoom-badge" class="badge" hidden></span>
<div class="dialog-tools"><span id="position" class="position"></span><button id="previous" aria-label="Previous part">←</button><button id="next" aria-label="Next part">→</button><button id="close">Close</button></div></div>
<div class="tabs" aria-label="Choose picture"><button data-zoom-view="overview" aria-pressed="true">Whole part</button><button data-zoom-view="detail" aria-pressed="false">Close-up</button><a id="open-image" target="_blank" rel="noreferrer">Open image</a></div>
<div class="image-stage"><img id="zoom-image" alt=""></div>
<p class="dialog-caption"><span id="measurements"></span><span class="static">Static picture of a generated pose. No assembly result is implied.</span></p>
</dialog>
<script type="application/json" id="records">__DATA__</script>
<script>
"use strict";
const records=JSON.parse(document.getElementById("records").textContent);
const dialog=document.getElementById("zoom"),picture=document.getElementById("zoom-image");
let active=0,view="overview";
function show(index,nextView){
 active=(index+records.length)%records.length;view=nextView||view;
 const item=records[active];
 document.getElementById("zoom-title").textContent=item.label;
 document.getElementById("position").textContent=(active+1)+" / "+records.length;
 const badge=document.getElementById("zoom-badge");badge.textContent=item.badge;badge.hidden=!item.badge;
 picture.src=item.images[view];picture.alt="Generated gripper position on "+item.label+" — "+(view==="detail"?"close-up":"whole part");
 document.getElementById("open-image").href=item.images[view];
 document.getElementById("measurements").textContent="Saved opening setting: "+item.opening_mm+" mm. Distance between the green rings: "+item.span_mm+" mm.";
 document.querySelectorAll("[data-zoom-view]").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.zoomView===view)));
 if(!dialog.open)dialog.showModal();
}
document.querySelectorAll("[data-index]").forEach(button=>button.addEventListener("click",()=>show(Number(button.dataset.index),button.dataset.view)));
document.querySelectorAll("[data-zoom-view]").forEach(button=>button.addEventListener("click",()=>show(active,button.dataset.zoomView)));
document.getElementById("close").addEventListener("click",()=>dialog.close());
document.getElementById("previous").addEventListener("click",()=>show(active-1));
document.getElementById("next").addEventListener("click",()=>show(active+1));
dialog.addEventListener("click",event=>{if(event.target===dialog){const bounds=dialog.getBoundingClientRect();if(event.clientX<bounds.left||event.clientX>bounds.right||event.clientY<bounds.top||event.clientY>bounds.bottom)dialog.close();}});
dialog.addEventListener("keydown",event=>{if(event.key==="ArrowLeft"){event.preventDefault();show(active-1);}if(event.key==="ArrowRight"){event.preventDefault();show(active+1);}});
</script>
</body>
</html>
'''


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python build_gallery.py OUTPUT_DIRECTORY")
    print(build_gallery(sys.argv[1]))
