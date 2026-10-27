"""Human gallery from completed pickup records; never calls a model or robot.

Run only after all 42 timed trials finish. Input pictures, VLM responses and
physical pickup records are shown separately so a drawn grasp is not mistaken
for an executed pickup. Only current pickup views are included.
"""
import hashlib
import html
import json
from pathlib import Path
import sys


ARMS = ("geometric", "grconvnet_score", "grconvnet_vlm")
ARM_LABELS = {"geometric": "Geometric search", "grconvnet_score": "GR-ConvNet score order",
              "grconvnet_vlm": "GR-ConvNet + VLM order"}
LABELS = {"Gear_Large": "Large gear", "Gear_Medium": "Medium gear", "Gear_Small": "Small gear",
          "Waterproof_Male": "Waterproof connector", "M12_Hex_Nut": "M12 nut", "M16_Hex_Nut": "M16 nut",
          "RGOCG16-50_16mm": "16 mm round pin", "KET16_Square_16mm": "16 mm rectangular pin",
          "RGOCG12-50_12mm": "12 mm round pin", "RGOCG8-50_8mm": "8 mm round pin",
          "KET12_Square_12mm": "12 mm rectangular pin", "KET8_Square_8mm": "8 mm rectangular pin",
          "USB_Male": "USB connector", "DSUB_Male": "D-sub connector"}
GROUPS = {"gears_and_nuts": ("Gear_Large", "Gear_Medium", "Gear_Small", "M12_Hex_Nut", "M16_Hex_Nut"),
          "pins": ("RGOCG16-50_16mm", "KET16_Square_16mm", "RGOCG12-50_12mm", "KET12_Square_12mm", "RGOCG8-50_8mm", "KET8_Square_8mm"),
          "connectors": ("Waterproof_Male", "USB_Male", "DSUB_Male")}
PHASES = {"approach": "moving toward the part", "descend": "lowering the open gripper",
          "close": "closing the fingers", "lift": "lifting the part", "hold_lift": "holding after lifting"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def escape(value):
    return html.escape(str(value), quote=True)


def pretty_body(value):
    text = str(value).replace("gripper0_", "gripper ").replace("robot0_", "robot ").replace("_", " ")
    return text or "unnamed contact body"


def result_reason(result):
    status = result.get("status", "missing_result")
    if status == "pickup_passed":
        return "Picked up and held for one second with both fingers touching."
    if status == "vlm_output_unavailable":
        return "No physical pickup: the VLM response did not pass the required output checks."
    if status == "grasp_failed":
        return "The fingers closed, but the recorded forces did not show both fingers holding the part."
    if status == "pickup_failed":
        reasons = []
        if result.get("lift_m", 0) < .12:
            reasons.append(f"Final height increase was {result.get('lift_m', 0)*1000:.1f} mm, below 120 mm")
        if not result.get("bilateral_after_lift_and_hold", False):
            reasons.append("both fingers were not holding after the one-second hold")
        if not result.get("robot_collision_penetration_below_1mm", False):
            reasons.append("recorded robot overlap reached the 1 mm limit")
        return "; ".join(reasons) + "."
    if result.get("error"):
        phase = PHASES.get(result.get("last_executed_phase"), str(result.get("last_executed_phase", "setup")))
        return f"Stopped during {phase}: {result['error']}"
    if status == "no_pickup_path_found_in_candidates":
        number = result.get("path_candidates_checked", 0)
        if not result.get("shortlisted_count", 0):
            return "No physical pickup: no candidate grasp survived the stated width and table checks."
        counts = {}
        details = []
        for rejected in result.get("planning_rejections", []):
            failed = next((c for c in rejected.get("checks", []) if c.get("status") != "passed"), None)
            if failed is None:
                continue
            phase = PHASES.get(failed.get("phase"), failed.get("phase", "movement"))
            if failed["status"] == "target_obstruction":
                cause = "the open fingers would overlap the part"
            elif failed["status"] == "ik_failed":
                cause = "robot joint angles were not found within the solver limits"
            elif failed["status"] == "collision":
                contacts = failed.get("contacts", [])
                bodies = sorted({pretty_body(body) for c in contacts for body in c.get("bodies", [])})
                cause = "the robot would touch " + (" / ".join(bodies) if bodies else "the table or itself")
            else:
                cause = str(failed["status"]).replace("_", " ")
            key = phase + ": " + cause
            counts[key] = counts.get(key, 0)+1
            if failed.get("reason"):
                details.append(str(failed["reason"]))
            for contact in failed.get("contacts", []):
                if "penetration_m" in contact:
                    details.append(f"largest listed overlap {contact['penetration_m']*1000:.3f} mm")
        joined = "; ".join(f"{count} during {cause}" for cause, count in counts.items())
        return f"No physical pickup. {number} candidate paths checked. {joined}." + (" " + "; ".join(dict.fromkeys(details)) if details else "")
    return str(status).replace("_", " ") + "."


def file_link(run, path):
    """Resolve only existing files inside this run; output is gallery-relative."""
    path = Path(path).resolve()
    if not path.is_relative_to(run) or not path.is_file():
        return None
    return "../" + path.relative_to(run).as_posix()


def picture(run, path, caption):
    link = file_link(run, path)
    if link is None:
        return ""
    return (f'<figure><a href="{escape(link)}" target="_blank"><img src="{escape(link)}" loading="lazy" '
            f'alt="{escape(caption)}"></a><figcaption>{escape(caption)}</figcaption></figure>')


def time_text(value):
    return "Unavailable" if value is None else f"{float(value):.3f} s"


def chosen_ids(run, arm, part, result):
    pool_index = result.get("selected_pool_index")
    if pool_index is None:
        return {"pool_number": None, "display_id": None, "network_candidate_id": None}
    candidates = result.get("candidates", [])
    network = candidates[pool_index].get("network_candidate_id") if pool_index < len(candidates) else None
    display = None
    mapping_path = run / arm / part / "display_id_map.json"
    if mapping_path.is_file():
        display = next((int(k) for k, v in read(mapping_path).items() if v["pool_index"] == pool_index), None)
    return {"pool_number": pool_index+1, "display_id": display, "network_candidate_id": network}


def arm_card(run, arm, part, result):
    folder = run / arm / part
    ids = chosen_ids(run, arm, part, result)
    success = result.get("status") == "pickup_passed"
    attempts = int(result.get("physical_attempts", 0))
    lifted = result.get("lift_m")
    lift_label = "Unavailable" if lifted is None else f"{float(lifted)*1000:.1f} mm"
    selection = ("None selected" if ids["pool_number"] is None else
                 f"Candidate pool entry {ids['pool_number']}" +
                 (f" · network candidate {ids['network_candidate_id']}" if ids["network_candidate_id"] is not None else "") +
                 (f" · displayed ID {ids['display_id']}" if ids["display_id"] is not None else ""))
    note = result_reason(result)
    card = [f'<article class="arm"><h3>{escape(ARM_LABELS[arm])}</h3>',
            f'<p class="status {"passed" if success else "failed"}">{escape("Pickup passed" if success else "Pickup not passed")}</p>',
            f'<p class="reason">{escape(note)}</p>', '<dl>',
            f'<dt>Physical pickup attempts</dt><dd>{attempts}</dd>',
            f'<dt>Final height increase</dt><dd>{escape(lift_label)}</dd>',
            f'<dt>Candidate paths checked</dt><dd>{int(result.get("path_candidates_checked", 0))}</dd>',
            f'<dt>Decision time</dt><dd>{escape(time_text(result.get("decision_seconds")))}</dd>',
            f'<dt>Recorded total time</dt><dd>{escape(time_text(result.get("total_seconds")))}</dd>',
            '</dl>', f'<p class="selection">{escape(selection)}</p>']
    if attempts:
        card.append(picture(run, folder / "closed.png", "Actual simulation frame after the fingers closed"))
        card.append(picture(run, folder / "lifted_all_parts_visible.png", "Actual simulation frame after the lift and one-second hold"))
    else:
        card.append('<p class="no-motion">No physical pickup was attempted in this condition.</p>')
    source = file_link(run, folder / "result.json")
    card.append(f'<p><a href="{escape(source)}">Full result and movement-check details</a></p></article>')
    return "".join(card), ids


def vlm_panel(run, part, result):
    folder = run / "grconvnet_vlm" / part
    text = [f'<div class="vlm"><h3>What the VLM received and returned</h3>',
            picture(run, folder / "rank_input.png", "Exact numbered pickup candidate sheet supplied to the VLM; green outline is the target, red line joins the jaws, blue lines show fingers"),
            f'<p>VLM status: {escape(result.get("vlm_status", "not reached"))}.</p>']
    order = result.get("vlm_ranked_display_ids")
    if order is not None:
        text.append('<p><strong>VLM preference, best first:</strong> ' + escape(" → ".join(map(str, order))) + '.</p>')
    ranking_file = folder / "vlm/ranking.json"
    if ranking_file.is_file():
        ranking = read(ranking_file)
        observations = ranking.get("parsed", {}).get("observations", [])
        if observations:
            text.append('<p>These are model observations, not verified physical results:</p><ul>')
            text.extend(f'<li><strong>ID {escape(item.get("id"))}:</strong> {escape(item.get("observation", ""))}</li>' for item in observations)
            text.append('</ul>')
        if ranking.get("parse_error"):
            text.append('<p>Output check: '+escape(ranking["parse_error"])+'.</p>')
        raw = ranking.get("raw_text", ranking.get("raw_response", {}).get("message", {}).get("content", ""))
        text.append('<details><summary>Full raw VLM response</summary><pre>'+escape(raw)+'</pre></details>')
    rejections = result.get("proposal_mapping_rejections", [])
    if rejections:
        text.append('<details><summary>Network candidates rejected before VLM review</summary><ul>')
        text.extend(f'<li>Network candidate {escape(item["network_candidate_id"])}: {escape(item["reason"])}</li>' for item in rejections)
        text.append('</ul></details>')
    mapping = file_link(run, folder / "display_id_map.json")
    if mapping:
        text.append(f'<p><a href="{escape(mapping)}">Display-ID to saved candidate-pose mapping</a></p>')
    text.append('</div>')
    return "".join(text)


def contact_sheets(run, output, results):
    """Actual frames only. Blank cards explicitly mark unattempted pickups."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    def font(size, bold=False):
        return ImageFont.truetype(bold_path if bold else font_path, size)
    files = {}
    for group, parts in GROUPS.items():
        sheet = Image.new("RGB", (1360, 110+len(parts)*430), "#f1f5f9")
        draw = ImageDraw.Draw(sheet)
        draw.text((18, 10), "Actual pickups: geometric search and GR-ConvNet + VLM", font=font(29, True), fill="#172b40")
        draw.text((18, 57), "Frames are from recorded simulation. Blank cards indicate no physical pickup attempt.", font=font(21), fill="#526171")
        for row, part in enumerate(parts):
            for col, arm in enumerate(("geometric", "grconvnet_vlm")):
                result = results[arm, part]
                x, y = col*680, 110+row*430
                draw.rectangle((x+3,y+3,x+677,y+426),fill="white",outline="#c3cfda",width=2)
                draw.text((x+16,y+10), LABELS[part],font=font(25,True),fill="#172b40")
                draw.text((x+16,y+46),ARM_LABELS[arm],font=font(20),fill="#526171")
                folder = run / arm / part
                lifted = folder / "lifted_all_parts_visible.png"
                closed = folder / "closed.png"
                path = lifted if lifted.is_file() else closed if closed.is_file() else None
                if path:
                    image = ImageOps.contain(Image.open(path).convert("RGB"),(650,292),Image.Resampling.LANCZOS)
                    sheet.paste(image,(x+(680-image.width)//2,y+85+(292-image.height)//2))
                    caption = "After lift and hold" if path == lifted else "Fingers closed; lift not completed"
                else:
                    draw.text((x+29,y+184),"No physical pickup attempt",font=font(25,True),fill="#8a442c")
                    caption = str(result.get("status", "Unavailable")).replace("_"," ")
                status = "PICKUP PASSED" if result.get("status")=="pickup_passed" else "PICKUP NOT PASSED"
                draw.text((x+16,y+380),status+" · "+caption,font=font(18,True),fill="#237045" if status=="PICKUP PASSED" else "#8a442c")
                draw.text((x+16,y+407),"Decision "+time_text(result.get("decision_seconds"))+" · Total "+time_text(result.get("total_seconds")),font=font(16),fill="#526171")
        name = group+"_actual_pickup_comparison.png"
        sheet.save(output/name)
        files[group]=name
    return files


def build(run_directory):
    run = Path(run_directory).resolve()
    manifest = read(run/"manifest.json")
    parts = manifest["parts"]
    if len(parts) != 14 or set(manifest["arms"]) != set(ARMS):
        raise ValueError("Expected the complete 14-part, three-condition run")
    results = {(arm, part): read(run/arm/part/"result.json") for arm in ARMS for part in parts}
    if len(read(run/"results.json")) != 42:
        raise ValueError("Wait for all 42 timed trials before building the gallery")
    hashes = {f"{arm}/{part}/result.json": digest(run/arm/part/"result.json") for arm in ARMS for part in parts}
    output = run/"gallery"
    output.mkdir(exist_ok=False)
    sections, chosen = [], {}
    for index, part in enumerate(parts, 1):
        cards=[]
        for arm in ARMS:
            card, ids=arm_card(run,arm,part,results[arm,part])
            cards.append(card)
            chosen[arm+"/"+part]=ids
        sections.append(f'<section id="{escape(part)}"><h2>{index:02d}. {escape(LABELS[part])}</h2><div class="arms">'+"".join(cards)+'</div>'+vlm_panel(run,part,results["grconvnet_vlm",part])+'</section>')
    group_files=contact_sheets(run,output,results)
    nav=" · ".join(f'<a href="#{escape(part)}">{escape(LABELS[part])}</a>' for part in parts)
    groups=" · ".join(f'<a href="{escape(file)}">{escape(group.replace("_"," "))}</a>' for group,file in group_files.items())
    page=HTML.replace("__NAV__",nav).replace("__SECTIONS__","\n".join(sections)).replace("__GROUPS__",groups)
    (output/"index.html").write_text(page,encoding="utf-8")
    after={path:digest(run/path) for path in hashes}
    record={"part_count":14,"condition_count":3,"result_count":42,"source_result_sha256":hashes,
            "source_results_unchanged":hashes==after,"gallery_source_sha256":digest(Path(__file__)),
            "physics_steps":0,"model_calls":0,"final_assembly_views":False,
            "selected_candidate_ids":chosen,"group_images":group_files,
            "scope":"Human evidence gallery; ranking inputs and physical execution pictures are separately labelled"}
    if not record["source_results_unchanged"]:
        raise AssertionError("A source result changed during rendering")
    (output/"manifest.json").write_text(json.dumps(record,indent=2)+"\n")
    (output/"README.md").write_text("# All 14 parts: pickup comparison\n\n[Open the complete gallery](index.html).\n\nEach part shows three conditions, physical pickup frames when attempted, exact candidate images supplied to the VLM, its ranked IDs and unverified observations. Failed/unavailable cases remain included. No final assembly view is used. The gallery runs no model or physics. Decision and recorded-total time have the definitions in the experiment report.\n")
    print("COMPLETE="+str(output),flush=True)
    return output


HTML='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All 14 parts: GR-ConvNet, VLM and geometric pickups</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:#172b40;font:16px/1.5 system-ui,sans-serif}main{max-width:1580px;margin:auto;padding:28px}h1{font-size:34px;line-height:1.2;margin:0 0 14px}h2{font-size:28px;margin:0 0 16px}h3{font-size:21px;margin:0 0 10px}p{max-width:1200px}.intro,.legend,.nav{color:#526171}.nav{font-size:14px}a{color:#145c83}section{border-top:1px solid #c2cfda;margin-top:28px;padding-top:24px}.arms{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.arm,.vlm{border:1px solid #cad5df;background:white;border-radius:10px;padding:17px;min-width:0}.status{font-size:18px;font-weight:700;margin:6px 0}.passed{color:#207045}.failed{color:#8a442c}.reason{font-size:14px;overflow-wrap:anywhere}.selection,.no-motion{font-size:13px;color:#526171}dl{display:grid;grid-template-columns:1fr auto;gap:4px 12px;font-size:13px}dt{color:#526171}dd{margin:0;font-variant-numeric:tabular-nums}.vlm{margin-top:16px}.vlm figure{max-width:850px}figure{margin:12px 0}figure img{display:block;max-width:100%;height:auto;border:1px solid #d4dee7}figcaption{font-size:12px;color:#526171;margin-top:4px}details{margin-top:12px}summary{cursor:pointer;color:#145c83}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#eef3f7;padding:14px}.foot{font-size:13px;color:#526171}@media(max-width:950px){.arms{grid-template-columns:1fr}main{padding:16px}.arm figure{max-width:650px}}</style></head>
<body><main><h1>All 14 parts: pickup candidates and actual results</h1>
<p class="intro">Compare geometric search, pretrained GR-ConvNet ordered by its own score, and the same GR-ConvNet proposal method ordered by a local VLM. All conditions use the same saved starting scene and physical pickup test. No retraining or final assembly views are included.</p>
<p class="legend">A pickup passes after a lift of at least 120 mm and a one-second hold with both fingers still touching, with recorded robot/table/self/board overlap below 1 mm. One physical pickup attempt at most is allowed per part and condition. A numbered grasp drawing or passed movement check is not a demonstrated pickup.</p>
<p class="legend">Decision time covers capture, grasp generation, any VLM ranking and numerical movement checks. Recorded total time additionally includes setup, actual movement, pictures and saved records; one-time model setup is reported separately in the experiment report. Other loose parts remain visible with contacts disabled. Perfect target masks and simulated camera calibration are supplied; aligned CAD is supplied only to the geometric generator.</p>
<p><a href="../report.md">Full timing and pickup report</a> · <a href="../comparison.csv">Per-part comparison data</a> · <a href="manifest.json">Gallery source records</a></p><p>Actual-frame contact sheets: __GROUPS__</p>
<p class="nav">__NAV__</p>__SECTIONS__
<p class="foot">The VLM's written observations are retained as model output. Physical success is determined from the saved trajectory and contact forces, not from those observations. This human gallery does not call a model, step physics, or modify source results.</p></main></body></html>'''


if __name__=="__main__":
    build(sys.argv[1])
