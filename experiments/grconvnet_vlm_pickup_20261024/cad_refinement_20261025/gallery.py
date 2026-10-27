"""Build a pickup-only HTML gallery from saved trial pictures and records."""
import html
import json
from pathlib import Path
import sys

from .audit import ARMS, PARTS, read
from .report import ARM_LABELS, LABELS


def escape(value):
    return html.escape(str(value),quote=True)


def link(run,path):
    path = Path(path).resolve()
    return "../"+path.relative_to(run).as_posix() if path.is_relative_to(run) and path.is_file() else None


def picture(run,path,caption):
    target = link(run,path)
    if target is None:
        return ""
    return f'<figure><a href="{escape(target)}" target="_blank"><img src="{escape(target)}" loading="lazy" alt="{escape(caption)}"></a><figcaption>{escape(caption)}</figcaption></figure>'


def candidate_panel(run,folder,result):
    text = []
    image = folder/"rank_input.png"
    if image.is_file():
        text.append(picture(run,image,"Exact pickup-candidate image supplied to the VLM. Numbered poses were fixed before ranking."))
    mapping = folder/"display_id_map.json"
    if mapping.is_file():
        text.append(f'<p><a href="{escape(link(run,mapping))}">Exact displayed-ID to gripper-pose map</a></p>')
    if "vlm_status" in result:
        text.append(f'<p>Local VLM status: {escape(result["vlm_status"])}.</p>')
    ranked = result.get("vlm_ranked_display_ids")
    if ranked is not None:
        text.append('<p>VLM preference: '+escape(" → ".join(map(str,ranked)))+'.</p>')
    response = folder/"vlm/ranking.json"
    if response.is_file():
        ranking = read(response)
        text.append('<details><summary>Raw VLM response (model opinion, not a measured outcome)</summary><pre>'+escape(ranking.get("raw_text",""))+'</pre></details>')
    rejected = result.get("proposal_mapping_rejections",[])
    if rejected:
        text.append('<details><summary>Proposals rejected before robot movement checks</summary><pre>'+escape(json.dumps(rejected,indent=2))+'</pre></details>')
    return "".join(text)


def build(run):
    run = Path(run).resolve()
    comparison = read(run/"comparison.json")
    lookup = {(row["arm"],row["part"]):row for row in comparison["parts"]}
    output = run/"gallery"
    output.mkdir(exist_ok=True)
    text = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Three-part CAD-refined pickup trial</title><style>body{font:16px/1.5 system-ui,sans-serif;color:#183044;background:#edf2f5;margin:0}main{max-width:1550px;margin:auto;padding:28px}h1,h2,h3{line-height:1.2}section{margin:35px 0}nav{display:flex;gap:20px}a{color:#176586}article{background:white;border:1px solid #d7e1e8;border-radius:10px;padding:19px;min-width:0}.conditions{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px}figure{margin:15px 0}img{width:100%;height:auto;border:1px solid #dce4e9}figcaption,.note{font-size:13px;color:#526675}dl{display:grid;grid-template-columns:1fr 1fr;gap:5px}dd{margin:0;text-align:right}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:460px;overflow:auto}.pass{color:#1e714e}.fail{color:#9a4c24}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d7e1e8;padding:9px;text-align:left;background:white}</style><main>',
        '<h1>CAD-refined pickup: large gear, medium gear and M16 nut</h1>',
        '<p>Five fresh conditions on each of the three earlier learned-method failures. The original 14-part starting scene is reused. Actual frames below show closing and lifting only. There are no final assembly views.</p>',
        '<p><a href="../report.md">Detailed report</a> · <a href="../comparison.csv">All 15 conditions</a> · <a href="../independent_audit.json">Independent audit</a></p>',
        '<nav>'+" ".join(f'<a href="#{escape(part)}">{escape(LABELS[part])}</a>' for part in PARTS)+'</nav>',
        '<p class="note">A drawn candidate is a proposed pickup. An actual attempt is shown only when the contact simulation moved. Other loose parts remain visible with contacts disabled; the table and fixed board hardware remain solid. Simulator part poses are used for audit measurements, not motion feedback.</p>']
    for part in PARTS:
        text += [f'<section id="{escape(part)}"><h2>{escape(LABELS[part])}</h2><div class="conditions">']
        for arm in ARMS:
            row = lookup[arm,part]
            folder = run/arm/part
            result = read(folder/"result.json")
            passed = row["pickup_passed"]
            text += [f'<article><h3>{escape(ARM_LABELS[arm])}</h3>',
                     f'<p class="{"pass" if passed else "fail"}"><strong>{"Pickup passed" if passed else "Pickup did not pass"}</strong></p>',
                     f'<p>{escape(row["reason"])}</p>', '<dl>',
                     f'<dt>Physical attempts</dt><dd>{row["physical_attempts"]}</dd>',
                     f'<dt>Candidate count</dt><dd>{row["candidate_count"]}</dd>',
                     f'<dt>Candidate paths checked</dt><dd>{row["paths_checked"]}</dd>',
                     f'<dt>Decision time</dt><dd>{row["decision_seconds"]:.3f} s</dd>',
                     f'<dt>Total recorded time</dt><dd>{row["total_seconds"]:.3f} s</dd></dl>']
            selected = result.get("selected_pool_index")
            text.append('<p>Selected pool entry: '+("None" if selected is None else str(selected+1))+'.</p>')
            if selected is not None and "network_candidate_id" in result["candidates"][selected]:
                text.append('<p>Original network proposal: '+str(result["candidates"][selected]["network_candidate_id"])+'.</p>')
            if row["physical_attempts"]:
                closeup = output/"closeups"/part
                has_closeup = arm.startswith("cad_") and (closeup/"close.png").is_file() and (closeup/"hold.png").is_file()
                if has_closeup:
                    text.append(picture(run,closeup/"close.png","Post-run closeup rendered from the actual saved state after closing; no additional physics steps"))
                    text.append(picture(run,closeup/"hold.png","Post-run closeup rendered from the actual saved state after lift and hold; no additional physics steps"))
                    text.append('<p class="note">These closeups use CAD score-order states. The independent audit verifies that the CAD + VLM trajectory and recorded contacts are identical. Camera changes are for this post-run illustration only.</p><details><summary>Original wide evidence frames</summary>')
                text.append(picture(run,folder/"closed.png","Original recorded simulation frame after the fingers closed"))
                text.append(picture(run,folder/"lifted_all_parts_visible.png","Original recorded simulation frame after lifting and the one-second hold"))
                if has_closeup:
                    text.append('</details>')
            else:
                text.append('<p>No physical pickup was attempted.</p>')
            changes = row["relative_changes"]
            if changes:
                text.append('<dl>')
                for key,label in (("during_closing","Shift while closing"),("closed_to_end_hold","Shift after closure"),("maximum_after_closure","Largest sampled post-closure shift")):
                    if key in changes:
                        value = changes[key]
                        text.append(f'<dt>{escape(label)}</dt><dd>{value["cad_origin_displacement_m"]*1000:.3f} mm / {value["rotation_deg"]:.3f}°</dd>')
                text.append('</dl>')
            text.append(f'<p><a href="{escape(link(run,folder/"independent_grip_relative.json"))}">Actual part-in-gripper transforms</a> · <a href="{escape(link(run,folder/"result.json"))}">Full condition result</a></p>')
            text.append(candidate_panel(run,folder,result))
            text.append('</article>')
        text.append('</div></section>')
    text += ['<section><h2>Part pose relative to the gripper</h2><p><code>grip_T_cad = inverse(world_T_grip) @ world_T_cad</code>. The gripper frame is the Panda grip site, and lengths are meters. The planned relation uses the commanded grasp. Recorded relations after closing and after lifting use the actual gripper and part poses. A change after closure measures the part shifting inside the gripper.</p><p>One possible attempt per condition in this reused scene does not establish performance on new poses or physical hardware.</p></section></main></html>']
    destination = output/"index.html"
    destination.write_text("\n".join(text),encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(build(Path(sys.argv[1])))
