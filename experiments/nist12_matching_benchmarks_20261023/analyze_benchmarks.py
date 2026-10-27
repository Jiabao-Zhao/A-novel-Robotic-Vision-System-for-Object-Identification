"""Evaluate frozen CAD-to-crop benchmark rankings; identities enter only here."""

import csv
import hashlib
import html
import json
import os
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = HERE.parent / "nist12_qwen_clip_reranking_20261023"
DEPTH = BASE / "depth_verification_20261023"
OUT = HERE / "analysis"
MAIN_METHODS = ["clip_depth_contact0", "cnos", "sam6d_rgb", "sam6d_rgbd", "nids_noadapter",
                "nids_ffa_only", "clip_depth_clearance1"]
MATCHED_METHODS = ["clip5_" + method for method in ["cnos", "sam6d_rgb", "sam6d_rgbd", "nids_noadapter", "nids_ffa_only"]]
METHODS = MAIN_METHODS + MATCHED_METHODS
NAMES = {"clip_depth_contact0": "CLIP + depth", "cnos": "CNOS", "sam6d_rgb": "SAM-6D RGB",
         "sam6d_rgbd": "SAM-6D RGB-D", "nids_noadapter": "NIDS no adapter",
         "nids_ffa_only": "NIDS FFA only (auxiliary)",
         "clip_depth_clearance1": "CLIP + depth, 1 mm clearance (sensitivity)"}
NAMES.update({method: NAMES[method.removeprefix("clip5_")] + " (same CLIP top five)" for method in MATCHED_METHODS})
PRIMARY = ["clip_depth_contact0", "cnos", "sam6d_rgb", "sam6d_rgbd", "nids_noadapter"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local(value):
    value = str(value).replace("\\", "/")
    if len(value) > 1 and value[1] == ":" and os.name != "nt":
        value = "/mnt/" + value[0].lower() + value[2:]
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def relative(path):
    try:
        return Path(path).relative_to(REPO).as_posix()
    except ValueError:
        return Path(path).as_posix()


def save(name, value):
    path = OUT / name
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        assert path.read_text(encoding="utf-8") == text, f"Existing derived evidence differs: {path}"
    else:
        path.write_text(text, encoding="utf-8")


def write_csv(name, rows):
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def family(target):
    if target.startswith("RGOCG"):
        return "round pin"
    if target.startswith("KET"):
        return "rectangular pin"
    return target


def product_name(target):
    if target.startswith("RGOCG"):
        return "Round pin " + target.split("_")[-1].replace("mm", " mm")
    if target.startswith("KET"):
        return "Rectangular pin " + target.split("_")[-1].replace("mm", " mm")
    return target.replace("_", " ")


def load_identity_and_baselines():
    """Read saved audit identities only after the new completion checks pass."""
    audit_file = DEPTH / "analysis/per_task.json"
    baseline_file = DEPTH / "selections.json"
    old_completion_file = DEPTH / "analysis/completion.json"
    old_completion = read(old_completion_file)
    assert digest(audit_file) == old_completion["artifact_sha256"]["per_task.json"]
    assert digest(baseline_file) == old_completion["selections_sha256"]
    protected = read(HERE / "protocol.json")["source_sha256"]
    assert digest(baseline_file) == protected[relative(baseline_file)]
    truth, inverse = {}, {}
    for row in read(audit_file):
        key = row["state"], row["target"]
        pid = row["truth_proposal_id"]
        assert key not in truth or truth[key] == pid
        assert (row["state"], pid) not in inverse or inverse[(row["state"], pid)] == row["target"]
        truth[key] = pid
        inverse[(row["state"], pid)] = row["target"]
    names = {"contact_0mm_geometry": "clip_depth_contact0", "clearance_1mm_geometry": "clip_depth_clearance1"}
    baselines = [dict(row, method=names[row["method"]]) for row in read(baseline_file) if row["method"] in names]
    assert len(baselines) == 192
    return truth, inverse, baselines, [audit_file, baseline_file, old_completion_file]


def image_path(record):
    """Native image path from the label-free common-input record."""
    return local(record["file"])


def outcome(row, truth):
    selected = row["selected_ids"]
    if row["status"] == "selected":
        return "correct" if selected == [truth] else "wrong"
    return "tied" if row["status"] == "tied" else "unavailable"


def paired_label(base, current):
    if base["outcome"] == "wrong" and current["outcome"] == "correct":
        return "fixed"
    if base["outcome"] == "correct" and current["outcome"] == "wrong":
        return "introduced_wrong"
    if base["outcome"] == "correct" and current["outcome"] not in ("correct", "wrong"):
        return "lost_correct_to_" + current["outcome"]
    if current["outcome"] == "wrong":
        return "unchanged_wrong" if base["selected_ids"] == current["selected_ids"] else "changed_wrong"
    if current["outcome"] in ("tied", "unavailable"):
        return current["outcome"]
    return "unchanged_correct"


def counts(rows):
    c = Counter(row["outcome"] for row in rows)
    return dict(tasks=len(rows), correct=c["correct"], wrong=c["wrong"], tied=c["tied"],
                unavailable=c["unavailable"], accuracy=c["correct"] / len(rows),
                wrong_same_family=sum(row["family_error"] == "same_family_wrong_size" for row in rows),
                wrong_other_family=sum(row["family_error"] == "wrong_family" for row in rows))


def figures(per_object, summary, targets):
    matrix = np.array([[next(row["correct"] for row in per_object
                            if row["target"] == target and row["method"] == method)
                       for method in MAIN_METHODS] for target in targets])
    short = ["CLIP + depth", "CNOS", "SAM-6D\nRGB", "SAM-6D\nRGB-D", "NIDS\nno adapter",
             "NIDS\nFFA only*", "CLIP + depth\n1 mm*"]
    with plt.rc_context({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}):
        fig, (bars, heat) = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [1, 1.4]})
        values = [summary[method]["correct"] for method in MAIN_METHODS]
        colors = ["#126A70", "#4C75A8", "#4C75A8", "#4C75A8", "#4C75A8", "#AEB8C4", "#AEB8C4"]
        bars.barh(range(len(MAIN_METHODS)), values, color=colors, height=.62)
        bars.set_yticks(range(len(MAIN_METHODS)), [name.replace("\n", " ") for name in short])
        bars.invert_yaxis()
        bars.set_xlim(0, 103)
        bars.set_xticks([0, 24, 48, 72, 96])
        bars.set_xlabel("Correct CAD-to-crop matches / 96")
        bars.set_title("Overall accuracy", fontweight="bold", pad=16)
        bars.grid(axis="x", alpha=.15)
        bars.set_axisbelow(True)
        for i, value in enumerate(values):
            bars.text(value + 1, i, str(value), va="center", fontweight="bold")
        heat.imshow(matrix, vmin=0, vmax=8, cmap="YlGnBu", aspect="auto")
        heat.set_yticks(range(len(targets)), [product_name(target) for target in targets])
        heat.set_xticks(range(len(MAIN_METHODS)), short, rotation=25, ha="right")
        heat.tick_params(axis="both", length=0)
        heat.set_title("Per object: correct matches / 8 scenes", fontweight="bold", pad=16)
        for (y, x), value in np.ndenumerate(matrix):
            heat.text(x, y, f"{value}/8", ha="center", va="center", color="white" if value >= 5 else "#102A40",
                      fontsize=12, fontweight="bold")
        fig.suptitle("Twelve objects in the same eight simulation scenes", fontsize=19, fontweight="bold", y=.98)
        fig.text(.02, .015, "CLIP + depth uses five retrieved crops; comparison methods rank all twelve. "
                 "* Auxiliary / simulator-clearance sensitivity. Controlled matching modules on common images.", fontsize=11)
        fig.subplots_adjust(left=.12, right=.985, top=.88, bottom=.19, wspace=.69)
        fig.savefig(OUT / "comparison.png", dpi=160, facecolor="white")
        plt.close(fig)


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + \
           "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table>"


def report(rows, per_object, summary, targets, records, candidates, truth, inverse):
    def href(path):
        return quote(os.path.relpath(path, OUT).replace("\\", "/"), safe="/:")

    crops = {(r["state"], r["proposal_id"]): r for r in records if r["kind"] == "crop"}
    cads = {(r["target"], r["view_index"]): r for r in records if r["kind"] == "cad"}
    reference_pool = read(HERE / "template_pool.json")["eligible_record_indices"]
    by_task = defaultdict(list)
    for row in rows:
        by_task[(row["state"], row["target"])].append(row)
    case_html, image_paths = [], set()

    def image_card(record, caption):
        path = image_path(record)
        assert path.is_file(), path
        image_paths.add(path)
        return f'<figure><a href="{href(path)}"><img loading="lazy" src="{href(path)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'

    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"

    for (state, target), task_rows in sorted(by_task.items()):
        relevant = [row for row in task_rows if row["method"] not in ("clip_depth_contact0", "clip_depth_clearance1")
                    and (row["outcome"] != "correct" or row["paired_change"] == "fixed")]
        if not relevant:
            continue
        body = [f'<section id="{html.escape(state + "__" + target)}"><h2>{html.escape(state + " — " + product_name(target))}</h2>']
        body.append(table(["Method", "Outcome", "Selected crop", "Selected product", "Compared with CLIP + depth"],
                          [[NAMES[r["method"]], r["outcome"], r["selected_ids"], r["predicted_product"], r["paired_change"]]
                           for r in task_rows]))
        true_pid = truth[(state, target)]
        body.append('<div class="images">' + image_card(crops[(state, true_pid)], f"Correct crop {true_pid}: {product_name(target)}") + '</div>')
        for row in relevant:
            method = row["method"]
            score_method = method.removeprefix("clip5_")
            body.append(f'<details><summary>{html.escape(NAMES[method] + ": " + row["paired_change"])}</summary>')
            body.append('<div class="images">')
            for pid in row["selected_ids"]:
                pair = candidates[(state, target, pid)]
                if score_method.startswith("nids"):
                    selected_ref = pair["nids"]["best_view_index"]
                    pool = reference_pool[target]
                    view = records[pool[selected_ref]]["view_index"]
                else:
                    view = pair["sam6d_best_CAD_view"]
                body.append(image_card(cads[(target, view)], f"Target CAD view {view}, chosen for crop {pid}"))
                body.append(image_card(crops[(state, pid)], f"Selected crop {pid}: {product_name(inverse[(state, pid)])}"))
            body.append('</div>')
            pairs = [candidates[(state, target, pid)] for pid in row["ranked_ids"]]
            body.append(table(["Crop", "Audit identity", "True crop?", NAMES[method], "CNOS semantic", "SAM appearance",
                               "SAM bbox IoU", "Patch coverage r", "NIDS FFA", "NIDS appearance"],
                [[pair["proposal_id"], product_name(inverse[(state, pair["proposal_id"])]), pair["proposal_id"] == true_pid,
                  number(pair["scores"][score_method]), number(pair["semantic"]["semantic_score"]),
                  number(pair["appearance"]["appearance_score"]), number(pair["geometry"].get("geometry_bbox_iou")),
                  number(pair["appearance"]["visible_ratio_diagnostic"]), number(pair["nids"].get("score")),
                  number((pair["nids_appearance"] or {}).get("score"))] for pair in pairs]))
            body.append('</details>')
        body.append('</section>')
        case_html.append("".join(body))
    intro = '<h1>CAD-to-crop matching: eight scenes, twelve objects</h1><p>Each request supplies one known CAD. ' \
            'Every comparison method independently ranks the same twelve scene crops. The saved CLIP + depth method ' \
            'selects among its frozen top five crops. No identity labels enter image encoding or scoring.</p>' \
            '<p>These are matching modules on shared 224-pixel white-background images and a shared CAD-view pool ' \
            '(20 views per product; 18 available USB views), rather than full detector/segmentation reproductions. ' \
            'The eight development scenes and CADs have been inspected previously. These results do not establish ' \
            'held-out, real-camera, or assembly performance.</p><img class="comparison" src="comparison.png">'
    overall = table(["Method", "Correct / 96", "Wrong", "Tied", "Unavailable", "Fixes", "Introduces wrong", "Wrong size", "Wrong family"],
        [[NAMES[m], f'{summary[m]["correct"]}/96', summary[m]["wrong"], summary[m]["tied"], summary[m]["unavailable"],
          summary[m]["fixed"], summary[m]["introduced_wrong"], summary[m]["wrong_same_family"], summary[m]["wrong_other_family"]] for m in MAIN_METHODS])
    matched = table(["Same frozen CLIP top-five crops", "Correct / 96", "Wrong", "Tied", "Unavailable", "Fixes", "Introduces wrong"],
        [[NAMES[m], f'{summary[m]["correct"]}/96', summary[m]["wrong"], summary[m]["tied"], summary[m]["unavailable"],
          summary[m]["fixed"], summary[m]["introduced_wrong"]] for m in ["clip_depth_contact0"] + MATCHED_METHODS])
    objects = table(["Object"] + [NAMES[m] for m in MAIN_METHODS],
        [[product_name(t)] + [next(f'{r["correct"]}/8' for r in per_object if r["method"] == m and r["target"] == t)
                              for m in MAIN_METHODS] for t in targets])
    links = '<p><a href="per_task.csv">All tasks CSV</a> · <a href="per_object.csv">Per object CSV</a> · ' \
            '<a href="paired_changes.csv">Fixed/introduced cases CSV</a> · <a href="confusions.json">Confusions</a> · ' \
            '<a href="../protocol.json">Frozen protocol</a> · <a href="../candidate_scores.json">All native scores</a></p>'
    style = 'body{font:16px system-ui;color:#173045;margin:32px auto;max-width:1600px;padding:0 24px}h1{font-size:30px}' \
            'table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}td,th{padding:9px 10px;border-bottom:1px solid #d4dce4;text-align:left}' \
            'th{background:#eaf1f5}section{margin:36px 0;border-top:2px solid #c9d5df;padding-top:12px}' \
            '.images{display:flex;gap:20px;flex-wrap:wrap}figure{width:320px;margin:12px 0}figure img{width:100%;height:230px;object-fit:contain;background:#f6f8fa}' \
            'figcaption{font-size:14px}.comparison{width:100%;height:auto}details{margin:15px 0;padding:12px;background:#f6f8fa}summary{cursor:pointer;font-weight:600}'
    document = f'<!doctype html><html><head><meta charset="utf-8"><title>CAD matching comparison</title><style>{style}</style></head><body>' + \
               intro + overall + links + '<h2>Matched top-five candidate budget</h2>' + matched + '<h2>Per object</h2>' + objects + '<h2>Every fixed or wrong case</h2>' + "".join(case_html) + '</body></html>'
    (OUT / "index.html").write_text(document, encoding="utf-8")
    return image_paths


def verify_report():
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paths = []

        def handle_starttag(self, tag, attrs):
            self.paths.extend(value for key, value in attrs if key in ("href", "src"))

    parser = Links()
    parser.feed((OUT / "index.html").read_text(encoding="utf-8"))
    for value in parser.paths:
        assert (OUT / unquote(value)).resolve().is_file(), value
    return dict(local_links_checked=len(parser.paths), all_local_links_exist=True)


def main():
    assert (HERE / "completion.json").is_file(), "Inference must finish before identities are read"
    completion = read(HERE / "completion.json")
    for name, key in [("candidate_scores.json", "scores_sha256"), ("selections.json", "selections_sha256"),
                      ("protocol.json", "protocol_sha256"), ("features.npz", "features_sha256")]:
        assert digest(HERE / name) == completion[key], name
    assert completion["evaluation_labels_read"] is False
    assert completion["source_hashes_verified"] is True
    protocol = read(HERE / "protocol.json")
    records = read(HERE / "input_records.json")
    decisions = read(HERE / "selections.json")
    candidates = {(r["state"], r["target"], r["proposal_id"]): r for r in read(HERE / "candidate_scores.json")}
    assert len(decisions) == 960 and len(candidates) == 1152 and len(records) == 336
    truth, inverse, baseline_rows, audit_sources = load_identity_and_baselines()
    decisions += baseline_rows
    assert len(truth) == 96 and len(inverse) == 96 and len(decisions) == 1152
    enriched = []
    for row in decisions:
        selected = row["selected_ids"]
        state, target = row["state"], row["target"]
        expected = truth[(state, target)]
        predicted = inverse[(state, selected[0])] if len(selected) == 1 else None
        status = outcome(row, expected)
        error = None if status != "wrong" else ("same_family_wrong_size" if family(target) == family(predicted) else "wrong_family")
        enriched.append(dict(method=row["method"], state=state, target=target, product=product_name(target),
            true_proposal_id=expected, status=row["status"], outcome=status, selected_ids=selected,
            predicted_product=predicted, family_error=error, top_score=row.get("top_score"),
            gap=row.get("gap"), ranked_ids=row.get("ranked_ids", [])))
    baseline = {(r["state"], r["target"]): r for r in enriched if r["method"] == "clip_depth_contact0"}
    for row in enriched:
        base = baseline[(row["state"], row["target"])]
        row["baseline_selected_ids"] = base["selected_ids"]
        row["baseline_outcome"] = base["outcome"]
        row["paired_change"] = paired_label(base, row)
    enriched.sort(key=lambda r: (METHODS.index(r["method"]), r["state"], protocol["targets"].index(r["target"])))
    summary, per_object, confusions = {}, [], {}
    for method in METHODS:
        selected = [r for r in enriched if r["method"] == method]
        assert len(selected) == 96
        changes = Counter(r["paired_change"] for r in selected)
        summary[method] = dict(name=NAMES[method], **counts(selected), fixed=changes["fixed"],
            introduced_wrong=changes["introduced_wrong"], changes=dict(changes),
            candidate_budget=5 if method.startswith(("clip_depth", "clip5_")) else 12,
            role="primary" if method in PRIMARY else "matched_top5_control" if method.startswith("clip5_") else "auxiliary" if method == "nids_ffa_only" else "sensitivity")
        summary[method]["pins_correct_out_of_48"] = sum(r["outcome"] == "correct" for r in selected if family(r["target"]) in ("round pin", "rectangular pin"))
        summary[method]["nonpins_correct_out_of_48"] = sum(r["outcome"] == "correct" for r in selected if family(r["target"]) not in ("round pin", "rectangular pin"))
        lost = changes["lost_correct_to_tied"] + changes["lost_correct_to_unavailable"]
        assert 94 + changes["fixed"] - changes["introduced_wrong"] - lost == summary[method]["correct"]
        assert sum(summary[method][key] for key in ("correct", "wrong", "tied", "unavailable")) == 96
        for target in protocol["targets"]:
            selected_target = [r for r in selected if r["target"] == target]
            assert len(selected_target) == 8
            per_object.append(dict(method=method, target=target, product=product_name(target), **counts(selected_target)))
        grouped = defaultdict(list)
        for row in selected:
            if row["outcome"] == "wrong":
                grouped[(row["target"], row["predicted_product"], row["family_error"])].append(row["state"])
        confusions[method] = [dict(target=t, selected_product=p, error_type=e, count=len(states), states=states)
                              for (t, p, e), states in sorted(grouped.items())]
    assert summary["clip_depth_contact0"]["correct"] == 94
    assert summary["clip_depth_clearance1"]["correct"] == 95
    OUT.mkdir(exist_ok=True)
    save("summary.json", dict(baseline="clip_depth_contact0", methods=summary,
        classification_unit="One supplied CAD -> one selected scene crop", tasks_per_method=96,
        same_scenes=True, scene_count=8, objects_per_scene=12,
        limitations=["Previously inspected development scenes, not independent held-out data.",
          "Matching modules on shared inputs, not full paper pipeline reproductions.",
          "Comparison methods rank twelve crops; saved CLIP + depth ranks five retrieved crops.",
          "One millimeter clearance is a separate simulator sensitivity, not the primary result.",
          "Depth is ideal existing single-camera simulation depth; no real sensor validation."]))
    save("per_task.json", enriched)
    save("per_object.json", per_object)
    save("paired_changes.json", enriched)
    save("changes_by_method.json", {method: {change: [row for row in enriched if row["method"] == method and row["paired_change"] == change]
            for change in sorted(set(row["paired_change"] for row in enriched if row["method"] == method))}
            for method in METHODS})
    save("matched_top5.json", [row for row in enriched if row["method"] in MATCHED_METHODS + ["clip_depth_contact0"]])
    save("confusions.json", confusions)
    write_csv("per_task.csv", enriched)
    write_csv("per_object.csv", per_object)
    write_csv("paired_changes.csv", enriched)
    write_csv("matched_top5.csv", [row for row in enriched if row["method"] in MATCHED_METHODS + ["clip_depth_contact0"]])
    figures(per_object, summary, protocol["targets"])
    image_paths = report(enriched, per_object, summary, protocol["targets"], records, candidates, truth, inverse)
    save("verification.json", dict(**verify_report(), new_inference_completion_verified_before_identity_read=True,
        task_rows=1152, candidate_pair_rows=1152, identities=96, per_object_rows=len(per_object),
        paired_accuracy_arithmetic_passed=True, method_task_count_checks_passed=True))
    source_paths = [HERE / name for name in ("completion.json", "protocol.json", "selections.json", "candidate_scores.json",
                    "input_records.json", "template_pool.json", "nids_source_audit.json", "sam6d_source_audit.json", "analyze_benchmarks.py")]
    source_paths += [HERE / name for name in ("encoding_completion.json", "execution_adjustment.json", "run_cached.py", "descriptor_statuses.json")
                     if (HERE / name).is_file()]
    source_paths += list(audit_sources) + sorted(image_paths)
    save("source_lock.json", {relative(p): digest(p) for p in sorted(set(source_paths))})
    for name, key in [("candidate_scores.json", "scores_sha256"), ("selections.json", "selections_sha256")]:
        assert digest(HERE / name) == completion[key]
    outputs = sorted(path for path in OUT.iterdir() if path.name != "completion.json" and path.is_file())
    save("completion.json", dict(tasks=1152, new_scores_frozen_before_audit=True,
        baseline_contact0_correct=94, baseline_clearance1_correct=95,
        source_files=len(set(source_paths)), outputs={p.name: digest(p) for p in outputs}))
    print(json.dumps({m: summary[m] for m in METHODS}, indent=2))


if __name__ == "__main__":
    main()
