"""Compare recorded geometric and pretrained/VLM pickup trials, without reruns."""
import csv
import json
from pathlib import Path
import statistics
import sys

from .audit import PARTS, PREVIOUS, read


ARMS = ("geometric", "grconvnet_score", "grconvnet_vlm")
LABELS = {
    "Gear_Large": "Large gear", "Gear_Medium": "Medium gear", "Gear_Small": "Small gear",
    "Waterproof_Male": "Waterproof connector", "M12_Hex_Nut": "M12 nut", "M16_Hex_Nut": "M16 nut",
    "RGOCG16-50_16mm": "16 mm round pin", "KET16_Square_16mm": "16 mm rectangular pin",
    "RGOCG12-50_12mm": "12 mm round pin", "RGOCG8-50_8mm": "8 mm round pin",
    "KET12_Square_12mm": "12 mm rectangular pin", "KET8_Square_8mm": "8 mm rectangular pin",
    "USB_Male": "USB connector", "DSUB_Male": "D-sub connector",
}
ARM_LABELS = {"geometric": "Geometric search", "grconvnet_score": "GR-ConvNet score order",
              "grconvnet_vlm": "GR-ConvNet + VLM order"}


def contact_description(row):
    """Name the contacting bodies and overlap, rather than saying collision."""
    names = row.get("bodies") or row.get("geoms")
    if not names:
        names = [row.get("body", "unnamed body")]
    overlap = row.get("penetration_m", 0.) * 1000
    return " / ".join(names) + f" overlap {overlap:.3f} mm"


def failed_path_description(check):
    phase = {"approach": "moving above the part", "descend": "lowering open fingers",
             "lift": "lifting the held part"}.get(check.get("phase"), check.get("phase", "unknown step"))
    status = check.get("status")
    if status == "ik_failed":
        return f"Robot angles not found while {phase}: {check.get('reason', 'solver gave no angles')}"
    if status == "target_obstruction":
        return f"Open hand would enter the part while {phase}: " + "; ".join(
            contact_description(row) for row in check.get("contacts", []))
    if status == "collision":
        return f"Robot would touch an obstacle while {phase}: " + "; ".join(
            contact_description(row) for row in check.get("contacts", []))
    return f"Stopped while {phase}: {status}"


def result_reason(record, folder=None):
    if record["status"] == "pickup_passed":
        return f"Lifted {record['lift_m']*1000:.1f} mm and held one second with both fingers touching."
    if record.get("error"):
        return f"Stopped during {record.get('last_executed_phase', 'setup')}: {record['error']}"
    if record["status"] == "grasp_failed":
        return "The fingers closed but their saved forces did not show both fingers holding the part."
    if record["status"] == "pickup_failed":
        causes = []
        if record.get("lift_m", 0.) < .12:
            causes.append(f"part rose only {record.get('lift_m', 0.)*1000:.1f} mm, below 120 mm")
        if not record.get("bilateral_after_lift_and_hold", False):
            causes.append("both fingers were not touching after the hold")
        if not record.get("robot_collision_penetration_below_1mm", False):
            causes.append("robot overlap with the table, itself or fixed hardware reached 1 mm")
        return "; ".join(causes) + "."
    if record["status"] == "vlm_output_unavailable":
        detail = "The local VLM did not return a complete valid order."
        if folder is not None and (Path(folder)/"vlm/ranking.json").exists():
            rank = read(Path(folder)/"vlm/ranking.json")
            detail += " " + str(rank.get("parse_error", rank.get("status", "unavailable")))
        return "No physical pickup attempted. " + detail
    failures = [check for item in record.get("planning_rejections", [])
                for check in item.get("checks", []) if check.get("status") != "passed"]
    if failures and all(check["status"] == "target_obstruction" and check["phase"] == "descend" for check in failures):
        maxima = [max(contact["penetration_m"] for contact in check["contacts"])*1000 for check in failures]
        return (f"No physical pickup attempted. All {len(failures)} candidate paths were rejected while lowering the open fingers: "
                f"the fingers would overlap the part by {min(maxima):.3f}–{max(maxima):.3f} mm per path "
                "(the approach check allows at most 0.050 mm).")
    paths = []
    for item in record.get("planning_rejections", []):
        for check in item.get("checks", []):
            if check.get("status") != "passed":
                paths.append(failed_path_description(check))
    if paths:
        return "No physical pickup attempted. " + "; ".join(dict.fromkeys(paths))
    diagnostic_path = None if folder is None else Path(folder).parents[1]/"static_failure_diagnostics.json"
    if diagnostic_path is not None and diagnostic_path.is_file():
        diagnostics = read(diagnostic_path)
        diagnostic = next((item for item in diagnostics.get("parts", [])
                           if item["part"] == Path(folder).name and item["arm"] == Path(folder).parent.name), None)
        if diagnostic is not None:
            proposal = diagnostic["proposals"][0]
            geometry = proposal["saved_depth_geometry_recomputed"]
            best = max(proposal["six_original_depth_checks"], key=lambda item:item["minimum_table_gap_m"])
            return ("No physical pickup attempted. The network's one proposal sampled a visible nut point "
                    f"{geometry['surface_center_world_m'][2]*1000:.6f} mm above the table. The host conversion halved this to "
                    f"a grasp-center height of {geometry['center_world_m'][2]*1000:.6f} mm. All six depths failed the table check; "
                    f"the best (+{best['finger_depth_m']*1000:.0f} mm) placed the lowest finger point "
                    f"{-best['minimum_table_gap_m']*1000:.6f} mm below the table, missing the "
                    f"{best['required_margin_m']*1000:.3f} mm clearance by {best['additional_height_for_margin_m']*1000:.6f} mm. "
                    "This explains this proposal and host height conversion; it does not show that GR-ConvNet cannot grasp the M16 nut.")
    mapping = record.get("proposal_mapping_rejections", [])
    if mapping:
        return "No physical pickup attempted. " + "; ".join(
            f"Network proposal {item['network_candidate_id']}: {item['reason']}" for item in mapping)
    if record.get("reason"):
        return str(record["reason"])
    return "No physical pickup attempted: " + record["status"].replace("_", " ") + "."


def summarize(run):
    """The runner's frozen schema is bound here once written; no inference."""
    run = Path(run).resolve()
    records = {(arm, part): read(run / arm / part / "result.json")
               for arm in ARMS for part in PARTS}
    rows = []
    for part in PARTS:
        row = {"part": part, "label": LABELS[part]}
        for arm in ARMS:
            record = records[arm, part]
            row.update({arm + "_" + key: value for key, value in {
                "status": record["status"], "pickup_passed": record["status"] == "pickup_passed",
                "physical_attempts": record["physical_attempts"], "lift_m": record.get("lift_m"),
                "decision_seconds": record["decision_seconds"],
                "setup_seconds": record["setup_seconds"], "total_seconds": record["total_seconds"],
                "reason": result_reason(record, run/arm/part),
                "candidate_count": record["shortlisted_count"],
                "path_candidates_checked": record["path_candidates_checked"],
                "vlm_status": record.get("vlm_status", "not_used"),
                "stage_seconds": record["stage_seconds"],
            }.items()})
        rows.append(row)
    summary = {}
    for arm in ARMS:
        summary[arm] = {
            "part_count": len(PARTS), "physical_attempts": sum(row[arm+"_physical_attempts"] for row in rows),
            "pickup_passed": sum(row[arm+"_pickup_passed"] for row in rows),
            "median_decision_seconds": statistics.median(row[arm+"_decision_seconds"] for row in rows),
            "total_decision_seconds": sum(row[arm+"_decision_seconds"] for row in rows),
            "median_total_seconds": statistics.median(row[arm+"_total_seconds"] for row in rows),
            "total_seconds": sum(row[arm+"_total_seconds"] for row in rows),
            "median_setup_seconds": statistics.median(row[arm+"_setup_seconds"] for row in rows),
            "mean_total_seconds_successful_pickups_only": statistics.mean(row[arm+"_total_seconds"] for row in rows if row[arm+"_pickup_passed"]),
            "median_total_seconds_successful_pickups_only": statistics.median(row[arm+"_total_seconds"] for row in rows if row[arm+"_pickup_passed"]),
            "total_stage_seconds": {key:sum(record["stage_seconds"].get(key,0.) for (method,_),record in records.items() if method == arm)
                for key in sorted({key for (method,_),record in records.items() if method == arm for key in record["stage_seconds"]})},
        }
    rank_rows = []
    for part in PARTS:
        record = records["grconvnet_vlm",part]
        path = run/"grconvnet_vlm"/part/"vlm/ranking.json"
        ranking = read(path) if path.is_file() else None
        score_record = records["grconvnet_score",part]
        rank_rows.append({"part":part,"status":record.get("vlm_status"),
            "candidate_count":record["shortlisted_count"],
            "ranked_pool_indices":record["candidate_check_order"],
            "changed_network_order":record["candidate_check_order"] != list(range(record["shortlisted_count"])),
            "score_selected_pool_index":score_record.get("selected_pool_index"),
            "vlm_selected_pool_index":record.get("selected_pool_index"),
            "paths_saved":score_record["path_candidates_checked"]-record["path_candidates_checked"],
            "VLM_fixed_score_failure":score_record["status"] != "pickup_passed" and record["status"] == "pickup_passed",
            "VLM_introduced_pickup_failure":score_record["status"] == "pickup_passed" and record["status"] != "pickup_passed",
            "api_model_load_seconds":None if ranking is None else ranking["raw_response"].get("load_duration",0)/1e9,
            "api_wall_seconds":None if ranking is None else ranking["api_wall_seconds"],
            "bridge_wall_seconds":None if ranking is None else ranking["bridge_wall_seconds"]})
    called = [row for row in rank_rows if row["api_wall_seconds"] is not None]
    # First call is retained in all primary totals; report warm calls separately
    # so the model's cold load is visible rather than silently subtracted.
    first_call = next(row for row in rank_rows if row["api_wall_seconds"] is not None)
    warm_calls = [row for row in called if row is not first_call]
    vlm_summary = {"actual_calls":len(called),
        "valid_rankings":sum(row["status"] == "ok" for row in rank_rows),
        "skipped_no_candidates":sum(row["status"] == "skipped_no_candidates" for row in rank_rows),
        "changed_network_order":sum(row["changed_network_order"] for row in rank_rows),
        "different_selected_poses":sum(row["score_selected_pool_index"] != row["vlm_selected_pool_index"] for row in rank_rows),
        "paths_saved":sum(row["paths_saved"] for row in rank_rows),
        "rescued_pickups":sum(row["VLM_fixed_score_failure"] for row in rank_rows),
        "introduced_failures":sum(row["VLM_introduced_pickup_failure"] for row in rank_rows),
        "model_load_time_already_included_in_calls_seconds":sum(row["api_model_load_seconds"] or 0 for row in rank_rows),
        "first_call":first_call,
        "warm_call_count":len(warm_calls),
        "median_warm_api_wall_seconds":statistics.median(row["api_wall_seconds"] for row in warm_calls),
        "median_warm_bridge_wall_seconds":statistics.median(row["bridge_wall_seconds"] for row in warm_calls),
        "parts":rank_rows}
    manifest = read(run/"manifest.json")
    comparison = {"parts": rows, "summary": summary,
                  "vlm_ranking":vlm_summary,
                  "static_failure_diagnostics":read(run/"static_failure_diagnostics.json") if (run/"static_failure_diagnostics.json").is_file() else None,
                  "model_setup_seconds_one_time":manifest["model_setup_seconds_one_time"],
                  "scope": "Same14part saved scene; one possible physical pickup per part and method; no assembly views or assembly movements.",
                  "historical_small_search": {"pickup_passed": 14, "parts": 14,
                      "median_search_seconds": .2655183744999903,
                      "median_setup_inclusive_seconds": 1.9627710245000003,
                      "source": str(PREVIOUS)}}
    (run / "comparison.json").write_text(json.dumps(comparison, indent=2)+"\n", encoding="utf-8")
    csv_rows = [{key:(json.dumps(value,sort_keys=True) if isinstance(value,dict) else value) for key,value in row.items()} for row in rows]
    with (run / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_report(run, comparison)
    return comparison


def write_report(run, comparison):
    rows, summary = comparison["parts"], comparison["summary"]
    text = ["# Pretrained GR-ConvNet, VLM grasp ordering and geometric pickup comparison", "",
        "All 14 parts are included for each method, even when the search finds no candidate, the VLM answer is unavailable, or no movement can be planned. At most one physical simulation pickup is attempted per part per method. No final assembly views or assembly movements are used.", "",
        "A successful pickup means at least 120 mm rise after a one-second hold, both fingers still touching, and robot overlap with itself, the table or fixed board hardware below 1 mm.", "",
        "[Per-part decision-time chart](decision_times.png). Large and medium gear approaches were rejected by numerical checks before any physical movement; their reported intersections are predicted open-finger/part overlap, not collisions that the robot actually made.", "",
        "| Method | Parts picked up and held | Physical attempts | Median decision time | Total decision time | Median total recorded time |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        item = summary[arm]
        text.append(f"| {ARM_LABELS[arm]} | {item['pickup_passed']}/14 | {item['physical_attempts']} | {item['median_decision_seconds']:.3f} s | {item['total_decision_seconds']:.3f} s | {item['median_total_seconds']:.3f} s |")
    text += ["", "## Every part", "", "Decision time includes that method's grasp generation, required image capture/rendering, VLM call when used, and robot movement checks. Scene/model setup and executing the contact simulation are separately recorded. Any reused proposal timing must be identified in the run manifest; it is not a second independently measured generation run.", "",
        "| Part | Geometric pickup | GR-ConvNet score pickup | GR-ConvNet + VLM pickup | Geometric decision | GR score decision | GR + VLM decision |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        statuses = ["Passed" if row[arm+"_pickup_passed"] else
                    ("Pickup failed" if row[arm+"_physical_attempts"] else "Not attempted") for arm in ARMS]
        times = [f"{row[arm+'_decision_seconds']:.3f} s" for arm in ARMS]
        text.append("| " + " | ".join([row["label"], *statuses, *times]) + " |")
    text += ["", "## What actually happened", ""]
    for arm in ARMS:
        text += ["### " + ARM_LABELS[arm], "", "| Part | Actual result and measured stopping reason |", "| --- | --- |"]
        text += [f"| {row['label']} | {row[arm+'_reason'].replace('|', '/')} |" for row in rows]
        text.append("")
    diagnostics = comparison.get("static_failure_diagnostics")
    if diagnostics:
        text += ["## M16 table-clearance diagnostic", "", "This post-run static check replays the saved proposal without moving the simulator, calling either model, changing the six allowed depths or modifying the frozen trial results. Both learned methods produced the same proposal and the same six failures. Positive gaps are above the table; negative gaps place the lowest finger point below it. The required gap was +0.100 mm.", "", "| Finger-depth offset | Lowest finger point above table | Additional height needed for required clearance |", "| ---: | ---: | ---: |"]
        for item in diagnostics["parts"][0]["proposals"][0]["six_original_depth_checks"]:
            text.append(f"| {item['finger_depth_m']*1000:+.0f} mm | {item['minimum_table_gap_m']*1000:+.6f} mm | {item['additional_height_for_margin_m']*1000:.6f} mm |")
        text += ["", "The network's predicted image opening was 60.094 mm. The center depth was sampled from a low visible nut point (2.447057 mm above the table), rather than the nut's highest visible surface (13.019252 mm). The host used half the sampled height (1.223528 mm) as the grasp center. The +8 mm variant therefore missed the required clearance by 0.125506 mm. This is a failure of this proposal and its image-to-gripper height conversion; it is not proof that the M16 nut has no usable GR-ConvNet grasp. [Saved static calculation](static_failure_diagnostics.json).", ""]
    text += ["## Time spent at each step", "", "Stage totals below include all 14 parts in each method. An absent stage was not used by that method. Total recorded time includes different numbers of attempted physical pickups when methods fail to find a candidate; it is not a time-to-success measure.", "",
             "| Stage | Geometric | GR-ConvNet score | GR-ConvNet + VLM |", "| --- | ---: | ---: | ---: |"]
    stage_keys = sorted({key for arm in ARMS for key in summary[arm]["total_stage_seconds"]})
    for key in stage_keys:
        values = [f"{summary[arm]['total_stage_seconds'][key]:.3f} s" if key in summary[arm]["total_stage_seconds"] else "Not used" for arm in ARMS]
        text.append("| "+key.replace("_"," ")+" | "+" | ".join(values)+" |")
    ranking = comparison["vlm_ranking"]
    text += ["", f"GR-ConvNet loading took **{comparison['model_setup_seconds_one_time']:.3f} s once**, before the part trials. That shared setup is reported separately rather than added repeatedly to every part. VLM calls include their model-loading time already; the sum of API-reported model-load durations was **{ranking['model_load_time_already_included_in_calls_seconds']:.3f} s**. These durations are not added a second time.", "",
             f"The first VLM call, on the large gear, included **{ranking['first_call']['api_model_load_seconds']:.3f} s** of model loading and took **{ranking['first_call']['bridge_wall_seconds']:.3f} s** through the local bridge. The remaining {ranking['warm_call_count']} calls had median API wall time **{ranking['median_warm_api_wall_seconds']:.3f} s** and median full bridge time **{ranking['median_warm_bridge_wall_seconds']:.3f} s**. API and bridge timers belong to different processes; original values are retained, not adjusted to match.", "",
             "All decision and total times are measured wall-clock times on this computer, including saved-image/evidence work. Physical execution here means advancing the contact simulation; it is not a measured physical robot cycle. All three methods capture fresh RGB-D; each learned method also runs fresh inference, so no cached generation time is presented as an independent measurement.", "",
             "| Method | Total recorded time, all 14 parts | Mean recorded time for successful pickups only | Median recorded time for successful pickups only |", "| --- | ---: | ---: | ---: |"]
    for arm in ARMS:
        item = summary[arm]
        text.append(f"| {ARM_LABELS[arm]} | {item['total_seconds']:.3f} s | {item['mean_total_seconds_successful_pickups_only']:.3f} s ({item['pickup_passed']} pickups) | {item['median_total_seconds_successful_pickups_only']:.3f} s |")
    text += ["", "## What the VLM changed", "",
             f"There were **{ranking['actual_calls']} local VLM calls**, **{ranking['valid_rankings']} valid orders**, and **{ranking['skipped_no_candidates']} part without a call because no pose survived the geometry checks**. The VLM changed network order on {ranking['changed_network_order']} parts, changed the selected pose on {ranking['different_selected_poses']} parts, and avoided {ranking['paths_saved']} rejected robot-path checks. Compared with the same network candidates in score order, it rescued {ranking['rescued_pickups']} pickups and introduced {ranking['introduced_failures']} pickup failures.", "",
             "The VLM orders the actual fixed pose associated with each pictured candidate. It does not choose unseen depth variants. Each network rectangle receives one deterministic jaw direction and the first table-clear depth in the common six-depth list before ranking. The network's proposed jaw-line endpoints are **not measured CAD surface contacts**.", "",
             "| Part | Candidate count | VLM status | Score-order pose used | VLM-order pose used | Robot-path checks saved | VLM API model load, included in call |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in ranking["parts"]:
        selected = ["None" if row[key] is None else str(row[key]+1) for key in ("score_selected_pool_index","vlm_selected_pool_index")]
        load = "No call" if row["api_model_load_seconds"] is None else f"{row['api_model_load_seconds']:.3f} s"
        text.append("| "+" | ".join([LABELS[row["part"]],str(row["candidate_count"]),str(row["status"]),*selected,str(row["paths_saved"]),load])+" |")
    text += ["## Inputs and limits", "",
        "The geometric method assumes correct CAD identity and initial alignment. The learned method instead receives RGB-D and a perfect visible target mask from the renderer, with known simulator camera calibration. Its grasp proposals do not require CAD alignment. This compares grasp pipelines under those declared input assumptions; it does not compare classification or pose-estimation accuracy. True target pose is still available to the common simulator for evaluating motion and held-target collision checks; it is excluded from the learned proposal generator.", "",
        "GR-ConvNet uses frozen pretrained weights without retraining. The VLM uses only current pickup candidates with neutral IDs; saved outcomes and final assembly views are excluded. Camera calibration and the image-to-gripper conversion are host adaptations. This is not an exact reproduction of OWG's reported robot setup.", "",
        "The same Panda controller, finger geometry, contact settings and permitted finger depths are used. Other loose parts stay visible, with collisions disabled; saved positions are restored every physics substep, with the existing one-step gravity offset. The table and fixed board hardware remain solid in physics. New snapshots and trajectories are saved for each method.", "",
        "A single scene and one possible attempt per part/method do not establish general performance on new poses, real hardware or assembly. The earlier smaller geometric run achieved14/14, but its published search time excluded RGB-D/VLM work and should not be substituted for the fresh timing comparison.", "",
        "[Per-part CSV](comparison.csv), [machine-readable comparison](comparison.json), [independent record audit](independent_audit.json), [run manifest](manifest.json).", ""]
    (run / "report.md").write_text("\n".join(text), encoding="utf-8")


def chart(run, comparison):
    """Static per-part measured decision times; run only after timed trials."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = comparison["parts"]
    positions = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(12.5, 9.8))
    colors = ("#246f8f", "#94a5b4", "#cb743f")
    labels = []
    for row in rows:
        suffix = " *" if not row["grconvnet_vlm_pickup_passed"] else ""
        labels.append(row["label"]+suffix)
    for offset, arm, color in zip((-.25, 0., .25), ARMS, colors):
        ax.barh([p+offset for p in positions], [row[arm+"_decision_seconds"] for row in rows],
                height=.22, label=ARM_LABELS[arm], color=color)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Decision time (wall-clock seconds; setup and pickup execution excluded)")
    ax.set_title("Same 14-part scene: grasp generation, VLM ordering and movement checks", loc="left", pad=20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#dfe6eb", linewidth=.6)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1., .90))
    first = rows[0]["grconvnet_vlm_decision_seconds"]
    ax.annotate("First local VLM call includes 9.45 s model loading", xy=(first,.25),
                xytext=(5.1,-.95), fontsize=9.5, color="#7e411f",
                arrowprops={"arrowstyle":"->","color":"#a55e35","linewidth":.8})
    fig.text(.015,.025,"* No GR-ConvNet pickup attempted: all candidate approaches failed numerical checks, or no pose cleared the table.",fontsize=9)
    fig.text(.015,.008,"All 14 parts retained for every method. Fresh RGB-D/inference; no retraining or assembly views. One observation per part/method.",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(Path(run)/"decision_times.png",dpi=180)
    fig.savefig(Path(run)/"decision_times.pdf")
    plt.close(fig)


if __name__ == "__main__":
    result = summarize(Path(sys.argv[1]))
    print(json.dumps(result["summary"], indent=2))
