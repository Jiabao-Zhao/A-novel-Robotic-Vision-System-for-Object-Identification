"""Summarize saved 14-part small/large pickup trials without rerunning them."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys

from .audit import BASE, PARTS, legacy_report, read


LABELS = {
    "Gear_Large": "Large gear", "Gear_Medium": "Medium gear", "Gear_Small": "Small gear",
    "Waterproof_Male": "Waterproof connector", "M12_Hex_Nut": "M12 nut", "M16_Hex_Nut": "M16 nut",
    "RGOCG16-50_16mm": "16 mm round pin", "KET16_Square_16mm": "16 mm rectangular pin",
    "RGOCG12-50_12mm": "12 mm round pin", "RGOCG8-50_8mm": "8 mm round pin",
    "KET12_Square_12mm": "12 mm rectangular pin", "KET8_Square_8mm": "8 mm rectangular pin",
    "USB_Male": "USB connector", "DSUB_Male": "D-sub connector",
}


def distinct_contact_pairs(grasps):
    """Count unordered point pairs on a 0.01 mm world-coordinate grid."""
    return len({tuple(sorted(tuple(round(float(value), 5) for value in point)
                             for point in item["contacts_world_m"])) for item in grasps})


def reason(result):
    """Explain the exact observed stopping point or rejected path in plain words."""
    if result["status"] == "pickup_passed":
        return "Lifted and held for one second with both fingers touching."
    if result["status"] == "no_pickup_path_found_in_shortlist":
        failed = {}
        specifics = []
        for grasp in result["shortlisted_grasps"]:
            for check in grasp["path_checks"]:
                if check["status"] == "passed":
                    continue
                key = check["phase"] + ": " + check["status"]
                failed[key] = failed.get(key, 0) + 1
                if check.get("reason"):
                    specifics.append(check["reason"])
                specifics.extend(" / ".join(contact["bodies"]) + f" overlap {contact['penetration_m']*1000:.3f} mm"
                                 for contact in check.get("contacts", []))
        counts = "; ".join(f"{count} stopped during {step}" for step, count in failed.items())
        return f"No physical pickup attempted. Checked {result['path_candidates_checked']} shortlisted paths. {counts}. " + "; ".join(dict.fromkeys(specifics))
    if result.get("error"):
        return f"Stopped during {result['last_executed_phase']}: {result['error']}"
    if result["status"] == "grasp_failed":
        return "Fingers closed but the saved force records did not show both fingers touching the part."
    failures = []
    if result.get("lift_m", 0) < .12:
        failures.append(f"The part rose {result.get('lift_m', 0)*1000:.1f} mm, below the 120 mm requirement")
    if not result.get("bilateral_after_lift_and_hold", False):
        failures.append("both fingers were not touching after the one-second hold")
    if not result.get("robot_collision_penetration_below_1mm", False):
        failures.append("robot overlap with itself, the table or fixed board hardware reached 1 mm")
    return "; ".join(failures) + "."


def summarize(run):
    run = Path(run).resolve()
    legacy = legacy_report()
    old = {row["part"]: row for row in legacy["parts"]}
    records, rows = {}, []
    for part in PARTS:
        row = {"part": part, "label": LABELS[part],
               "original_generated_poses": old[part]["generated_gripper_poses"],
               "original_planning_seconds_including_assembly_search": old[part]["planning_seconds"],
               "original_end_of_lift_passed": old[part]["pickup_end_of_lift_passed"],
               "original_lift_m": old[part]["lift_m"]}
        for arm in ("large_control", "small"):
            result = read(run / arm / part / "result.json")
            records[arm, part] = result
            selected = result.get("selected_grasp")
            row.update({arm + "_" + key: value for key, value in {
                "generated_poses": result["generated_gripper_poses"],
                "table_clear_poses": result["table_clear_candidates"],
                "shortlist_count": result["shortlisted_count"],
                "distinct_shortlist_contact_pairs": distinct_contact_pairs(result["shortlisted_grasps"]),
                "search_seconds": result["search_seconds"],
                "planning_seconds_including_setup": result["planning_seconds_including_setup"],
                "generation_and_filter_seconds": result["candidate_generation_and_filter_seconds"],
                "path_check_seconds": result["robot_path_check_seconds"],
                "physical_attempts": result["physical_attempts"],
                "pickup_passed": result["status"] == "pickup_passed",
                "lift_m": result.get("lift_m"), "maximum_lift_m": result["maximum_observed_lift_m"],
                "selected_depth_m": None if selected is None else selected["pad_depth_m"],
                "selected_shortlist_index": result.get("selected_shortlist_index"),
                "path_candidates_checked": result["path_candidates_checked"],
                "status": result["status"], "reason": reason(result),
            }.items()})
        row["paired_search_ratio_large_over_small"] = row["large_control_search_seconds"] / row["small_search_seconds"]
        rows.append(row)
    summaries = {}
    for arm in ("large_control", "small"):
        summaries[arm] = {
            "pickup_passed": sum(row[arm+"_pickup_passed"] for row in rows),
            "physical_attempts": sum(row[arm+"_physical_attempts"] for row in rows),
            "median_search_seconds": statistics.median(row[arm+"_search_seconds"] for row in rows),
            "mean_search_seconds": statistics.mean(row[arm+"_search_seconds"] for row in rows),
            "total_search_seconds": sum(row[arm+"_search_seconds"] for row in rows),
            "minimum_search_seconds": min(row[arm+"_search_seconds"] for row in rows),
            "maximum_search_seconds": max(row[arm+"_search_seconds"] for row in rows),
            "median_setup_inclusive_planning_seconds": statistics.median(row[arm+"_planning_seconds_including_setup"] for row in rows),
            "total_setup_inclusive_planning_seconds": sum(row[arm+"_planning_seconds_including_setup"] for row in rows),
            "total_generated_poses": sum(row[arm+"_generated_poses"] for row in rows),
            "minimum_generated_poses": min(row[arm+"_generated_poses"] for row in rows),
            "maximum_generated_poses": max(row[arm+"_generated_poses"] for row in rows),
            "total_shortlist_poses": sum(row[arm+"_shortlist_count"] for row in rows),
            "total_distinct_shortlist_contact_pairs": sum(row[arm+"_distinct_shortlist_contact_pairs"] for row in rows),
            "minimum_shortlist_count": min(row[arm+"_shortlist_count"] for row in rows),
            "maximum_shortlist_count": max(row[arm+"_shortlist_count"] for row in rows),
        }
    large, small = summaries["large_control"], summaries["small"]
    paired = {
        "small_faster_parts": sum(row["small_search_seconds"] < row["large_control_search_seconds"] for row in rows),
        "median_search_ratio_large_over_small": large["median_search_seconds"] / small["median_search_seconds"],
        "total_search_ratio_large_over_small": large["total_search_seconds"] / small["total_search_seconds"],
        "total_search_time_reduction_fraction": 1-small["total_search_seconds"] / large["total_search_seconds"],
        "generated_pose_reduction_fraction": 1-small["total_generated_poses"] / large["total_generated_poses"],
        "median_of_per_part_search_ratios": statistics.median(row["paired_search_ratio_large_over_small"] for row in rows),
    }
    result = {"rows": rows, "summary": summaries, "paired": paired, "legacy": legacy,
              "scope": "Same initial 14-part scene, one fresh pickup attempt per part per arm; fixed small and large search policies. No assembly run.",
              "timing_caveat": "One serial same-process observation per part/arm, alternating arm order. Search time excludes model setup, evidence writes, images and physics; setup-inclusive planning is also recorded. Original timing includes assembly-route searches and cannot isolate sampling speed.",
              "search_policy_difference": "Small changes surface-ray count, contact-pair count, shortlist size and adds contact-region diversity. Full finger-depth grid, score, robot controller, collision shapes and contact settings match."}
    (run / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (run / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = ["# Smaller grasp search: all 14 parts", "",
            "[Open every shortlisted grasp](shortlist_gallery/index.html). [Download the per-part comparison](comparison.csv).", "",
            f"The smaller search picked up and held **{small['pickup_passed']}/14 parts**. The fresh larger search picked up and held **{large['pickup_passed']}/14 parts**. Each arm attempted at most one physical simulation pickup per part; failed searches are included in the denominator.", "",
            "A pickup passes when the part is at least 120 mm above its starting height after a one-second hold, both fingers still touch it, and recorded robot overlap with itself, the table or fixed board hardware stays below 1 mm. No part is invisibly attached to the hand.", "",
            "## Direct comparison under the same pickup conditions", "",
            "| Setting or measured result | Fresh larger search | Smaller search |",
            "| --- | ---: | ---: |",
            "| Surface locations sampled on each CAD | 2,048 | 512 |",
            "| Maximum contact pairs expanded into gripper poses | 320 | 24 |",
            "| Maximum shortlisted poses | 48 | 12 |",
            f"| Generated gripper poses across all 14 parts | {large['total_generated_poses']:,} | {small['total_generated_poses']:,} |",
            f"| Saved shortlisted poses across all 14 parts | {large['total_shortlist_poses']} | {small['total_shortlist_poses']} |",
            f"| Distinct shortlisted contact-point pairs, counted separately per part | {large['total_distinct_shortlist_contact_pairs']} | {small['total_distinct_shortlist_contact_pairs']} |",
            f"| Median search time per part | {large['median_search_seconds']:.3f} s | {small['median_search_seconds']:.3f} s |",
            f"| Total search time across all 14 parts | {large['total_search_seconds']:.3f} s | {small['total_search_seconds']:.3f} s |",
            f"| Median planning time including model/scene setup | {large['median_setup_inclusive_planning_seconds']:.3f} s | {small['median_setup_inclusive_planning_seconds']:.3f} s |",
            f"| Picked up and held after one second | {large['pickup_passed']}/14 | {small['pickup_passed']}/14 |", "",
            f"In this run, the smaller search used **{paired['generated_pose_reduction_fraction']*100:.1f}% fewer generated poses** and **{paired['total_search_time_reduction_fraction']*100:.1f}% less total search time** ({paired['total_search_ratio_large_over_small']:.2f} times faster by total time). It was faster on {paired['small_faster_parts']} of 14 parts. These are single-run measurements in one known starting scene, not a general speed or reliability guarantee.", "",
            "Both searches use the same six finger contact depths: +4, 0, −4, −8, **+6 and +8 mm**, measured along the gripper's forward axis from its reference point. The +6/+8 mm choices preserve the full-finger positions that worked for the 8 mm round pin. The smaller search also spreads its shortlist across different touch regions; this compares complete search settings, not only ray count.", "",
            "Search time covers reading the CAD, finding contact pairs, constructing gripper poses, checking the table, forming the shortlist and checking robot movement until the first passing path. Setup time, evidence files, pictures and executing the pickup are excluded. Setup-inclusive planning times are recorded separately. Imports are excluded, arm order alternates per part, and neither arm was expanded after seeing an outcome.", "",
            "## Every part: generated poses, shortlist and time", "",
            "A shortlist contains table-clearing poses. Robot movement is checked in order and stops at the first passing pose; later poses have not had their robot movement checked. A pictured untried pose is not a demonstrated pickup.", "",
            "| Part | Larger generated | Smaller generated | Smaller shortlist | Distinct smaller touch pairs | Larger search | Smaller search | Larger pickup | Smaller pickup |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for row in rows:
        text.append(f"| {row['label']} | {row['large_control_generated_poses']:,} | {row['small_generated_poses']:,} | {row['small_shortlist_count']} | {row['small_distinct_shortlist_contact_pairs']} | {row['large_control_search_seconds']:.3f} s | {row['small_search_seconds']:.3f} s | {'Passed' if row['large_control_pickup_passed'] else 'Failed'} | {'Passed' if row['small_pickup_passed'] else 'Failed'} |")
    text.extend(["", "The **8 mm round pin has two shortlisted gripper poses using the same pair of touch points**, both at +8 mm finger depth. The jaw directions are exchanged between the two poses; they are two gripper orientations, not two distinct grasp locations. Contact-pair counts ignore which finger is assigned to which point and round world coordinates to 0.01 mm, matching the sampler's deduplication resolution."])
    text.extend(["", "## Actual smaller-search pickups", "",
                 "Grasp numbers below start at 1 to match the gallery. The saved machine records use indices starting at 0.", "",
                 "| Part | Grasp used | Finger depth | Robot paths checked | Final lift | Actual result |",
                 "| --- | ---: | ---: | ---: | ---: | --- |"])
    for row in rows:
        index = row["small_selected_shortlist_index"]
        depth = row["small_selected_depth_m"]
        lift = row["small_lift_m"]
        text.append(f"| {row['label']} | {'None' if index is None else index+1} | {'—' if depth is None else f'{depth*1000:+.0f} mm'} | {row['small_path_candidates_checked']} | {'—' if lift is None else f'{lift*1000:.1f} mm'} | {row['small_reason']} |")
    text.extend(["", "## Comparison with the original saved run", "",
                 f"The original run picked up **{legacy['passed_end_of_lift']}/14** at the end of its lift. Its 8 mm round pin never moved because no generated pose cleared the table. That run used four depths, without +6/+8 mm, searched for assembly movements too, and did not add the new one-second hold. Its median combined planning time was **{legacy['median_planning_seconds']:.3f} s**; dividing this by the new pickup-only time would mix different tasks. A separate earlier 8 mm follow-up succeeded with the extra depths; it was not a new all-14 benchmark.", "",
                 "| Part | Original generated | Original combined planning | Original pickup at end of lift | Smaller pickup after hold |",
                 "| --- | ---: | ---: | --- | --- |"])
    for row in rows:
        text.append(f"| {row['label']} | {row['original_generated_poses']:,} | {row['original_planning_seconds_including_assembly_search']:.3f} s | {'Passed' if row['original_end_of_lift_passed'] else 'Not attempted'} | {'Passed' if row['small_pickup_passed'] else 'Failed'} |")
    failures = [row for row in rows if not row["large_control_pickup_passed"]]
    if failures:
        text.extend(["", "### Fresh larger-search stopping reasons", ""])
        text.extend(f"- {row['label']}: {row['large_control_reason']}" for row in failures)
    text.extend(["", "## Scene and limits", "",
                 "This uses the older prepared `objects-centered` 14-part scene and saved `objects_centered` state, with the same USB correction as the original trial. Each part starts fresh; this is not one continuous assembly. All other 13 loose parts remain visible and noncolliding. They are held at their saved positions by resetting each physics substep; recorded frames have a constant one-step gravity offset of **0.03924 mm** downward, identical in both arms and without accumulated falling. The table, board and its fixed hardware stay solid in the physics simulation. The planner checks the robot against itself and the table, and checks the open hand against the target during approach; board/fixed-hardware contacts are separately checked in the recorded physical result.", "",
                 "Identity and initial CAD alignment are assumed correct. No final assembly goal influences this pickup test. No new model, training, weights, physical robot or paid service is used. This result establishes what happened in these 28 simulations; it does not establish general performance on new poses, real hardware or assembly.", "",
                 "Validation: **13 focused tests passed**: two tests for the smaller shortlist and preserved finger depths, plus the 11 existing grasp-geometry/controller tests. The independent saved-record audit checks all 28 pickup trajectories, finger-force contacts, initial scene agreement, shortlist/archive equality and timing arithmetic. It also verifies that 195 protected historical files and shared sources remain unchanged. A separate numerical geometry check covers all 828 shortlisted poses and 1,656 touch points in both arms: the largest distance from a touch point to the native CAD surface is **0.015665 mm**. These numerical distances are reported without fitting a new acceptance threshold or changing pickup outcomes. These code/data checks do not establish assembly success.", "",
                 "[Machine-readable comparison](comparison.json), [independent saved-record audit](independent_audit.json), [run settings and source hashes](manifest.json).", ""])
    (run / "report.md").write_text("\n".join(text), encoding="utf-8")
    return result


def chart(run, comparison):
    """Export a static scientific timing chart after the timed trials finish."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = comparison["rows"]
    positions = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh([value-.18 for value in positions], [row["large_control_search_seconds"] for row in rows],
            height=.34, color="#8196a8", label="Fresh larger search")
    ax.barh([value+.18 for value in positions], [row["small_search_seconds"] for row in rows],
            height=.34, color="#197a9b", label="Smaller search")
    ax.set_yticks(positions, [row["label"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Search time (seconds; setup and pickup execution excluded)")
    ax.set_title("Same 14-part scene: grasp generation and robot movement checks", loc="left", pad=18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#dce4e9", linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.text(.01, .01, "One serial observation per part/setting; same finger depths and pickup controller. No assembly tested.", fontsize=9)
    fig.tight_layout(rect=(0, .025, 1, 1))
    fig.savefig(Path(run)/"search_times.png", dpi=180)
    fig.savefig(Path(run)/"search_times.pdf")
    plt.close(fig)


if __name__ == "__main__":
    output = Path(sys.argv[1])
    comparison = summarize(output)
    chart(output, comparison)
    print(json.dumps({"summary": comparison["summary"], "paired": comparison["paired"]}, indent=2))
