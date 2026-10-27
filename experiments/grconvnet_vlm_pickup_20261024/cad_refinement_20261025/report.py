"""Plain-language comparison of the saved three-part CAD-refinement trial."""
import csv
import json
from pathlib import Path
import statistics
import sys

from experiments.grconvnet_vlm_pickup_20261024.report import result_reason
from .audit import ARMS, OLD_RUN, PARTS, read, relative_trace, save


LABELS = {"Gear_Large":"Large gear", "Gear_Medium":"Medium gear", "M16_Hex_Nut":"M16 nut"}
ARM_LABELS = {"geometric":"Geometric search", "grconvnet_score":"GR-ConvNet score order",
              "grconvnet_vlm":"GR-ConvNet + VLM order", "cad_grconvnet_score":"CAD-refined GR-ConvNet score order",
              "cad_grconvnet_vlm":"CAD-refined GR-ConvNet + VLM order"}


def summarize(run):
    run = Path(run).resolve()
    rows, held_poses, vlm_calls, CAD_corrections = [], [], [], []
    for part in PARTS:
        for arm in ARMS:
            folder = run/arm/part
            result = read(folder/"result.json")
            relative = relative_trace(folder, result)
            save(folder/"independent_grip_relative.json",relative)
            row = {"part":part,"arm":arm,"status":result["status"],
                   "physical_attempts":result["physical_attempts"],"pickup_passed":result["status"] == "pickup_passed",
                   "candidate_count":result["shortlisted_count"],"paths_checked":result["path_candidates_checked"],
                   "decision_seconds":result["decision_seconds"],"setup_seconds":result["setup_seconds"],
                   "total_seconds":result["total_seconds"],"stage_seconds":result["stage_seconds"],
                   "lift_m":result.get("lift_m"),"selected_pool_index":result.get("selected_pool_index"),
                   "reason":result_reason(result,folder),"relative_changes":relative["changes"]}
            rows.append(row)
            if arm == "cad_grconvnet_score":
                table_z = result["cad_refinement"]["table_z_world_m"]
                for candidate in result["candidates"]:
                    CAD_corrections.append({"part":part,"network_candidate_id":candidate["network_candidate_id"],
                        "XY_contact_center_change_m":candidate["CAD_correction"]["xy_shift_m"],
                        "original_contact_center_above_table_m":candidate["before_depth_geometry"]["center_world_m"][2]-table_z,
                        "CAD_contact_center_above_table_m":candidate["depth_geometry"]["center_world_m"][2]-table_z,
                        "original_predicted_span_m":candidate["before_depth_geometry"]["predicted_width_m"],
                        "CAD_contact_span_m":candidate["width_m"],"fixed_finger_opening_m":candidate["opening_m"],
                        "fixed_finger_depth_m":candidate["pad_depth_m"],"table_gap_m":candidate["table_clearance_m"]})
            ranking_path = folder/"vlm/ranking.json"
            if arm.endswith("_vlm") and ranking_path.is_file():
                ranking = read(ranking_path)
                vlm_calls.append({"part":part,"arm":arm,"status":ranking["status"],
                    "api_wall_seconds":ranking["api_wall_seconds"],"bridge_wall_seconds":ranking["bridge_wall_seconds"],
                    "API_reported_load_seconds_already_in_call":ranking["raw_response"].get("load_duration",0)/1e9,
                    "candidate_count":result["shortlisted_count"],
                    "order_changed":result["candidate_check_order"] != list(range(result["shortlisted_count"]))})
            if arm.startswith("cad_") and row["pickup_passed"]:
                held_poses.append({"part":part,"arm":arm,
                    "frame":"CAD origin and axes relative to gripper0_grip_site",
                    "after_closing":relative["moments"]["close"],
                    "after_lift_and_hold":relative["moments"]["hold_lift"],
                    "changes":relative["changes"]})
    summary = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        summary[arm] = {"part_count":len(PARTS),"pickup_passed":sum(row["pickup_passed"] for row in selected),
            "physical_attempts":sum(row["physical_attempts"] for row in selected),
            "total_decision_seconds":sum(row["decision_seconds"] for row in selected),
            "median_decision_seconds":statistics.median(row["decision_seconds"] for row in selected),
            "total_recorded_seconds":sum(row["total_seconds"] for row in selected),
            "median_total_recorded_seconds":statistics.median(row["total_seconds"] for row in selected),
            "total_stage_seconds":{key:sum(row["stage_seconds"].get(key,0.) for row in selected)
                for key in sorted({key for row in selected for key in row["stage_seconds"]})}}
    value = {"parts":rows,"summary":summary,"held_part_poses":held_poses,"vlm_calls":vlm_calls,
             "CAD_proposal_corrections":CAD_corrections,
             "manifest":read(run/"manifest.json"),
             "scope":"Three previously unattempted GR-ConvNet parts in the same 14-part starting scene; five fresh conditions; no assembly views or assembly movement.",
             "previous_14part_run":str(OLD_RUN)}
    value["timing_exception"] = {"intended_common_warm_model_condition_met":not any(item["API_reported_load_seconds_already_in_call"] > 1. for item in vlm_calls),
        "calls_with_API_reported_load_over_one_second":[item for item in vlm_calls if item["API_reported_load_seconds_already_in_call"] > 1.],
        "policy":"Retain measured timers unchanged. Common preload and later timed-call reload are different actual costs; never add a call's load duration again."}
    save(run/"comparison.json",value)
    save(run/"held_part_poses.json",held_poses)
    held_flat = []
    for row in held_poses:
        for key in ("after_closing","after_lift_and_hold"):
            moment = row[key]
            held_flat.append({"part":row["part"],"arm":row["arm"],"moment":key,
                **dict(zip(("CAD_origin_grip_x_mm","CAD_origin_grip_y_mm","CAD_origin_grip_z_mm"),moment["CAD_origin_in_grip_site_mm"])),
                **dict(zip(("CAD_rotation_grip_qx","CAD_rotation_grip_qy","CAD_rotation_grip_qz","CAD_rotation_grip_qw"),moment["CAD_orientation_in_grip_site_quaternion_xyzw"]))})
    if held_flat:
        with (run/"held_part_poses.csv").open("w",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(held_flat[0]))
            writer.writeheader()
            writer.writerows(held_flat)
    flat = [{key:json.dumps(item,sort_keys=True) if isinstance(item,dict) else item for key,item in row.items()} for row in rows]
    with (run/"comparison.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    write_report(run,value)
    return value


def write_report(run,value):
    rows, summary = value["parts"],value["summary"]
    text = ["# Three-part CAD-refined GR-ConvNet pickup experiment", "",
        "This trial repeats the large gear, medium gear and M16 nut from the same original 14-part scene. The previous learned-method run attempted none of these three pickups. Every fresh condition includes all three parts, including any search failure or unavailable VLM answer. It does not repeat or replace the old 14-part benchmark.", "",
        "Success requires at least 120 mm rise, a one-second hold with both fingers touching, and robot overlap with itself, the table and fixed board hardware below 1 mm. At most one physical simulation pickup is attempted per part and condition. No assembly views or assembly movement are included.", "",
        "| Method | Picked up and held | Physical attempts | Median decision time | Total decision time | Total recorded time |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        item = summary[arm]
        text.append(f"| {ARM_LABELS[arm]} | {item['pickup_passed']}/3 | {item['physical_attempts']} | {item['median_decision_seconds']:.3f} s | {item['total_decision_seconds']:.3f} s | {item['total_recorded_seconds']:.3f} s |")
    text += ["", "Decision time includes fresh image capture, preprocessing, inference where used, grasp conversion/refinement, evidence saving, VLM rendering/calls where used, and robot-path checks. Total recorded time also includes per-condition setup, actual contact simulation and snapshots. A rejected condition may take less total time because it never executes a pickup. Model setup is reported separately in the run manifest; total recorded time is not a physical robot cycle time.", "", "## Every condition", "",
             "| Part | Method | Result and stopping reason | Decision time | Total time |", "| --- | --- | --- | ---: | ---: |"]
    for row in rows:
        text.append(f"| {LABELS[row['part']]} | {ARM_LABELS[row['arm']]} | {row['reason'].replace('|','/')} | {row['decision_seconds']:.3f} s | {row['total_seconds']:.3f} s |")
    text += ["", "## What CAD changed in the original network proposals", "",
        "CAD refinement keeps the original network closing direction. Each of at most four image proposals receives a fixed search of five nearby sideways offsets and three CAD height slices: 15 contact lines. Combining those lines with the six permitted finger depths and three opening margins forms at most 270 variants per proposal. The first variant passing the fixed geometry checks becomes that proposal's single candidate before VLM review; no later depth is substituted after ranking.", "",
        "The table shows the retained pose for each original network seed in the score-order condition; the independent audit checks that the VLM condition receives the identical corrected pool. Height refers to the contact midpoint above the known table. Finger opening is the separation commanded before descent. Finger-depth offset locates the proposed contact midpoint relative to the grip site along its approach axis.", "",
        "| Part | Original network seed | XY center correction | Contact height, original → CAD | Contact span, predicted → CAD | Fixed finger opening | Fixed finger-depth offset | Lowest finger gap above table |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in value["CAD_proposal_corrections"]:
        text.append(f"| {LABELS[row['part']]} | {row['network_candidate_id']} | {row['XY_contact_center_change_m']*1000:.3f} mm | {row['original_contact_center_above_table_m']*1000:.3f} → {row['CAD_contact_center_above_table_m']*1000:.3f} mm | {row['original_predicted_span_m']*1000:.3f} → {row['CAD_contact_span_m']*1000:.3f} mm | {row['fixed_finger_opening_m']*1000:.3f} mm | {row['fixed_finger_depth_m']*1000:+.0f} mm | {row['table_gap_m']*1000:.3f} mm |")
    text += ["", "## How much the part moved inside the gripper", "",
        "`world_T_grip` is the pose of the Panda grip site in the workspace. `world_T_cad` is the part's CAD/body pose in the workspace. Both use meters. The part pose relative to the gripper is `grip_T_cad = inverse(world_T_grip) @ world_T_cad`. It converts a CAD-local point into grip-site coordinates. The planned value uses the commanded candidate and the supplied initial CAD alignment. Actual values are independently recomputed from the recorded gripper and part poses after closing, after lifting and after holding.", "",
        "The planned relation can differ from the relation after closing because the robot does not reach the command perfectly and the fingers can move the part as they close. Movement measured after closure describes the part shifting relative to the gripper during the lift and hold; it does not claim a perfectly rigid attachment. Displacement tracks the CAD origin, while angle tracks the CAD axes. Simulator part poses here are audit measurements, not controller feedback.", "",
        "| Part | Method | During closing: displacement / angle | Planned to closed: displacement / angle | Closed to end hold: displacement / angle | Largest sampled movement after closure |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        changes = row["relative_changes"]
        values = []
        for key in ("during_closing","planned_to_closed","closed_to_end_hold","maximum_after_closure"):
            item = changes.get(key)
            values.append("No corresponding movement" if item is None else f"{item['cad_origin_displacement_m']*1000:.3f} mm / {item['rotation_deg']:.3f}°")
        text.append("| "+" | ".join([LABELS[row["part"]],ARM_LABELS[row["arm"]],*values])+" |")
    text += ["", "Each condition saves `independent_grip_relative.json`, including the actual transformation matrices and independent reconstruction checks. The numerical trace is separate from the pictures. The gallery retains the original wide pickup frames and also shows post-run closeups rendered from the actual saved closure/held states without additional physics. CAD score/VLM trajectories and contact records are independently verified identical, so both cards use the same score-arm closeups.", "", "## Time at each step", "",
             "Stage totals cover all three parts, including conditions with no pickup. An absent stage was not used.", "",
             "| Stage | "+" | ".join(ARM_LABELS[arm] for arm in ARMS)+" |",
             "| --- | "+" | ".join("---:" for _ in ARMS)+" |"]
    keys = sorted({key for item in summary.values() for key in item["total_stage_seconds"]})
    for key in keys:
        values = [f"{summary[arm]['total_stage_seconds'][key]:.3f} s" if key in summary[arm]["total_stage_seconds"] else "Not used" for arm in ARMS]
        text.append("| "+" | ".join([key.replace("_"," "),*values])+" |")
    manifest = value["manifest"]
    text += ["", f"GR-ConvNet setup took **{manifest['model_setup_seconds_one_time']:.3f} s once** before the trials. The common local Qwen preload took **{manifest['vlm_shared_warmup']['wall_seconds']:.3f} s once**, also outside all per-condition timers. These shared costs are reported once rather than assigned to the first method. [Shared preload evidence](../../vlm_shared_warmup.json).", ""]
    exceptions = value["timing_exception"]["calls_with_API_reported_load_over_one_second"]
    for item in exceptions:
        text += [f"**The intended all-warm VLM condition was not achieved.** Despite the preload, the {ARM_LABELS[item['arm']]} call for the {LABELS[item['part']].lower()} reported **{item['API_reported_load_seconds_already_in_call']:.6f} s** of model reloading, within **{item['api_wall_seconds']:.6f} s** API wall time and **{item['bridge_wall_seconds']:.6f} s** full bridge time. That reload remains inside the measured decision and total times; it is not subtracted or added twice. The reason for the reload is not established by these records. The other calls reported only millisecond-scale load durations. Comparing the two VLM conditions' totals therefore includes this unequal loading cost.", ""]
    text += ["The original 14-part run also retained its first cold VLM load inside the large-gear decision time. Those historical values remain unchanged and use a different set of parts. The fresh geometric and CAD-refined score-order conditions directly compare three successful pickups each without a VLM loading cost.", "",
             "| VLM condition | Actual calls | Valid orders | Median full bridge call time | Order changed on |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm in ("grconvnet_vlm","cad_grconvnet_vlm"):
        calls = [item for item in value["vlm_calls"] if item["arm"] == arm]
        median = "No calls" if not calls else f"{statistics.median(item['bridge_wall_seconds'] for item in calls):.3f} s"
        text.append(f"| {ARM_LABELS[arm]} | {len(calls)} | {sum(item['status'] == 'ok' for item in calls)} | {median} | {sum(item['order_changed'] for item in calls)} parts |")
    refined = [row for row in rows if row["arm"] == "cad_grconvnet_vlm"]
    if all(row["candidate_count"] == 1 for row in refined):
        text += ["", "CAD refinement retained exactly one candidate for each of these three parts. The VLM therefore received a single valid entry each time; these CAD-refined trials cannot demonstrate a benefit from choosing between several usable poses. Its call cost remains included."]
    text += ["", "## Current held-part position and orientation", "",
        "For each successful CAD-refined pickup, the table below gives the CAD origin's position relative to the Panda grip site after closing and after the lift/hold. Positions are millimeters. Orientation is a normalized quaternion in `[x, y, z, w]` order, describing the CAD axes relative to the grip-site axes; a quaternion and its negative describe the same orientation. These are simulator audit measurements, not a calibrated real-world held-part estimator.", "",
        "| Part | Method | Moment | CAD origin in grip-site coordinates [x, y, z], mm | CAD orientation quaternion [x, y, z, w] |",
        "| --- | --- | --- | --- | --- |"]
    for row in value["held_part_poses"]:
        for key,label in (("after_closing","After closing"),("after_lift_and_hold","After lift and hold")):
            moment = row[key]
            position = ", ".join(f"{item:.4f}" for item in moment["CAD_origin_in_grip_site_mm"])
            quaternion = ", ".join(f"{item:.7f}" for item in moment["CAD_orientation_in_grip_site_quaternion_xyzw"])
            text.append(f"| {LABELS[row['part']]} | {ARM_LABELS[row['arm']]} | {label} | [{position}] | [{quaternion}] |")
    text += ["", "If the measured part-in-gripper relation stays unchanged, a desired CAD-frame destination gives `world_T_grip_goal = world_T_cad_goal @ inverse(grip_T_cad)`. The destination must specify the same CAD frame, including the offset from the CAD origin to its mating feature. A real system must also account for camera/robot/grip-site calibration, any movement during closure/lift and orientation ambiguity of symmetric parts. This trial checks the transformation arithmetic; it does not plan or execute an assembly move.", "",
             "[Full closure/held matrices and movement measurements](held_part_poses.json), [positions and quaternions as CSV](held_part_poses.csv).", ""]
    text += ["", "## Assumptions and limits", "",
        "GR-ConvNet weights and the pickup-ranking prompt remain frozen. The CAD-refined conditions use the supplied correct CAD alignment to improve conversion of the same network image proposals into gripper poses. The network itself is not retrained. Geometric generation also receives the correct aligned CAD. All learned conditions receive the same renderer-provided perfect visible target mask and known camera calibration. This does not measure object-identification, segmentation or pose-estimation error.", "",
        "All other loose parts remain visible with contacts disabled; the original controller resets their saved positions each physics step, followed by the existing one-step gravity offset. The table and fixed board hardware remain solid. The trial is one reused development scene, three selected earlier failures and one possible pickup attempt per condition. It cannot establish general performance on other poses or physical hardware.", "",
        "[Pickup gallery](gallery/index.html), [per-condition CSV](comparison.csv), [full comparison](comparison.json), [independent audit](independent_audit.json), [run manifest](manifest.json).", ""]
    (Path(run)/"report.md").write_text("\n".join(text),encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1]))["summary"],indent=2))
