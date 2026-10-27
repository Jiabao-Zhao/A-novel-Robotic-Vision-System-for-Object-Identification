"""Plain-language report/gallery of frozen observation-only assembly attempts."""
import csv
import html
import json
import math
from pathlib import Path
import re
import sys

from .audit import ROOT, PARTS, goal_error, grip_T_cad, local_path, quaternion_xyzw, read, save


LABELS = {"Gear_Large":"Large gear","Gear_Medium":"Medium gear","M16_Hex_Nut":"M16 nut"}
PHASES = {"turn_for_nominal_goal":"turning the held part toward the nominal board orientation",
          "transfer_above_board":"moving above the board","pickup_approach":"moving above the part",
          "pickup_descend":"lowering the open fingers","pickup_lift":"lifting the part"}


def save_transform_files(run,flat_estimates):
    run = Path(run)
    save(run/"observed_part_in_gripper_poses.json",flat_estimates)
    latest = [next(row for row in reversed(flat_estimates) if row["part"] == part and row["status"] == "available") for part in PARTS
        if any(row["part"] == part and row["status"] == "available" for row in flat_estimates)]
    save(run/"latest_observed_part_in_gripper_poses.json",latest)
    csv_rows = [{key:json.dumps(item,sort_keys=True) if isinstance(item,(dict,list)) else item for key,item in row.items()} for row in flat_estimates]
    if csv_rows:
        with (run/"observed_part_in_gripper_poses.csv").open("w",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)


def fit_reason(estimate):
    status,diagnostics = estimate["status"],estimate["diagnostics"]
    if status == "available":
        return "An observation-based CAD fit was available."
    if status == "unavailable_no_CAD_depth_match":
        bank = diagnostics["candidate_components"]
        return ("No observed component matched the supplied CAD." if not bank else
                f"Best observed component score {bank[0]['score']:.4f} was below the fixed 0.5500 limit.")
    if status == "unavailable_ambiguous_CAD_components":
        bank = diagnostics["candidate_components"]
        return f"Two observed components remained plausible: scores {bank[0]['score']:.4f} and {bank[1]['score']:.4f}; their gap was below 0.0200."
    if status == "unavailable_insufficient_nonrobot_depth":
        return "Fewer than 30 usable depth points remained after known-robot removal, the observed motion-prior crop and downsampling."
    if status == "unavailable_poor_CAD_depth_agreement":
        best = (read(local_path(estimate["ranked_candidates_path"]))[0] if "ranked_candidates_path" in estimate
                else diagnostics["local_pose_candidates"][0])
        forward = diagnostics["forward_depth_check"]
        causes = []
        fraction = best.get("inlier_fraction_within_1mm",best.get("observed_point_inlier_fraction_within_1mm"))
        if fraction < .5:
            causes.append(f"{fraction*100:.2f}% of retained depth points were within 1 mm of CAD; at least 50% was required")
        if not forward["passed"]:
            causes.append(f"{forward['fraction_within_2mm']*100:.2f}% of {forward['assessable_CAD_pixels']} assessable CAD image pixels agreed with measured depth within 2 mm; at least 50% and 30 pixels were required")
        extra = (f" Forward CAD-to-image depth agreement passed at {forward['fraction_within_2mm']*100:.2f}%."
                 if forward["passed"] else "")
        return "; ".join(causes)+"."+extra
    if status == "unavailable_exact_yaw_tie":
        return (f"Two distinct CAD orientations had a numerical score tie: squared-loss gap "
            f"{diagnostics['distinct_solution_combined_loss_gap_m2']:.6g} m² was within "
            f"{diagnostics['numerical_exact_tie_tolerance_m2']:.6g} m². No accepted pose was supplied.")
    return status.replace("_"," ")+"."


def stopping_reason(result,estimates):
    status = result["status"]
    latest = estimates[-1]["estimate"] if estimates else None
    if latest is not None and latest["status"] != "available":
        return "Stopped after "+estimates[-1]["label"].replace("_"," ")+": "+fit_reason(latest)+" No later move or release was commanded."
    if status == "robot_path_rejected":
        if result.get("stopping_phase") == "no_staged_exact_goal_route":
            route = result["staged_route_search"]
            return f"No bounded staged route to the exact original nominal CAD goal passed robot joint-angle and collision checks ({len(route['candidates'])} routes checked). No turn or board transfer was executed."
        movement = result["movement_checks"][-1]
        failed = next(row for row in movement["reports"] if row["status"] != "passed")
        phase = PHASES.get(movement["phase"],movement["phase"].replace("_"," "))
        if failed["status"] == "ik_failed":
            detail = failed.get("reason","")
            position = re.search(r"position_error_m=([\d.eE+-]+)",detail)
            rotation = re.search(r"rotation_error_rad=([\d.eE+-]+)",detail)
            if position and rotation:
                detail = (f"The next sampled gripper pose remained {float(position.group(1))*1000:.3f} mm "
                    f"from its requested position and {math.degrees(float(rotation.group(1))):.3f}° from its orientation "
                    f"after {failed.get('checked_waypoints',0)} earlier waypoints passed.")
            return "The planned move was rejected before execution while "+phase+": robot joint angles were not found. "+detail
        contacts = [" / ".join(row.get("bodies",row.get("geoms",[])))+f" ({row['penetration_m']*1000:.3f} mm overlap)" for row in failed.get("contacts",[])]
        return "The planned move was rejected before execution while "+phase+": "+"; ".join(contacts)+"."
    if status == "observed_part_shift_during_lowering":
        row = result["board_observation_checks"][-1]
        change = row["observed_grip_relation_change"]
        position_detail = ("exceeding the 3 mm limit" if change["translation_m"] > .003 else "within the 3 mm limit")
        rotation_detail = ("exceeding the 5° limit" if change["rotation_deg"] > 5. else "below the 5° limit")
        return f"Stopped during board lowering: observed part position relative to the hand changed {change['translation_m']*1000:.3f} mm, {position_detail}; rotation changed {change['rotation_deg']:.3f}°, {rotation_detail}."
    if status == "observed_lowering_stalled":
        row = result["board_observation_checks"][-1]
        return f"Stopped during board lowering: expected drop {row['expected_drop_m']*1000:.3f} mm, observed drop {row['observed_drop_m']*1000:.3f} mm. Their difference exceeded the larger of 1 mm and half the expected drop."
    if status == "observed_pickup_rise_too_small":
        return f"Observed pickup rise was {result['observed_pickup_rise_m']*1000:.3f} mm, below the 120 mm requirement."
    if status == "no_observation_based_pickup_path":
        return "No candidate passed the observation-based CAD/gripper and robot-path checks; no fingers-close pickup attempt was executed."
    if status == "observed_nominal_pose_outside_tolerance":
        error = result["observed_goal_error_before_release"]
        return f"Kept the fingers closed: observed nominal errors were position {error['position_m']*1000:.3f} mm / signed CAD axis {error['signed_CAD_Z_axis_deg']:.3f}° / full CAD rotation {error['raw_rotation_deg']:.3f}°. The recorded final pose gate failed."
    if status == "observed_nominal_pose_reached_no_engagement_claim":
        return "Observed nominal position and signed CAD-axis limits passed after release/retreat. Gear engagement or nut threading was not verified."
    if result.get("error"):
        return "Stopped with recorded controller/estimation error: "+result["error"]
    return status.replace("_"," ")+"."


def summarize(run):
    run = Path(run).resolve()
    manifest = read(run/"manifest.json")
    offline = {row["part"]:row for row in read(run/"offline_evaluation.json")["parts"]}
    rpc = read(run/"client_rpc_log.json")
    rpc_costs = {part:{} for part in PARTS}
    active = None
    for call in rpc:
        request = read(run/call["request"])
        if call["action"] == "reset":
            active = request["part"]
        if active is not None:
            costs = rpc_costs[active]
            costs[call["action"]] = costs.get(call["action"],0.)+call["wall_seconds"]
        if call["action"] == "finish":
            active = None
    rows,flat_estimates = [],[]
    for part in PARTS:
        result = read(run/"controller"/part/"result.json")
        goal = manifest["nominal_goals"][part]["world_T_cad_goal"]
        entries = []
        for record in result["observed_estimates"]:
            estimate = read(local_path(record["estimate_path"]))
            feedback = read(local_path(record["robot_feedback_path"]))
            error = None if estimate["status"] != "available" else goal_error(estimate["world_T_cad"],goal)
            relative = None if estimate["status"] != "available" else grip_T_cad(feedback["world_T_grip"],estimate["world_T_cad"])
            item = {**record,"estimate":estimate,"observed_goal_error":error,"estimated_grip_T_cad":relative}
            entries.append(item)
            flat_estimates.append({"part":part,"label":record["label"],"status":estimate["status"],
                "world_T_cad":estimate["world_T_cad"],"grip_T_cad":relative,
                "CAD_origin_in_grip_site_mm":None if relative is None else [relative[i][3]*1000 for i in range(3)],
                "CAD_orientation_in_grip_site_quaternion_xyzw":None if relative is None else quaternion_xyzw(relative),
                "local_six_parameter_support":estimate["diagnostics"].get("local_six_parameter_support"),
                "global_CAD_orientation_uniquely_verified":False,"observed_goal_error":error,
                "rigid_attachment_assumed":record["label"] not in {"initial","after_release","after_retreat"},
                "fit_reason":fit_reason(estimate),"fit_seconds":estimate["diagnostics"].get("seconds"),
                "source_estimate_path":record["estimate_path"]})
        accepted = [row for row in entries if row["estimate"]["status"] == "available"]
        last = accepted[-1] if accepted else None
        row = {"part":part,"status":result["status"],"reason":stopping_reason(result,entries),
            "physical_pickup_attempts":result["physical_pickup_attempts"],
            "observed_pickup_passed":result.get("observed_pickup_passed",False),
            "observed_pickup_rise_m":result.get("observed_pickup_rise_m"),
            "board_lowering_moves":result["board_lowering_moves"],"release_commanded":result["release_commanded"],
            "latest_observation_label":entries[-1]["label"] if entries else None,
            "latest_observation_status":entries[-1]["estimate"]["status"] if entries else None,
            "last_accepted_label":None if last is None else last["label"],
            "last_accepted_observed_goal_error":None if last is None else last["observed_goal_error"],
            "offline":offline[part],"wall_seconds":result["wall_seconds"],"RPC_wall_seconds_by_action":rpc_costs[part],
            "fit_seconds_only":sum(row["estimate"]["diagnostics"].get("seconds",0.) for row in entries),
            "estimate_count":len(entries),"available_estimate_count":len(accepted)}
        rows.append(row)
    value = {"parts":rows,"observed_pose_updates":flat_estimates,"manifest":manifest,
        "summary":{"part_count":3,"pickup_attempts":sum(row["physical_pickup_attempts"] for row in rows),
            "observed_pickups":sum(row["observed_pickup_passed"] for row in rows),
            "offline_height_and_two_finger_pickups":sum(bool(row["offline"]["pickup_OFFLINE_ONLY"] and row["offline"]["pickup_OFFLINE_ONLY"]["height_and_two_finger_hold_passed"]) for row in rows),
            "parts_with_board_lowering_move":sum(row["board_lowering_moves"] > 0 for row in rows),
            "parts_released":sum(row["release_commanded"] for row in rows),
            "observed_nominal_pose_passed":sum(row["status"] == "observed_nominal_pose_reached_no_engagement_claim" for row in rows),
            "mechanical_engagement_verified":0,"total_part_wall_seconds":sum(row["wall_seconds"] for row in rows)}}
    save(run/"comparison.json",value)
    save_transform_files(run,flat_estimates)
    write_report(run,value)
    write_gallery(run,value)
    return value


def write_report(run,value):
    summary = value["summary"]
    yaw_feedback = "yaw_protocol" in value["manifest"]
    gear_errors = [next(item for item in row["offline"]["observations"] if item["label"] == "after_pickup")["estimated_pose_error"]
        for row in value["parts"] if row["part"].startswith("Gear_") and any(item["label"] == "after_pickup" for item in row["offline"]["observations"])]
    wrong_gear = any(error is not None and error["raw_rotation_deg"] > 2. for error in gear_errors)
    phase_lead = ("**The gear orientations were wrong despite available depth fits.** The independent check found large full-CAD rotation errors after pickup. Small position and directed-axis errors did not establish the gear's correct rotation around its axis. These different CAD phases are not declared equivalent, and these results do not establish full 6D pose accuracy." if wrong_gear else
        "The post-pickup full CAD orientation was compared with withheld truth after the run. The table below gives actual measured errors for this simulation. Finite depth-ranking gaps remain uncalibrated; these frames do not establish unique orientation accuracy with real sensor noise or on hardware.")
    text = ["# Observation-only held-part updates and nominal board attempts", "",
        f"All three requested parts are included. There were **{summary['pickup_attempts']}/3 pickup attempts**, **{summary['observed_pickups']}/3 observed pickup passes**, and **{summary['parts_with_board_lowering_move']}/3 parts with an executed board-lowering move**. **{summary['parts_released']}/3 parts were released**. The observed nominal-pose check passed for **{summary['observed_nominal_pose_passed']}/3**; mechanical gear engagement or nut threading was verified for **0/3**.", "",
        ("This run's nominal-pose check includes position within 1 mm, directed CAD-axis angle within 2° and full observed CAD rotation within 2°." if "nominal_final_full_CAD_rotation_tolerance_deg" in value["manifest"]["controller_protocol"] else
         "This run's nominal-pose check includes position within 1 mm and directed CAD-axis angle within 2°; full rotation around that axis is not checked."), "",
        phase_lead, "",
        "| Part | Post-pickup CAD-origin error, offline only | Directed CAD-axis error, offline only | Full CAD rotation error, offline only |",
        "| --- | ---: | ---: | ---: |"]
    for row in value["parts"]:
        pickup = next((item for item in row["offline"]["observations"] if item["label"] == "after_pickup"),None)
        error = None if pickup is None else pickup["estimated_pose_error"]
        values = ["No accepted pose"]*3 if error is None else [f"{error['position_m']*1000:.6f} mm",f"{error['signed_CAD_Z_axis_deg']:.6f}°",f"{error['raw_rotation_deg']:.6f}°"]
        text.append("| "+" | ".join([LABELS[row["part"]],*values])+" |")
    initial_errors = []
    for row in value["parts"]:
        if not row["part"].startswith("Gear_"):
            continue
        initial = next(item for item in row["offline"]["observations"] if item["label"] == "initial")
        if initial["estimated_pose_error"] is not None:
            initial_errors.append(f"{LABELS[row['part']]} {initial['estimated_pose_error']['raw_rotation_deg']:.6f}°")
    text += ["",f"**{sum(row['available_estimate_count'] for row in value['parts'])}/{sum(row['estimate_count'] for row in value['parts'])} fits were marked available; that does not mean all were accurate poses.** Initial table gear rotation errors were "+", ".join(initial_errors)+(". The fresh full-circle held search corrected the gear angles after pickup; the initial table fits were still wrong." if yaw_feedback else ". Availability did not resolve the gear angles."), ""]
    text += ["",
        "The controller used RGB-D, four known fixed camera views, robot encoders/forward kinematics, supplied native CAD and the previously approved nominal static goals. Engine target poses, target segmentation and labelled target contacts were unavailable through the online interface. All online files were frozen and the plant was shut down before a separate offline evaluator read target truth. The offline comparison does not change any stop decision.", "",
        "| Part | Pickup attempts | Observed pickup pass | Actual board-lowering moves | Released | What stopped or passed |",
        "| --- | ---: | --- | ---: | --- | --- |"]
    for row in value["parts"]:
        text.append(f"| {LABELS[row['part']]} | {row['physical_pickup_attempts']} | {'Yes' if row['observed_pickup_passed'] else 'No'} | {row['board_lowering_moves']} | {'Yes' if row['release_commanded'] else 'No'} | {row['reason'].replace('|','/')} |")
    text += ["", "## Last accepted measurement and final offline state", "",
        "A commanded height is a requested move, not a measured part height. An unavailable fit supplies no accepted pose. The last accepted observation and the final simulator state are therefore listed separately.", "",
        "| Part | Last accepted observation | Observed height above nominal goal | Observed XY error | Observed signed CAD-axis error | Final actual height above goal, offline only | Final actual XY error, offline only |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in value["parts"]:
        observed = row["last_accepted_observed_goal_error"]
        final = row["offline"]["final_actual_goal_error_OFFLINE_ONLY"]
        observed_values = ["Unavailable"]*3 if observed is None else [f"{observed['height_m']*1000:.3f} mm",f"{observed['xy_m']*1000:.3f} mm",f"{observed['signed_CAD_Z_axis_deg']:.3f}°"]
        text.append("| "+" | ".join([LABELS[row["part"]],str(row["last_accepted_label"]),*observed_values,f"{final['height_m']*1000:.3f} mm",f"{final['xy_m']*1000:.3f} mm"])+" |")
    text += ["", "## Every observation-based update", "",
        ("The initial pose is found from connected components of observed depth above the calibrated table. For each fresh gear pickup, the held estimator tests a full circle (60 starts for the large gear, 40 for the medium gear) around the observation/FK prior, then ranks fitted CADs by point-to-surface and raw projected-depth error. Later gear updates test five declared CAD-angle starts. Known robot/table/board depth is removed for fitting; ranking uses raw depth while excluding pixels hidden by known static geometry. The nut keeps the original local held fitter. No engine target mask, true target crop, true target pose or earlier diagnostic-selected pose initializes these fits." if yaw_feedback else
        "The initial pose is found from connected components of observed depth above the calibrated table. Held updates use the preceding observed pose and measured robot motion as a prior, then fit XYZ and local rotation from depth. Known robot geometry predicts robot-only depth for removing robot pixels. No engine target mask, true target crop or true target pose initializes these fits."), "",
        "The estimated part-in-gripper transform is `grip_T_cad = inverse(world_T_grip_FK) @ estimated_world_T_cad`. A desired CAD-frame pose gives `world_T_grip_goal = world_T_cad_goal @ inverse(grip_T_cad)`. The transform is refreshed from each accepted observation. After release, a part-to-gripper transform describes only their current spatial relation; it is no longer a rigid attachment assumption.", "",
        "| Part | Stage | Availability and exact quality reason | Fit time | Local six-parameter support |",
        "| --- | --- | --- | ---: | --- |"]
    for row in value["observed_pose_updates"]:
        elapsed = "Unavailable" if row["fit_seconds"] is None else f"{row['fit_seconds']:.3f} s"
        support = "Table-constrained initial fit" if row["local_six_parameter_support"] is None else str(row["local_six_parameter_support"])
        text.append(f"| {LABELS[row['part']]} | {row['label']} | {row['fit_reason']} | {elapsed} | {support} |")
    text += ["", ("Local support and ranking gaps are numerical diagnostics, not probabilities of correct CAD phase. The final gate checks nominal position within 1 mm, directed CAD Z axis within 2° and full observed CAD rotation within 2°. Mechanical gear engagement and nut thread phase remain unverified." if "nominal_final_full_CAD_rotation_tolerance_deg" in value["manifest"]["controller_protocol"] else
        "Local support is a numerical diagnostic. Repeated gear/nut geometry still leaves global orientation alternatives. The final gate checks position within 1 mm and the directed CAD Z axis within 2°; gear tooth phase, full yaw agreement and nut thread phase remain unverified."), "",
        "## Independent offline accuracy check", "",
        "These errors compare each frozen estimate with the target pose at the same recorded robot state, after the experiment ended. Translation is CAD-origin error. Axis error compares the directed CAD Z axes. Raw rotation is the full CAD-frame error: locally similar visible surfaces do not make different keyway, tooth or thread phases equivalent. An available local fit does not verify those phases.", "",
        "| Part | Stage | CAD-origin error | Directed CAD-axis error | Raw CAD rotation error | Prior CAD-origin error before visual update |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in value["parts"]:
        for observation in row["offline"]["observations"]:
            error,prior = observation["estimated_pose_error"],observation["transport_prior_error"]
            if error is None:
                values = ["No accepted estimate"]*3
            else:
                values = [f"{error['position_m']*1000:.3f} mm",f"{error['signed_CAD_Z_axis_deg']:.3f}°",f"{error['raw_rotation_deg']:.3f}°"]
            prior_value = "Initial observation" if prior is None else f"{prior['position_m']*1000:.3f} mm"
            text.append("| "+" | ".join([LABELS[row["part"]],observation["label"],*values,prior_value])+" |")
    if yaw_feedback:
        text += ["", "## Fresh gear-angle ranking evidence", "",
            "The full-circle search is repeated from this run's own pickup observations. It does not reuse the diagnostic's selected poses. Later banks contain five declared nearby starts. RMS below is the square root of the equal-weight clipped point/forward-depth squared loss; it is a geometric error measure, not a probability.", "",
            "The numeric depth-agreement limits stay unchanged. Projected-pixel eligibility now excludes occlusion by known robot/table/board geometry; the first run's forward check excluded known robot occlusion only. Fitting also removes observed depth that matches the known static scene within 1 mm.", "",
            "| Part | Stage | Starts | Selected / next distinct clipped RMS | Full rotation change from observed-motion prior | Exact numerical tie |",
            "| --- | --- | ---: | ---: | ---: | --- |"]
        bank_summary = []
        for row in value["observed_pose_updates"]:
            estimate = read(local_path(row["source_estimate_path"]))
            if "ranked_candidates_path" not in estimate:
                continue
            diagnostics = estimate["diagnostics"]
            bank = read(local_path(estimate["ranked_candidates_path"]))
            distinct = diagnostics.get("distinct_yaw_ranked_candidate_ids",[])
            best = bank[0] if bank else None
            next_distinct = next((item for item in bank if len(distinct)>1 and item["candidate_id"] == distinct[1]),None)
            rms = "Unavailable" if best is None else f"{math.sqrt(best['combined_loss_m2'])*1e6:.3f} µm / "+("Unavailable" if next_distinct is None else f"{math.sqrt(next_distinct['combined_loss_m2'])*1e6:.3f} µm")
            rotation = diagnostics.get("rotation_change_from_motion_prior_deg")
            rotation_text = "Unavailable" if rotation is None else f"{rotation:.6f}°"
            text.append(f"| {LABELS[row['part']]} | {row['label']} | {len(bank)} | {rms} | {rotation_text} | {diagnostics.get('numerical_exact_tie_between_distinct_rotations','Not evaluated')} |")
            bank_summary.append({"part":row["part"],"stage":row["label"],"status":row["status"],
                "candidate_count":len(bank),"selected_candidate_id":diagnostics.get("selected_candidate_id"),
                "next_distinct_candidate_id":distinct[1] if len(distinct)>1 else None,
                "selected_combined_clipped_rms_m":None if best is None else math.sqrt(best["combined_loss_m2"]),
                "next_distinct_combined_clipped_rms_m":None if next_distinct is None else math.sqrt(next_distinct["combined_loss_m2"]),
                "rotation_change_from_motion_prior_deg":rotation,"ranking_confidence_calibrated":False})
        save(Path(run)/"gear_yaw_bank_summary.json",bank_summary)
    phase_path = Path(run)/"offline_initial_phase_diagnostics.json"
    if phase_path.is_file():
        phase_diagnostics = read(phase_path)
        text += ["", "## Why the initial gear phase is not established", "",
            phase_diagnostics["interpretation"], "",phase_diagnostics["held_update_limit"], "",
            "| Part | Highest-scoring tested pose: score / full rotation error | Next score gap | Closest tested rotation: error / score | Tested poses within 2° of withheld truth |",
            "| --- | --- | ---: | --- | ---: |"]
        for item in phase_diagnostics["parts"]:
            best = item["score_ranked_hypotheses"][0]
            closest = item["closest_tested_full_rotation"]
            text.append(f"| {LABELS[item['part']]} | {best['geometry_score']:.9f} / {best['pose_error_OFFLINE_ONLY']['raw_rotation_deg']:.6f}° | {item['top_score_gap']:.9f} | {closest['pose_error_OFFLINE_ONLY']['raw_rotation_deg']:.6f}° / {closest['geometry_score']:.9f} | {item['hypotheses_within_2deg_OFFLINE_ONLY']} |")
        text += ["", ("The full-circle held bank supplies additional phase evidence in this new trial. The initial component-ambiguity check alone did not resolve these orientation alternatives. The observed full-rotation release gate by itself would not repair a globally wrong CAD frame." if yaw_feedback else
            "An observed full-rotation goal check would compare the estimated CAD frame with the requested frame. It would not detect a globally incorrect initial CAD frame. A cue or an ambiguity decision is still needed before claiming the correct gear phase.")]
    contacts_path = Path(run)/"offline_contact_diagnostics.json"
    if contacts_path.is_file():
        contact_diagnostics = read(contacts_path)
        text += ["", "## Separate offline contact diagnosis", "",contact_diagnostics["scope"], "",
            contact_diagnostics["sampling"], "",contact_diagnostics["collision_limitation"], "",
            "| Part | First sampled target/board contact | Named geometry pair | Maximum target/board overlap, every physics step | Peak sampled sum of board normal forces |",
            "| --- | --- | --- | ---: | ---: |"]
        for item in contact_diagnostics["parts"]:
            first = item["first_target_board_contact"]
            peak = item["peak_sampled_board_force"]
            pair = "None" if first is None else " / ".join(first["contacts"][0]["geoms"])
            stage = "None" if first is None else f"{first['phase']}, {first['simulation_time_s']:.3f} simulated s"
            force = "0.000 N" if peak is None else f"{peak['sum_normal_force_n']:.3f} N"
            text.append(f"| {LABELS[item['part']]} | {stage} | {pair} | {item['every_physics_step_max_overlap_m']['target_board']*1000:.3f} mm | {force} |")
        robot_maxima = {kind:max(item["every_physics_step_max_overlap_m"][kind] for item in contact_diagnostics["parts"]) for kind in ("robot_self","robot_table","robot_board")}
        text += ["", "Maximum recorded overlaps across all parts were "+", ".join(f"{kind.replace('_',' / ')}: {value*1000:.3f} mm" for kind,value in robot_maxima.items())+". The contacts show where the simulated solid collision shapes met; they do not override the public RGB-D stop reasons above."]
    text += ["", "## Measured time", "",
        "Part totals include fresh detection, sensor requests, one-shot estimator process startup, fitting, frozen pretrained GR-ConvNet grasp generation, planning, contact simulation and recorded frames. Fit-only time excludes worker imports, camera capture and other pipeline work. Sensor and move/hold rows are complete RPC wall times. Simulation pauses between commands while fitting; this is stepwise visual feedback, not continuously running real-time tracking or a physical robot cycle.", "",
        "| Part | Full recorded wall time | Sensor request wall time | Pose fitting only | Move/hold simulation and recorded frames |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in value["parts"]:
        costs = row["RPC_wall_seconds_by_action"]
        text.append(f"| {LABELS[row['part']]} | {row['wall_seconds']:.3f} s | {costs.get('observe',0):.3f} s | {row['fit_seconds_only']:.3f} s | {costs.get('move',0)+costs.get('hold',0):.3f} s |")
    text += ["", f"Frozen GR-ConvNet loading took {value['manifest']['GR_model_load_seconds_once']:.3f} s once before part trials. No retraining, new weights, VLM calls or paid APIs were used.", "",
        ]
    preceding_path = ROOT/"outputs/observed_held_assembly_20261025/runs/20261025T201252Z/comparison.json"
    if yaw_feedback and preceding_path.is_file():
        preceding = {row["part"]:row for row in read(preceding_path)["parts"]}
        text += ["## Comparison with the preserved first observation run", "",
            "These are fresh resets and pickups in the same saved scene. The newer controller also changes static-depth removal, route search and number of executed motion segments, so total wall times are whole-attempt measurements with different work, not isolated estimator speed comparisons. Every requested part remains in the table.", "",
            "| Part | Earlier / new post-pickup full CAD rotation error, offline | Earlier / new executed board-lowering moves | Earlier / new full wall time |",
            "| --- | ---: | ---: | ---: |"]
        for row in value["parts"]:
            prior = preceding[row["part"]]
            def pickup_angle(item):
                entry = next((value for value in item["offline"]["observations"] if value["label"] == "after_pickup"),None)
                return "No accepted pose" if entry is None or entry["estimated_pose_error"] is None else f"{entry['estimated_pose_error']['raw_rotation_deg']:.6f}°"
            text.append(f"| {LABELS[row['part']]} | {pickup_angle(prior)} / {pickup_angle(row)} | {prior['board_lowering_moves']} / {row['board_lowering_moves']} | {prior['wall_seconds']:.3f} s / {row['wall_seconds']:.3f} s |")
        text.append("")
    text += ["## Limits and evidence", "",
        "The original 14-part scene and earlier pickup trials remain unchanged. Other loose parts stay visible with their contacts disabled. The robot/static planning model contains no live products; its held CAD is placed only from observed estimates. Robot self/table/board intersections veto planned moves. Estimated held-CAD/static contacts are diagnostic, not a complete insertion certificate. Conservative convex hulls close native CAD bores, and the physical model does not simulate gear meshing or nut threading. The goals remain nominal simulation assumptions.", "",
        "The four camera views and calibration are explicit simulation inputs. The initial table search uses CAD-derived upright support faces; it does not search every possible 3D resting orientation. Pose fitting is training-free; the grasp proposal network uses frozen pretrained weights. One scene and one possible pickup/board attempt per part do not establish performance on real depth, new layouts or hardware. The full human-guided interactive runtime is not claimed.", "",
        "[Pickup/board attempt gallery and videos](gallery/index.html), [latest accepted part-in-gripper transforms](latest_observed_part_in_gripper_poses.json), [all estimated part-in-gripper transforms](observed_part_in_gripper_poses.json), [transform CSV](observed_part_in_gripper_poses.csv), [offline evaluation](offline_evaluation.json), [offline initial-phase diagnosis](offline_initial_phase_diagnostics.json), [offline contact diagnosis](offline_contact_diagnostics.json), [independent audit](independent_audit.json), [online freeze before truth evaluation](online_freeze_before_offline_eval.json), [run manifest](manifest.json).", ""]
    (Path(run)/"report.md").write_text("\n".join(text),encoding="utf-8")


def write_gallery(run,value):
    run = Path(run).resolve()
    gallery = run/"gallery";gallery.mkdir(exist_ok=True)
    esc = lambda item:html.escape(str(item),quote=True)
    def link(path):
        path = Path(path)
        return "../"+path.relative_to(run).as_posix()
    text = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Observation-only board attempts</title>',
        '<style>body{font:16px/1.5 system-ui,sans-serif;color:#183044;background:#eef3f6;margin:0}main{max-width:1300px;margin:auto;padding:28px}article,section{background:white;border:1px solid #d4dfe6;border-radius:10px;padding:22px;margin:20px 0}h1,h2,h3{line-height:1.2}img,video{max-width:100%;height:auto}a{color:#176586}.note,figcaption{color:#526675;font-size:14px}figure{margin:20px 0}table{border-collapse:collapse}td,th{border:1px solid #d4dfe6;padding:8px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}</style><main>',
        '<h1>Observation-only held-part updates and board attempts</h1><p>All three parts are retained. Cyan CAD overlays are estimates from raw RGB-D, not simulator target poses. Videos show the actual fixed-camera frames recorded during motion. Fitting happens between motion segments with simulation paused.</p>',
        '<p>Post-pickup full CAD rotation errors are checked separately in the report after all online decisions were frozen. Matching outlines and available fits do not prove the correct orientation around the gear axis; different full CAD phases are not declared equivalent.</p>',
        '<p><a href="../report.md">Detailed report</a> · <a href="../observed_part_in_gripper_poses.json">Estimated relative transforms</a> · <a href="../independent_audit.json">Independent audit</a></p>']
    for row in value["parts"]:
        part = row["part"]
        text += [f'<section><h2>{esc(LABELS[part])}</h2><p>{esc(row["reason"])}</p>',
                 f'<p>Pickup attempts: {row["physical_pickup_attempts"]}. Board-lowering moves actually executed: {row["board_lowering_moves"]}. Release commanded: {esc(row["release_commanded"])}.</p>']
        video = gallery/"videos"/(part+".mp4")
        if video.is_file():
            text.append(f'<video controls preload="metadata" src="{esc(link(video))}"></video><p class="note">Actual recorded camera frames at their simulation-time rate. Pose-fitting pauses are not motion frames; full wall time is in the report.</p>')
        for estimate in value["observed_pose_updates"]:
            if estimate["part"] != part:
                continue
            image = gallery/"visualizations"/part/(estimate["label"]+"_estimated_pose.png")
            text.append(f'<article><h3>{esc(estimate["label"].replace("_"," "))}</h3><p>{esc(estimate["fit_reason"])}</p>')
            if image.is_file():
                text.append(f'<figure><a href="{esc(link(image))}" target="_blank"><img src="{esc(link(image))}" loading="lazy" alt="Raw camera views with observation-fitted CAD"></a><figcaption>Four calibrated camera views. Unavailable fits have no accepted CAD pose overlay.</figcaption></figure>')
            if estimate["grip_T_cad"] is not None:
                origin = ", ".join(f"{number:.3f}" for number in estimate["CAD_origin_in_grip_site_mm"])
                quaternion = ", ".join(f"{number:.6f}" for number in estimate["CAD_orientation_in_grip_site_quaternion_xyzw"])
                text.append(f'<p>Estimated CAD origin in grip-site coordinates: [{origin}] mm. CAD orientation relative to the grip site, quaternion [x,y,z,w]: [{quaternion}]. Keyway, tooth and thread phase remains unverified; different full-CAD orientations are not declared equivalent.</p>')
                if estimate["label"] == "initial":
                    text.append('<p class="note"><strong>Before pickup:</strong> this transform describes the part on the table relative to the still distant hand. It is not a rigid held-part relationship. The hand-to-held-part estimate starts after pickup.</p>')
            text.append(f'<p><a href="{esc(link(local_path(estimate["source_estimate_path"])))}">Frozen estimate, fit candidates and exact input provenance</a></p></article>')
        text.append('</section>')
    text.append('<p>Static destinations are nominal assumptions. Reaching a pose would not verify gear engagement or nut threading. Offline truth errors are separated in the report and never enter these estimates or online stops.</p></main></html>')
    (gallery/"index.html").write_text("\n".join(text),encoding="utf-8")


if __name__ == "__main__":
    value = summarize(Path(sys.argv[1]))
    print(json.dumps(value["summary"],indent=2))
