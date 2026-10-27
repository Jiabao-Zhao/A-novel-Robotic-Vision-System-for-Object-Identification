"""Post-run phase/contact diagnosis; never imported by an online controller."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from .audit import PARTS, digest, freeze_online_before_truth, goal_error, local_path, multiply, read, save


def require_completed_frozen_audit(run):
    """Reuse the completed full audit; do not rehash gigabytes per report table."""
    audit_path = run/"independent_audit.json"
    if not audit_path.is_file() or not read(audit_path)["passed"]:
        raise RuntimeError("Complete the independent online-input/offline evaluation audit first")
    frozen = {name.replace("\\","/"):sha for name,sha in read(run/"online_freeze_before_offline_eval.json").items()}
    for name in ("manifest.json","results.json","client_rpc_log.json"):
        if digest(run/name) != frozen[name]:
            raise RuntimeError("Terminal online manifest/commands changed after the complete audit")
    evaluation = read(run/"offline_evaluation.json")
    if not evaluation["online_files_frozen_before_truth_reads"] or not evaluation["online_files_unchanged_after_evaluation"]:
        raise RuntimeError("The independent evaluation did not preserve the freeze boundary")


def initial_phase_diagnostics(run):
    """Compare already-scored orientation hypotheses with withheld truth."""
    run = Path(run).resolve()
    require_completed_frozen_audit(run)
    offline = {row["part"]:row for row in read(run/"offline_evaluation.json")["parts"]}
    parts = []
    for part in PARTS:
        record = read(run/"controller"/part/"result.json")["observed_estimates"][0]
        estimate_path = local_path(record["estimate_path"])
        estimate = read(estimate_path)
        packet_path = estimate_path.parent/"packet.json"
        packet = read(packet_path)
        calibration_path = local_path(packet["observations"][0]["calibration_path"])
        calibration = read(calibration_path)
        truth = offline[part]["observations"][0]["actual_world_T_cad_OFFLINE_ONLY"]
        bank = []
        components = estimate["diagnostics"].get("candidate_components",[])
        if components:
            for index,hypothesis in enumerate(components[0]["pose_hypotheses"]):
                world_pose = multiply(calibration["world_T_camera"],hypothesis["T_camera_from_cad"])
                bank.append({"index":index,"support_index":hypothesis["support_index"],
                    "initial_yaw_deg":hypothesis["initial_yaw_deg"],
                    "geometry_score":hypothesis["verification"]["geometry_score"],
                    "pose_error_OFFLINE_ONLY":goal_error(world_pose,truth)})
        ranked = sorted(bank,key=lambda row:row["geometry_score"],reverse=True)
        parts.append({"part":part,"hypothesis_count":len(bank),
            "score_ranked_hypotheses":ranked,
            "top_score_gap":None if len(ranked)<2 else ranked[0]["geometry_score"]-ranked[1]["geometry_score"],
            "closest_tested_full_rotation":None if not bank else min(bank,key=lambda row:row["pose_error_OFFLINE_ONLY"]["raw_rotation_deg"]),
            "hypotheses_within_2deg_OFFLINE_ONLY":sum(row["pose_error_OFFLINE_ONLY"]["raw_rotation_deg"] <= 2. for row in bank),
            "source_hashes":{str(path.relative_to(run)):digest(path) for path in
                (estimate_path,packet_path,calibration_path,run/"offline_evaluation.json")}})
    yaw_feedback = "yaw_protocol" in read(run/"manifest.json")
    held_limit = ("This new trial adds full-circle gear-angle starts after pickup, with each start separately refined by the unchanged 20 degree bounded local fitter. The bank can therefore move beyond the original initial phase; later updates use the declared five local phase starts." if yaw_feedback else
        "Held registration searches local yaw starts 0/+5/-5/+10/-10 degrees with a 20 degree bound around its observation/FK prior. A locally available update cannot resolve a more than 100 degree initial CAD-phase error in one update.")
    value = {"scope":"Offline-only replay of saved initial orientation hypotheses; no inference, fitting, motion or controller correction.",
        "interpretation":"Small score gaps between wrong CAD phases show weak or misranked phase evidence for this view/scorer. This is not proof of exact CAD symmetry or that a cue is physically unobservable. The 0.02 initial ambiguity gate compares distinct components, not orientation hypotheses within one component.",
        "held_update_limit":held_limit,
        "parts":parts}
    save(run/"offline_initial_phase_diagnostics.json",value)
    return value


def contact_diagnostics(run):
    """Keep actual engine contact labels isolated from observed stop causes."""
    run = Path(run).resolve()
    require_completed_frozen_audit(run)
    parts = []
    for part in PARTS:
        folder = run/"plant_private"/part
        contacts = read(folder/"contacts.json")
        controller = read(folder/"controller_audit.json")
        actual_xml = ET.parse(folder/"scene.xml")
        geoms = {geom.get("name"):geom.attrib for geom in actual_xml.findall(".//geom")}
        touched = []
        for index,frame in enumerate(contacts):
            rows = [row for row in frame["contacts"] if row["kind"] == "target_board"]
            if rows:
                touched.append({"recorded_state_index":index,"simulation_time_s":frame["time_s"],
                    "phase":frame["phase"],"contacts":rows,
                    "sum_normal_force_n":sum(row["normal_force_n"] for row in rows),
                    "max_penetration_m":max(row["penetration_m"] for row in rows)})
        final = contacts[-1]
        final_fingers = [row for row in final["contacts"] if row["kind"] == "finger_target"]
        finger_forces = {}
        for row in final_fingers:
            name = next(name for name in row["bodies"] if name.startswith("gripper0_"))
            finger_forces[name] = finger_forces.get(name,0.)+row["normal_force_n"]
        used_geom_names = {name for frame in touched for row in frame["contacts"] for name in row["geoms"]}
        parts.append({"part":part,"sampled_target_board_contact_frames":len(touched),
            "first_target_board_contact":touched[0] if touched else None,
            "last_target_board_contact":touched[-1] if touched else None,
            "peak_sampled_board_force":max(touched,key=lambda row:row["sum_normal_force_n"]) if touched else None,
            "max_sampled_target_board_overlap_m":max((row["max_penetration_m"] for row in touched),default=0.),
            "every_physics_step_max_overlap_m":controller["max_penetration_m"],
            "target_board_geometry_definitions":{name:geoms[name] for name in sorted(used_geom_names)},
            "final_finger_contacts":final_fingers,"final_finger_normal_force_sums_n":finger_forces,
            "source_hashes":{name:digest(folder/name) for name in ("contacts.json","controller_audit.json","scene.xml")}})
    value = {"scope":"Offline engine contact diagnosis after all online decisions and evidence were frozen. Contact labels never determine or revise the observed stop statuses.",
        "sampling":"Named contact rows are recorded every 0.04 simulated seconds; global maxima in controller_audit include every 0.002 second physics step. Normal-force sums add all sampled contact points and are not calibrated measured sensor forces.",
        "collision_limitation":"The contacted part and fixture shapes use mesh collision geometry. Its convex representation treats each section as a solid outer shape and fills the CAD holes. The simulated gear shaft or screw can therefore hit a solid surface where the real hole should be. Gear meshing and nut thread engagement are not modeled. A contact is not verified engagement.",
        "parts":parts}
    save(run/"offline_contact_diagnostics.json",value)
    return value


if __name__ == "__main__":
    run = Path(sys.argv[1])
    phases,contacts = initial_phase_diagnostics(run),contact_diagnostics(run)
    print(json.dumps({"parts":len(phases["parts"]),"target_board_contact_frames":{
        row["part"]:row["sampled_target_board_contact_frames"] for row in contacts["parts"]}},indent=2))
