"""Audit existing task-board targets without inventing final mating transforms.

Run with the existing Python + NumPy environment. Geometry is in metres. The
generated JSON is an input/evidence audit, not an assembly success report.
"""

import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / "outputs/simulation/nist_task_board_1/assets"
TARGETS = (
    "Gear_Large", "M16_Hex_Nut", "RGOCG16-50_16mm", "KET16_Square_16mm",
    "Waterproof_Male", "USB_Male", "DSUB_Male", "BNC_Male",
    "RGOCG12-50_12mm", "RGOCG8-50_8mm", "KET12_Square_12mm", "KET8_Square_8mm",
)
PILOT_WINDOWS = Path("C:/Users/Jiabao Zhao/Documents/Codex/2026-06-02/can-you-add-this-skill-to/round-pin-assembly-pilot")
PILOT_WSL = Path("/mnt/c/Users/Jiabao Zhao/Documents/Codex/2026-06-02/can-you-add-this-skill-to/round-pin-assembly-pilot")
PILOT = PILOT_WINDOWS if PILOT_WINDOWS.is_dir() else PILOT_WSL
WORKSPACE = PILOT.parent / "nist-taskboard-workspace"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_mesh(path):
    """Read the existing binary STL; retain its metric mesh frame exactly."""
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + count * 50:
        raise ValueError(f"Expected a binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
                      ("attribute", "<u2")])
    records = np.frombuffer(raw, dtype=dtype, offset=84, count=count)
    vertices, inverse = np.unique(records["vertices"].reshape(-1, 3), axis=0, return_inverse=True)
    return vertices.astype(float), inverse.reshape(-1, 3)


def top_boundaries(vertices, triangles):
    top = float(vertices[:, 2].max())
    face = triangles[np.all(np.isclose(vertices[triangles, 2], top, atol=1e-7), axis=1)]
    counts = Counter(tuple(sorted((int(a), int(b))))
                     for tri in face for a, b in zip(tri, np.roll(tri, -1)))
    graph = defaultdict(set)
    for (a, b), count in counts.items():
        if count == 1:
            graph[a].add(b)
            graph[b].add(a)
    if not all(len(neighbours) == 2 for neighbours in graph.values()):
        raise ValueError("Nonmanifold plate top boundary")
    unseen = set(graph)
    boundaries = []
    while unseen:
        stack, component = [unseen.pop()], []
        while stack:
            index = stack.pop()
            component.append(index)
            for neighbour in graph[index]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        xy = vertices[component, :2]
        extent = np.ptp(xy, axis=0)
        center = (xy.min(axis=0) + xy.max(axis=0)) / 2
        fit = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))],
                              np.sum(xy * xy, axis=1), rcond=None)[0]
        radii = np.linalg.norm(xy - fit[:2], axis=1)
        circular = len(xy) >= 12 and np.ptp(radii) < .00005
        record = {"boundary_vertices": len(xy), "center_board_xy_m": center.tolist(),
                  "extent_xy_m": extent.tolist(), "type": "circular" if circular else "other",
                  "top_boundary_xy_m": xy.tolist()}
        if circular:
            angles = np.arctan2(xy[:, 1] - fit[1], xy[:, 0] - fit[0])
            polygon = xy[np.argsort(angles)]
            edges = np.roll(polygon, -1, axis=0) - polygon
            rel = fit[:2] - polygon
            apothems = np.abs(edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]) / np.linalg.norm(edges, axis=1)
            record.update(center_board_xy_m=fit[:2].tolist(), diameter_m=float(np.mean(radii) * 2),
                          minimum_inscribed_diameter_m=float(apothems.min() * 2))
        boundaries.append(record)
    circles = sorted((b for b in boundaries if b["type"] == "circular"),
                     key=lambda b: (b["diameter_m"], b["center_board_xy_m"]))
    for index, boundary in enumerate(circles):
        boundary["audit_hole_id"] = f"opening_{index:02d}"
    other = [b for b in boundaries if b["type"] != "circular"]
    return circles, other


def transform(position, yaw_deg=0.):
    angle = np.deg2rad(yaw_deg)
    result = np.eye(4)
    result[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    result[:3, 3] = position
    return result.tolist()


def source_constants():
    tree = ast.parse((ROOT / "simulation/nist_task_board_1.py").read_text())
    result = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name == "FIXTURES":
            result[name] = ast.literal_eval(node.value)
        elif name in ("BOARD_CENTER_XY_M", "BOARD_SIZE_M"):
            result[name] = ast.literal_eval(node.value.args[0])
        elif name in ("FLOOR_Z_M",):
            result[name] = ast.literal_eval(node.value)
    return result


def nominal_goals(parts, holes, other, fixture_records, dimensions, board_top, source_scene):
    """Explicit synthetic goals authorized for this isolated simulation trial."""
    center = np.asarray(source_scene["board"]["center_xy_m"])
    result = {}
    fixture_map = {"M12_Hex_Nut": 6, "M16_Hex_Nut": 7, "Waterproof_Male": 14,
                   "USB_Male": 9, "DSUB_Male": 13, "BNC_Male": 8}
    for name in parts:
        height = dimensions[name][2]
        yaw = 0.
        detail = {}
        if name.startswith("RGOCG"):
            size = int(name.split("-")[0][5:])
            hole = holes[{8: "opening_23", 12: "opening_27", 16: "opening_28"}[size]]
            xy = center + hole["center_board_xy_m"]
            z = board_top + height / 2 - .012
            kind = "nominal_round_pin_guided_insertion_12mm"
            detail = {"opening_id": hole["audit_hole_id"], "guided_depth_m": .012,
                      "goal_assumption": "Vertical CAD Z axis, 12 mm lower-end depth below plate top; round-pin yaw set to zero."}
            if size == 12:
                detail["destination_limitation"] = "Larger opening_27 clearance demonstration; not a verified official 12 mm destination."
        elif name.startswith("KET"):
            extent_xy = np.asarray(dimensions[name][:2])
            matches = [h for h in other if np.all((np.sort(h["extent_xy_m"]) - np.sort(extent_xy) > 0)
                       & (np.sort(h["extent_xy_m"]) - np.sort(extent_xy) < .0005))]
            if len(matches) != 1:
                result[name] = {"world_T_cad_goal": None, "status": "unavailable_no_unique_rectangular_opening"}
                continue
            hole = matches[0]
            xy = center + hole["center_board_xy_m"]
            z = board_top + height / 2 - .012
            yaw = 0. if np.argmax(hole["extent_xy_m"]) == np.argmax(extent_xy) else 90.
            kind = "nominal_rectangular_pin_guided_insertion_12mm"
            detail = {"opening_extent_xy_m": hole["extent_xy_m"], "guided_depth_m": .012,
                      "goal_assumption": "Dimension-matched rectangular opening, 12 mm lower-end depth; yaw aligns long cross-section axes. Taper and insertion fit unvalidated."}
        elif name.startswith("Gear_"):
            shaft_index = {"Gear_Large": 1, "Gear_Medium": 2, "Gear_Small": 3}[name]
            fixture = fixture_records[shaft_index]
            xy = center + fixture["center_xy_relative_to_board_center_m"]
            gear_plate_top = fixture_records[0]["world_z_in_current_scene_m"] + dimensions["Gear_Plate"][2] / 2
            z = gear_plate_top + height / 2
            kind = "nominal_gear_on_shaft"
            detail = {"fixture_index": shaft_index,
                      "goal_assumption": "Large/medium/small assigned to consecutive left/middle/right shaft anchors; gear bottom at gear-plate top; yaw zero. Shaft bore and tooth phase unvalidated."}
        else:
            fixture = fixture_records[fixture_map[name]]
            xy = center + fixture["center_xy_relative_to_board_center_m"]
            yaw = fixture["yaw_deg"]
            if "Hex_Nut" in name:
                z = board_top + height / 2
                kind = "nominal_nut_at_plate_top"
                assumption = "Nut underside at plate top, centered on matching screw, yaw zero. No thread engagement or tightening is represented."
            else:
                fixture_top = fixture["world_z_in_current_scene_m"] + dimensions[fixture["name"]][2] / 2
                z = fixture_top + height / 2
                kind = "nominal_connector_alignment_at_fixture_top"
                assumption = "Male lower bounding plane touches female upper bounding plane; yaw follows fixture. This is a nominal alignment target with zero assigned engagement, not a validated connector mating transform."
            detail = {"fixture_index": fixture["fixture_index"], "goal_assumption": assumption}
        result[name] = {"world_T_cad_goal": transform([*xy, z], yaw),
            "status": "nominal_simulation_assumption", "goal_kind": kind,
            "goal_frame": "existing objects-centered scene world; metres",
            "official_final_assembly_pose": False, "actual_insertion_validated": False, **detail}
    return result


def main():
    constants = source_constants()
    provenance = json.loads((ASSETS / "provenance.json").read_text())
    plate = ASSETS / "meshes/GMC_Laser_Plate_Virtual_m.stl"
    vertices, triangles = native_mesh(plate)
    circles, other = top_boundaries(vertices, triangles)
    board_height = float(np.ptp(vertices[:, 2]))
    board_top = .020 + board_height
    fixture_records = []
    for index, (name, x_mm, y_mm, base_mm, yaw) in enumerate(constants["FIXTURES"]):
        xy = ((np.array([x_mm, y_mm]) - 192) * .001).tolist()
        center_z = board_top + base_mm * .001 + provenance["dimensions_m"][name][2] / 2
        fixture_records.append({"fixture_index": index, "name": name,
            "center_xy_relative_to_board_center_m": xy,
            "world_z_in_current_scene_m": center_z, "yaw_deg": yaw,
            "center_is_mating_pose": False,
            "mounting_model": "existing approximate fixture mounting; not validated assembly geometry"})
    prior = json.loads((PILOT / "geometry_preflight.json").read_text()) if (PILOT / "geometry_preflight.json").is_file() else None
    if prior:
        for item, saved in zip(circles, prior["circular_openings"]):
            assert item["audit_hole_id"] == saved["audit_hole_id"]
            np.testing.assert_allclose(item["center_board_xy_m"], saved["center_board_m"], atol=1e-8)
    holes = {item["audit_hole_id"]: item for item in circles}
    fixture_for = {"M16_Hex_Nut": [7], "Waterproof_Male": [14], "USB_Male": [9],
                   "DSUB_Male": [13], "BNC_Male": [8], "Gear_Large": [1, 2, 3]}
    parts = {}
    for name in TARGETS:
        record = {"cad_path": str(ASSETS / f"meshes/{name}_m.stl"),
            "cad_sha256": sha256(ASSETS / f"meshes/{name}_m.stl"),
            "extent_m": provenance["dimensions_m"][name],
            "final_assembly_world_T_cad": None,
            "final_assembly_pose_status": "not_provided_in_existing_metadata",
            "actual_assembly_success_supported_by_current_model": False}
        if name in fixture_for:
            record["known_fixture_anchors"] = [fixture_records[i] for i in fixture_for[name]]
            record["missing_goal_information"] = ["part-to-mating-fixture transform", "final engagement depth",
                                                   "assembly approach and mechanism success predicate"]
            if name == "Gear_Large":
                record["missing_goal_information"].append("authoritative gear-to-shaft assignment and tooth phase")
                record["collision_limitation"] = "Current gear collision does not preserve its shaft bore or tooth meshing."
            elif name == "M16_Hex_Nut":
                record["collision_limitation"] = "Convex nut collision closes its bore; screw/nut threading is not modeled."
            else:
                record["collision_limitation"] = "Convex connector collision closes mating cavities; latches/locking are not modeled."
        elif name.startswith("RGOCG"):
            size = int(name.split("-")[0][5:])
            candidates = [h for h in circles if 0 < h["minimum_inscribed_diameter_m"] - size * .001 < .001]
            record["dimension_compatible_openings"] = [{k: value for k, value in h.items() if k != "top_boundary_xy_m"}
                                                         for h in candidates]
            hole_id = {8: "opening_23", 12: "opening_27", 16: "opening_28"}[size]
            hole = holes[hole_id]
            entry_position = [*hole["center_board_xy_m"], board_top + .025]
            record["prior_controller_demo_opening"] = hole_id
            record["board_center_xy_relative_world_T_cad_at_entry"] = transform(entry_position)
            record["entry_definition"] = "CAD Z axis vertical; 50 mm pin lower end at plate top. This is an entry pose, not the final assembled pose."
            record["experimental_goal_if_12mm_guided_insertion_assumed"] = transform([*entry_position[:2], entry_position[2] - .012])
            record["experimental_goal_authority"] = "Explicit experiment assumption matching prior insertion_demo.py commands; not an official mating-pose specification."
            record["collision_limitation"] = "Round-pin convex hull is suitable for contact, but tolerances/soft contact require independent validation."
            if size == 12:
                record["destination_caveat"] = "Near-size opening_26 is occupied by M12 screw fixture_6. Prior pilot used larger opening_27 only as a clearance-hole demonstration; official intended destination is not established."
            elif size == 8:
                record["destination_caveat"] = "Prior successful contact-driven sequence rotated the board 180 degrees; original-board trials were unsuccessful."
        elif name.startswith("KET"):
            extent_xy = np.asarray(record["extent_m"][:2])
            candidates = []
            for boundary in other:
                extent = np.asarray(boundary["extent_xy_m"])
                if np.all((np.sort(extent) - np.sort(extent_xy) > 0) & (np.sort(extent) - np.sort(extent_xy) < .0005)):
                    candidates.append(boundary)
            record["dimension_compatible_openings"] = candidates
            record["missing_goal_information"] = ["authoritative part-to-hole assignment", "final insertion depth", "orientation/clearance validated across the full plate thickness"]
            if len(candidates) == 1:
                hole = candidates[0]
                yaw = 0. if np.argmax(hole["extent_xy_m"]) == np.argmax(extent_xy) else 90.
                record["candidate_board_center_xy_relative_world_T_cad_at_entry"] = transform([*hole["center_board_xy_m"], board_top + .025], yaw)
                record["entry_definition"] = "Bounding-box dimension match and vertical 50 mm pin; geometry candidate only, not an authorized final assembly pose."
            record["collision_limitation"] = "Plate openings preserved, but native cut taper and rectangular-pin clearance have not been validated for insertion."
        parts[name] = record
    source_scene_path = WORKSPACE / "scene.json"
    source_scene = json.loads(source_scene_path.read_text())
    existing14 = list(source_scene["catalog"])
    # The current twelve-object classification study includes BNC instead of
    # medium/small gears and M12 nut; preserve the distinction in this audit.
    goal_parts = list(dict.fromkeys([*existing14, *TARGETS]))
    goals = nominal_goals(goal_parts, holes, other, fixture_records,
                          provenance["dimensions_m"], board_top, source_scene)
    for name in goal_parts:
        parts.setdefault(name, {"cad_path": str(ASSETS / f"meshes/{name}_m.stl"),
            "cad_sha256": sha256(ASSETS / f"meshes/{name}_m.stl"),
            "extent_m": provenance["dimensions_m"][name], "final_assembly_world_T_cad": None,
            "final_assembly_pose_status": "not_provided_in_existing_metadata",
            "actual_assembly_success_supported_by_current_model": False})
        parts[name]["nominal_simulation_goal"] = goals[name]
        parts[name]["nominal_world_T_cad_goal"] = goals[name]["world_T_cad_goal"]
    sources = [ROOT / "simulation/nist_task_board_1.py", ASSETS / "provenance.json", plate,
               ASSETS / "assembly_instructions.docx", ASSETS / "cad.zip",
               ROOT / "simulation/nist_capture.py", source_scene_path]
    sources.extend(path for path in [PILOT / "geometry_preflight.json", PILOT / "README.md", PILOT / "insertion_demo.py", PILOT / "full_sequence.py"] if path.is_file())
    result = {"schema_version": 1, "purpose": "Audit available assembly-goal evidence for the new runtime grasp-generation experiment.",
        "units": "metres and degrees", "cad_frame": "existing centered metric STL frame, preserved exactly",
        "current_nist12_parts": list(TARGETS), "existing_objects_centered_parts": existing14,
        "part_count": len(parts), "runtime_experiment_part_count": len(existing14),
        "nominal_goals_authorization": "User authorized nominal simulation goals during this new runtime-grasp experiment; these are explicit trial assumptions, not recovered official mating transforms.",
        "nominal_simulation_goals": {name: goals[name] for name in existing14},
        "optional_current12_only_nominal_goals": {name: goals[name] for name in TARGETS if name not in existing14},
        "existing_scene_board_center_world_xy_m": constants["BOARD_CENTER_XY_M"],
        "prior_controller_scene_board_center_world_xy_m": [-.520, -.275],
        "plate_thickness_m": board_height, "plate_top_world_z_m": board_top,
        "coordinate_note": "Entry/candidate matrices have world-aligned axes, XY relative to the board center, and absolute scene Z. Add the selected scene board-center XY; compose any board yaw about its center explicitly.",
        "collision_scope": {"other_loose_parts": "excluded by user request",
            "robot_self_table": "required", "target_finger_contact": "required to grasp",
            "board_and_mating_fixtures": "must remain physical for any insertion/assembly-success claim; a grasp-only simplified trial may exclude them only with explicit labeling"},
        "official_fabrication_caveat": "The cached NIST assembly instructions require drilling round-peg pilot holes and tapping bolt holes after fabrication. Existing simulation retains the distributed STL openings; no drilling/threading is modeled.",
        "general_limitations": ["No authoritative complete final assembled part-pose map was found in the inspected sources.",
            "Known female/shaft/screw centers are fixture anchors, not male-part mating transforms.",
            "Current NistTaskBoardEnvironment._check_success always returns False.",
            "Current scene uses approximate fixture mounting and convex component collision; it is not an assembly benchmark.",
            "Final pose, contact insertion and release must be assessed separately from grasp candidate availability."],
        "parts": parts, "plate_top_boundaries": {"circular": circles, "other": other},
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in sources]}
    destination = HERE / "goal_audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(destination), "parts": len(parts), "circular_openings": len(circles),
                      "rectangular_or_outer_boundaries": len(other), "authoritative_final_poses": 0}, indent=2))


if __name__ == "__main__":
    main()
