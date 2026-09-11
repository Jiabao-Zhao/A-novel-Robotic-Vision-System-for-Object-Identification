import json
import re
from pathlib import Path

from CADPointCloudRegistration import CADPointCloudRegistration
from cad_object_association import associate_cad_to_candidates, raw_observed_cloud_path
from helper_function import CADRetrieval
from LLM_planner import plan_from_outputs
from point_cloud_localization import PointCloudLocalization


OPEN_3D_VISUALIZATION = False
USE_SAVED_RAW_CAPTURE = False
OUTPUT_ROBOT_BASE_POSE = True
RUN_LLM_PLANNER = False
RUN_CAD_REGISTRATION = False  # Experiments save rankings and stop after association.
USER_TEXT = "put the white gear on top of the red block"
# Semantic target descriptions are currently supplied by the experiment caller.
# Extracting them from USER_TEXT is outside the CAD association stage.
TARGET_DESCRIPTIONS = ("white gear", "red block")
# Supply a complete {target_description: known_cad_id} mapping for experiments.
# None retains text retrieval as a separate baseline; never infer ground truth here.
TARGET_CAD_IDS = None
SAVED_RGB_PATH = Path("outputs/physical/raw/RGB.png")
SAVED_DEPTH_PATH = Path("outputs/physical/raw/depth_data.npz")
CAD_LIBRARY_PATH = Path("CAD/cad_library.json")
ASSOCIATION_OUTPUT_DIR = Path("outputs/physical/cad_association")


def run_pipeline():
    localizer = PointCloudLocalization()

    if USE_SAVED_RAW_CAPTURE:
        return localizer.run_from_saved_raw(
            rgb_path=SAVED_RGB_PATH,
            depth_path=SAVED_DEPTH_PATH,
            visualize=OPEN_3D_VISUALIZATION,
        )

    return localizer.run(visualize=OPEN_3D_VISUALIZATION)


def build_local_embedding_function(texts):
    vocabulary = sorted(
        {
            token
            for text in texts
            for token in re.findall(r"[a-z0-9]+", str(text).lower())
        }
    )

    def embed(text):
        tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        return [float(tokens.count(token)) for token in vocabulary]

    return embed


def resolved_associations_from_vlm(vlm_path):
    payload = json.loads(Path(vlm_path).read_text(encoding="utf-8"))
    associations = payload.get("associations")
    if not isinstance(associations, list) or not associations:
        raise ValueError("VLM output does not contain semantic association results.")
    resolved_states = {
        "vlm_accepted",
        "human_confirmed",
        "human_corrected",
        "target_not_present",
    }
    unresolved = [
        item
        for item in associations
        if item.get("resolution") not in resolved_states
    ]
    if unresolved:
        for item in unresolved:
            print(
                f"Target {item.get('target_description')!r} requires human "
                f"clarification; proposal={item.get('vlm_object_id')}, "
                f"score={item.get('association_score')}, "
                f"threshold={item.get('threshold')}."
            )
            print(f"Visual prompt: {payload.get('visual_prompt_path')}")
        raise SystemExit("CAD retrieval stopped until semantic association is resolved.")
    return associations


def observed_cloud_from_localization(localization_payload, object_id):
    payload = localization_payload
    for item in payload["objects"]:
        if item["object_id"] == object_id:
            return raw_observed_cloud_path(item)
    raise ValueError(f"Selected object_id was not found in localization output: {object_id}")


def retrieve_cad_model(target_description, cad_id=None):
    retriever = CADRetrieval()
    cad_models = CADRetrieval.load_library(CAD_LIBRARY_PATH)
    if cad_id is not None:
        matches = [model for model in cad_models if str(model.cad_id) == str(cad_id)]
        if len(matches) != 1:
            raise ValueError(f"Known CAD ID must identify exactly one library model: {cad_id!r}")
        model = matches[0]
        return model, {"selected_cad_id": str(model.cad_id), "selected_cad_name": model.cad_name,
                       "selected_file_path": model.file_path, "selection_method": "known_cad_id",
                       "score": None}
    model_texts = [retriever.build_model_text(model) for model in cad_models]
    embedding_function = build_local_embedding_function([target_description] + model_texts)
    model, result = retriever.retrieve(
        classification_text=target_description,
        cad_models=cad_models,
        embedding_function=embedding_function,
    )
    result["selection_method"] = "text_retrieval"
    return model, result


def cad_request_from_association(association):
    target_description = str(association.get("target_description") or "").strip()
    if not target_description:
        raise ValueError("Semantic association is missing target_description.")
    final_object_id = association.get("final_object_id")
    if final_object_id is None:
        raise LookupError(
            f"Target {target_description!r} is not present; CAD retrieval stopped."
        )
    return {
        "target_description": target_description,
        "object_id": str(final_object_id),
    }


def association_conflicts(associations):
    """Flag independent target queries selecting the same observed object."""
    by_object = {}
    for item in associations:
        object_id = item.get("selected_object_id")
        if object_id is not None:
            by_object.setdefault(object_id, []).append({
                "target_description": item["target_description"], "cad_id": item["cad_id"],
            })
    return [{"object_id": object_id, "targets": targets}
            for object_id, targets in sorted(by_object.items()) if len(targets) > 1]


def associate_targets_with_cad(target_descriptions, localization_payload, rgb_path,
                               plane_model=None, output_dir=ASSOCIATION_OUTPUT_DIR, *, known_cad_ids=None):
    """Fix target -> CAD before observing any pair scores; reuse scene features."""
    targets = list(target_descriptions)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Provide nonempty, unique target descriptions for the association set.")
    if known_cad_ids is not None:
        if set(known_cad_ids) != set(targets) or any(value is None for value in known_cad_ids.values()):
            raise ValueError("known_cad_ids must supply one explicit CAD ID for every target description.")
        retrieved_cads = {target: retrieve_cad_model(target, cad_id=known_cad_ids[target]) for target in targets}
    else:
        retrieved_cads = {target: retrieve_cad_model(target) for target in targets}
    associations, scene_cache = [], {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for target, (cad_model, cad_result) in retrieved_cads.items():
        result = associate_cad_to_candidates(
            cad_model, localization_payload, rgb_path, plane_model,
            target_description=target, scene_cache=scene_cache,
        )
        result["cad_retrieval"] = cad_result
        associations.append(result)
    conflicts = association_conflicts(associations)
    set_resolution = "conflicting" if conflicts else "ranking_only"
    for index, result in enumerate(associations, 1):
        result["association_set_resolution"] = set_resolution
        result["conflicts"] = conflicts
        if conflicts:
            # Retain independent predictions for evaluation; block downstream handoff.
            result["final_object_id"] = None
        result_path = output_dir / f"target_{index:03d}.json"
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(f"CAD association: {result['target_description']} -> {result['cad_id']} -> "
              f"{result['selected_object_id']} ({result['resolution']}); saved {result_path}")
    if conflicts:
        print(f"Conflicting association set: {json.dumps(conflicts)}")
    # Compact adapter for the planner's existing associations/final_object_id contract.
    summary_path = output_dir / "semantic_associations.json"
    summary = {"resolution": set_resolution, "conflicts": conflicts,
               "associations": [{key: item[key] for key in (
        "target_description", "cad_id", "cad_path", "selected_object_id", "final_object_id", "resolution",
        "association_set_resolution",
    )} for item in associations]}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return associations, retrieved_cads, summary_path


def register_resolved_associations(
    associations,
    localization_payload,
    plane_model,
    cad_retrieval=None,
    registrar=None,
    registration_root=Path("outputs/physical/registered_point_cloud"),
    retrieved_cads=None,
):
    if association_conflicts(associations) or any(item.get("association_set_resolution") == "conflicting" for item in associations):
        raise ValueError("Conflicting association set cannot proceed to CAD registration.")
    requests = [cad_request_from_association(item) for item in associations]
    registration_records = []
    registration_root = Path(registration_root)
    multi_object = len(requests) > 1
    cad_retrieval = retrieve_cad_model if cad_retrieval is None else cad_retrieval
    registrar = CADPointCloudRegistration() if registrar is None else registrar

    for association, request in zip(associations, requests):
        object_id = request["object_id"]
        target_description = request["target_description"]
        observed_cloud_path = observed_cloud_from_localization(localization_payload, object_id)
        if retrieved_cads is not None:
            cad_model, cad_result = retrieved_cads[target_description]
            if (str(cad_model.cad_id) != association["cad_id"]
                    or str(cad_model.file_path) != association["cad_path"]):
                raise ValueError("CAD identity changed between association and registration.")
            if (association["resolution"] != "ranking_only"
                    or association["selected_object_id"] != object_id):
                raise ValueError("CAD association is unresolved or has inconsistent object IDs.")
        else:
            if "cad_id" in association:
                raise ValueError("CAD association requires its original retrieved_cads handoff.")
            # Compatibility for the separately invoked VLM experimental baseline.
            cad_model, cad_result = cad_retrieval(target_description)
        cad_id = str(cad_model.cad_id) if retrieved_cads is not None else cad_result.get("selected_cad_id")
        output_dir = registration_root / object_id if multi_object else registration_root

        print(f"Resolved target: {target_description} -> {object_id}")
        print(f"Selected CAD model: {cad_result['selected_cad_name']}")
        print(f"Selected CAD file: {cad_result['selected_file_path']}")

        registration_result = registrar.run(
            cad_path=cad_model.file_path,
            observed_cloud_path=observed_cloud_path,
            plane_model=plane_model,
            cad_metadata={"cad_id": cad_id,
                          "target_description": target_description, "object_id": object_id},
            output_dir=output_dir,
        )
        print(f"Saved aligned CAD point cloud: {registration_result['aligned_cad_cloud_path']}")
        print(f"Saved augmented point cloud: {registration_result['augmented_cloud_path']}")
        print(f"Saved registration JSON: {registration_result['result_path']}")

        registration_records.append(
            {
                "object_id": object_id,
                "target_description": target_description,
                "cad_id": cad_id,
                "cad_retrieval": cad_result,
                "registration": registration_result,
            }
        )

    registration_root.mkdir(parents=True, exist_ok=True)
    summary_path = registration_root / "cad_registration_summary.json"
    summary_path.write_text(json.dumps(registration_records, indent=2), encoding="utf-8")
    print(f"Saved registration summary: {summary_path}")
    return registration_records


def output_robot_base_pose():
    from robot_controller import RTDECommander, RTDEStateFeedback

    state = None
    robot = None
    try:
        state = RTDEStateFeedback()
        robot = RTDECommander(state)
        pose_result = robot.output_object_pose_base()
    finally:
        if robot is not None:
            robot.disconnect()
        if state is not None:
            state.stop()

    x, y, z, roll, pitch, yaw = pose_result["object_pose_base_xyz_rpy_mm_deg"]
    rx, ry, rz = pose_result["tcp_pick_rotvec_base_rad"]
    print(
        "Object pose in robot base frame: "
        f"x={x:.3f} mm, y={y:.3f} mm, z={z:.3f} mm, "
        f"roll={roll:.3f} deg, pitch={pitch:.3f} deg, yaw={yaw:.3f} deg"
    )
    print(
        "Top-down UR pick orientation: "
        f"rx={rx:.6f} rad, ry={ry:.6f} rad, rz={rz:.6f} rad"
    )
    print("Saved robot-base pose JSON: outputs/physical/robot_pose/object_pose_base.json")
    return pose_result


def output_llm_plan(association_path=ASSOCIATION_OUTPUT_DIR / "semantic_associations.json"):
    planner_path = plan_from_outputs(USER_TEXT, vlm_path=association_path)
    print(f"Saved LLM planner result: {planner_path}")
    return planner_path


def main():
    paths = run_pipeline()

    print(f"Saved raw RGB image: {paths['rgb']}")
    if paths.get("unfiltered_depth") is not None:
        print(f"Saved unfiltered depth data: {paths['unfiltered_depth']}")
    print(f"Saved raw depth data: {paths['depth']}")
    print(f"Detected object count: {paths['object_count']}")
    print(f"Saved localization JSON: {paths['localization']}")
    print(f"Saved annotated RGB image: {paths['annotated_rgb']}")
    print(f"Saved workspace point cloud: {paths['workspace_cloud']}")
    print(f"Saved table point cloud: {paths['table_cloud']}")
    print(f"Saved segmented point cloud: {paths['segmented_cloud']}")

    localization_payload = json.loads(Path(paths["localization"]).read_text(encoding="utf-8"))
    plane_model = localization_payload.get("plane_model")

    associations, retrieved_cads, association_path = associate_targets_with_cad(
        TARGET_DESCRIPTIONS, localization_payload, paths["rgb"], plane_model,
        known_cad_ids=TARGET_CAD_IDS,
    )
    if any(item["association_set_resolution"] == "conflicting" for item in associations):
        raise SystemExit("Conflicting target selections; rankings saved. Downstream execution stopped.")
    if not RUN_CAD_REGISTRATION:
        print(f"Association experiment complete; saved rankings and summary: {association_path}")
        return associations
    if any(item["resolution"] != "ranking_only" for item in associations):
        raise SystemExit("No computable candidate ranking; CAD registration stopped.")
    registration_records = register_resolved_associations(
        associations,
        localization_payload,
        plane_model,
        retrieved_cads=retrieved_cads,
    )

    if OUTPUT_ROBOT_BASE_POSE:
        if len(registration_records) == 1:
            output_robot_base_pose()
        else:
            print("Skipped robot-base pose output because multiple objects were registered.")

    if RUN_LLM_PLANNER:
        if len(registration_records) == 1:
            output_llm_plan(association_path)
        else:
            print("Skipped LLM planner because multiple registered objects need role-aware planning.")

if __name__ == "__main__":
    main()
