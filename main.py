import json
import re
from pathlib import Path

from CADPointCloudRegistration import CADPointCloudRegistration
from helper_function import CADRetrieval
from LLM_planner import plan_from_outputs
from point_cloud_localization import PointCloudLocalization
from vlm_module import classify_from_localization


OPEN_3D_VISUALIZATION = False
USE_SAVED_RAW_CAPTURE = False
OUTPUT_ROBOT_BASE_POSE = True
RUN_LLM_PLANNER = False
INTERACTIVE_CLARIFICATION = False
USER_TEXT = "put the white gear on top of the red block"
SAVED_RGB_PATH = Path("output/raw/RGB.png")
SAVED_DEPTH_PATH = Path("output/raw/depth_data.npz")
CAD_LIBRARY_PATH = Path("CAD/cad_library.json")


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


def selected_objects_from_vlm(vlm_path):
    payload = json.loads(Path(vlm_path).read_text(encoding="utf-8"))
    result = payload["normalized_result"]

    if result["needs_human_clarification"]:
        print("VLM requires human clarification.")
        print(f"Clarification reason: {result.get('clarification_reason')}")
        print(f"Clarification question: {result.get('clarification_question')}")
        print("Object evaluations:")
        for evaluation in result.get("object_evaluations", []):
            print(
                "  "
                f"{evaluation.get('object_id')}: "
                f"target_match={evaluation.get('target_match')}, "
                f"predicted_type={evaluation.get('predicted_type')}, "
                f"spatial_description={evaluation.get('spatial_description')}"
            )
            print(f"    visual_evidence: {evaluation.get('visual_evidence')}")
            print(
                "    missing_or_uncertain_cues: "
                f"{evaluation.get('missing_or_uncertain_cues')}"
            )

        if not INTERACTIVE_CLARIFICATION:
            raise SystemExit(0)

        valid_object_ids = {
            evaluation.get("object_id")
            for evaluation in result.get("object_evaluations", [])
        }
        chosen_text = input(
            "Type object_id values to use, separated by commas, or press Enter to stop: "
        ).strip()
        if not chosen_text:
            raise SystemExit(0)
        chosen_object_ids = [
            item.strip()
            for item in chosen_text.split(",")
            if item.strip()
        ]
        invalid_object_ids = [
            object_id
            for object_id in chosen_object_ids
            if object_id not in valid_object_ids
        ]
        if invalid_object_ids:
            raise ValueError(
                "Invalid object_id from clarification: "
                f"{', '.join(invalid_object_ids)}"
            )
        evaluations_by_id = {
            evaluation.get("object_id"): evaluation
            for evaluation in result.get("object_evaluations", [])
        }
        return [
            selected_object_record(evaluations_by_id[object_id])
            for object_id in chosen_object_ids
        ]

    selected_objects = result.get("selected_objects") or []
    if selected_objects:
        return selected_objects

    object_id = result["selected_object_id"]
    object_type = result["selected_object_type"]
    if object_id is None or object_type is None:
        raise ValueError("VLM did not select a visible object for CAD registration.")
    return [
        {
            "object_id": object_id,
            "object_type": object_type,
            "object_class": object_type,
            "target_match": "match",
            "instruction_role": None,
        }
    ]


def selected_object_record(evaluation):
    return {
        "object_id": evaluation.get("object_id"),
        "object_type": evaluation.get("predicted_type") or USER_TEXT,
        "object_class": evaluation.get("predicted_type") or USER_TEXT,
        "target_match": evaluation.get("target_match"),
        "instruction_role": evaluation.get("instruction_role"),
        "spatial_description": evaluation.get("spatial_description"),
    }


def observed_cloud_from_localization(localization_payload, object_id):
    payload = localization_payload
    for item in payload["objects"]:
        if item["object_id"] == object_id:
            cloud_path = Path(item["pointcloud_path"])
            if cloud_path.stem.endswith("_downsampled"):
                cloud_path = cloud_path.with_name(
                    cloud_path.stem.removesuffix("_downsampled") + cloud_path.suffix
                )
            return cloud_path
    raise ValueError(f"Selected object_id was not found in localization output: {object_id}")


def retrieve_cad_model(classification_text):
    retriever = CADRetrieval()
    cad_models = CADRetrieval.load_library(CAD_LIBRARY_PATH)
    model_texts = [retriever.build_model_text(model) for model in cad_models]
    embedding_function = build_local_embedding_function([classification_text] + model_texts)
    return retriever.retrieve(
        classification_text=classification_text,
        cad_models=cad_models,
        embedding_function=embedding_function,
    )


def register_selected_objects(selected_objects, localization_payload, plane_model):
    registration_records = []
    registration_root = Path("output/registered_point_cloud")
    multi_object = len(selected_objects) > 1
    registrar = CADPointCloudRegistration()

    for selected_object in selected_objects:
        object_id = selected_object["object_id"]
        object_type = selected_object["object_type"]
        observed_cloud_path = observed_cloud_from_localization(localization_payload, object_id)
        cad_model, cad_result = retrieve_cad_model(object_type)
        output_dir = registration_root / object_id if multi_object else registration_root

        print(f"Selected object: {object_id} ({object_type})")
        if selected_object.get("instruction_role"):
            print(f"Instruction role: {selected_object['instruction_role']}")
        print(f"Selected CAD model: {cad_result['selected_cad_name']}")
        print(f"Selected CAD file: {cad_result['selected_file_path']}")

        registration_result = registrar.run(
            cad_path=cad_model.file_path,
            observed_cloud_path=observed_cloud_path,
            plane_model=plane_model,
            output_dir=output_dir,
        )
        print(f"Saved aligned CAD point cloud: {registration_result['aligned_cad_cloud_path']}")
        print(f"Saved augmented point cloud: {registration_result['augmented_cloud_path']}")
        print(f"Saved registration JSON: {registration_result['result_path']}")

        registration_records.append(
            {
                "object_id": object_id,
                "object_type": object_type,
                "object_class": selected_object.get("object_class"),
                "instruction_role": selected_object.get("instruction_role"),
                "target_match": selected_object.get("target_match"),
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
    print("Saved robot-base pose JSON: output/robot_pose/object_pose_base.json")
    return pose_result


def output_llm_plan():
    planner_path = plan_from_outputs(USER_TEXT)
    print(f"Saved LLM planner result: {planner_path}")
    return planner_path


def main():
    paths = run_pipeline()

    print(f"Saved raw RGB image: {paths['rgb']}")
    print(f"Saved raw depth data: {paths['depth']}")
    print(f"Detected object count: {paths['object_count']}")
    print(f"Saved localization JSON: {paths['localization']}")
    print(f"Saved annotated RGB image: {paths['annotated_rgb']}")
    print(f"Saved workspace point cloud: {paths['workspace_cloud']}")
    print(f"Saved table point cloud: {paths['table_cloud']}")
    print(f"Saved segmented point cloud: {paths['segmented_cloud']}")

    localization_payload = json.loads(Path(paths["localization"]).read_text(encoding="utf-8"))
    plane_model = localization_payload.get("plane_model")

    vlm_path = classify_from_localization(
        user_text=USER_TEXT,
        image_path=paths["annotated_rgb"],
        localization_path=paths["localization"],
    )
    print(f"Saved VLM result: {vlm_path}")

    selected_objects = selected_objects_from_vlm(vlm_path)
    registration_records = register_selected_objects(
        selected_objects,
        localization_payload,
        plane_model,
    )

    if OUTPUT_ROBOT_BASE_POSE:
        if len(registration_records) == 1:
            output_robot_base_pose()
        else:
            print("Skipped robot-base pose output because multiple objects were registered.")

    if RUN_LLM_PLANNER:
        if len(registration_records) == 1:
            output_llm_plan()
        else:
            print("Skipped LLM planner because multiple registered objects need role-aware planning.")

if __name__ == "__main__":
    main()
