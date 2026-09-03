import json
import re
from pathlib import Path

from CADPointCloudRegistration import CADPointCloudRegistration
from helper_function import CADRetrieval
from LLM_planner import plan_from_outputs
from point_cloud_localization import PointCloudLocalization
from vlm_module import (
    PROVISIONAL_ASSOCIATION_THRESHOLD,
    associate_targets_from_localization,
    console_human_resolver,
)


OPEN_3D_VISUALIZATION = False
USE_SAVED_RAW_CAPTURE = False
OUTPUT_ROBOT_BASE_POSE = True
RUN_LLM_PLANNER = False
INTERACTIVE_CLARIFICATION = True
USER_TEXT = "put the white gear on top of the red block"
TARGET_DESCRIPTIONS = ("white gear", "red block")
SAVED_RGB_PATH = Path("outputs/physical/raw/RGB.png")
SAVED_DEPTH_PATH = Path("outputs/physical/raw/depth_data.npz")
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


def resolved_associations_from_vlm(vlm_path):
    payload = json.loads(Path(vlm_path).read_text(encoding="utf-8"))
    associations = payload.get("associations")
    if not isinstance(associations, list) or not associations:
        raise ValueError("VLM output does not contain semantic association results.")
    unresolved = [
        item
        for item in associations
        if item.get("requires_human_clarification")
    ]
    if unresolved:
        for item in unresolved:
            print(
                f"Target {item.get('target_description')!r} requires human "
                f"clarification; proposal={item.get('vlm_object_id')}, "
                f"score={item.get('association_score')}, "
                f"threshold={item.get('threshold')}."
            )
            print(f"Visual prompt: {item.get('visual_prompt_path')}")
        raise SystemExit("CAD retrieval stopped until semantic association is resolved.")
    return associations


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


def retrieve_cad_model(target_description):
    retriever = CADRetrieval()
    cad_models = CADRetrieval.load_library(CAD_LIBRARY_PATH)
    model_texts = [retriever.build_model_text(model) for model in cad_models]
    embedding_function = build_local_embedding_function([target_description] + model_texts)
    return retriever.retrieve(
        classification_text=target_description,
        cad_models=cad_models,
        embedding_function=embedding_function,
    )


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


def register_resolved_associations(
    associations,
    localization_payload,
    plane_model,
    cad_retrieval=None,
    registrar=None,
    registration_root=Path("outputs/physical/registered_point_cloud"),
):
    requests = [cad_request_from_association(item) for item in associations]
    registration_records = []
    registration_root = Path(registration_root)
    multi_object = len(requests) > 1
    cad_retrieval = retrieve_cad_model if cad_retrieval is None else cad_retrieval
    registrar = CADPointCloudRegistration() if registrar is None else registrar

    for request in requests:
        object_id = request["object_id"]
        target_description = request["target_description"]
        observed_cloud_path = observed_cloud_from_localization(localization_payload, object_id)
        cad_model, cad_result = cad_retrieval(target_description)
        output_dir = registration_root / object_id if multi_object else registration_root

        print(f"Resolved target: {target_description} -> {object_id}")
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
                "target_description": target_description,
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


def output_llm_plan():
    planner_path = plan_from_outputs(USER_TEXT)
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

    vlm_path = associate_targets_from_localization(
        target_descriptions=TARGET_DESCRIPTIONS,
        image_path=paths["annotated_rgb"],
        localization_path=paths["localization"],
        threshold=PROVISIONAL_ASSOCIATION_THRESHOLD,
        human_resolver=(
            console_human_resolver if INTERACTIVE_CLARIFICATION else None
        ),
    )
    print(f"Saved VLM result: {vlm_path}")

    associations = resolved_associations_from_vlm(vlm_path)
    registration_records = register_resolved_associations(
        associations,
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
