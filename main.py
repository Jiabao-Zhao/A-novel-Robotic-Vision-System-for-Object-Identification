import json
import re
from pathlib import Path

from CADPointCloudRegistration import CADPointCloudRegistration
from helper_function import CADRetrieval
from point_cloud_localization import PointCloudLocalization
from vlm_module import classify_from_localization


OPEN_3D_VISUALIZATION = False
USE_SAVED_RAW_CAPTURE = False
USER_TEXT = "find the cable shark device"
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


def selected_object_from_vlm(vlm_path):
    payload = json.loads(Path(vlm_path).read_text(encoding="utf-8"))
    result = payload["result"]
    object_id = result["object_id"]
    object_type = result["object_type"]
    if object_id is None or object_type is None:
        raise ValueError("VLM did not select a visible object for CAD registration.")
    return object_id, object_type


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

    object_id, object_type = selected_object_from_vlm(vlm_path)
    observed_cloud_path = observed_cloud_from_localization(localization_payload, object_id)
    cad_model, cad_result = retrieve_cad_model(object_type)
    print(f"Selected object: {object_id} ({object_type})")
    print(f"Selected CAD model: {cad_result['selected_cad_name']}")
    print(f"Selected CAD file: {cad_result['selected_file_path']}")

    registration_result = CADPointCloudRegistration().run(
        cad_path=cad_model.file_path,
        observed_cloud_path=observed_cloud_path,
        plane_model=plane_model,
    )
    print(f"Saved aligned CAD point cloud: {registration_result['aligned_cad_cloud_path']}")
    print(f"Saved augmented point cloud: {registration_result['augmented_cloud_path']}")
    print(f"Saved registration JSON: {registration_result['result_path']}")


if __name__ == "__main__":
    main()
