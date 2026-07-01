import base64
import json
from dataclasses import asdict, dataclass
from math import ceil, floor, sqrt
from pathlib import Path

import open3d as o3d

from prompt import _vlm_classification_localization_prompt, _vlm_classification_prompt
from realsense_pointcloud_pose import CALIBRATED_INTRINSICS, PipelineConfig, PoseResult, run_pipeline as run_pose_pipeline


@dataclass(frozen=True)
class Config:
    output_dir: Path = Path("outputs/pointcloud_vlm_test")
    instruction: str = "identify all objects in the workspace, white gear, blue block, red block, green block, cable shark device, usb-connector."
    #instruction: str = "identify all objects in the workspace."
    warmup_frames: int = 30

    openai_model: str = "gpt-5.5"
    openai_detail: str = "high"


CONFIG = Config()


def save_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_rgb_png(path: Path, image_rgb) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_image(str(path), o3d.geometry.Image(image_rgb)):
        raise RuntimeError(f"Failed to write image: {path}")
    return path


def encode_image_data_url(path: Path):
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{image_b64}"


def pose_prompt_payload(result: PoseResult):
    return [result.id, result.box]


def image_tokens_for_request(image_rgb, config: Config):
    height, width = image_rgb.shape[:2]
    detail = config.openai_detail
    patch_budget = 10_000 if detail in {"original", "auto"} else 2_500
    max_dimension = 6_000 if detail in {"original", "auto"} else 2_048

    scale = min(1.0, max_dimension / max(width, height))
    patch_count = ceil(width * scale / 32) * ceil(height * scale / 32)
    if patch_count > patch_budget:
        shrink = sqrt((32 * 32 * patch_budget) / (width * height))
        adjusted = shrink * min(
            floor(width * shrink / 32) / (width * shrink / 32),
            floor(height * shrink / 32) / (height * shrink / 32),
        )
        scale = min(scale, adjusted)

    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    image_tokens = ceil(resized_width / 32) * ceil(resized_height / 32)

    return int(image_tokens)


def call_openai_vlm(image_path: Path, prompt: str, config: Config):
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=config.openai_model,
        messages=[
            {
                "role": "system",
                "content": "You are a robot vision classification module. Follow the requested JSON format exactly.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encode_image_data_url(image_path),
                            "detail": config.openai_detail,
                        },
                    },
                ],
            },
        ],
    )

    return {
        "provider": "openai",
        "model": config.openai_model,
        "image_path": str(image_path),
        "text": response.choices[0].message.content or "",
        "usage": response.usage.model_dump() if response.usage else None,
    }


def usage_summary(result: dict):
    usage = result.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "image_tokens": usage.get("image_tokens"),
        "non_image_prompt_tokens": usage.get("non_image_prompt_tokens"),
    }


def build_prompt_jobs(config: Config, prompt_detections):
    return [
        {
            "name": "classification_only",
            "prompt": _vlm_classification_prompt(config.instruction, prompt_detections),
        },
        {
            "name": "classification_and_localization",
            "prompt": _vlm_classification_localization_prompt(config.instruction),
        },
    ]


def run_pipeline(config: Config = CONFIG):
    pose_config = PipelineConfig(
        output_dir=config.output_dir / "pose",
        warmup_frames=config.warmup_frames,
    )
    pose_run = run_pose_pipeline(pose_config, save_results=False, visualize=False)
    pose_results = [asdict(result) for result in pose_run.results]
    prompt_detections = [pose_prompt_payload(result) for result in pose_run.results]

    original_path = save_rgb_png(config.output_dir / "camera_original.png", pose_run.color_image_rgb)
    image_tokens = image_tokens_for_request(pose_run.color_image_rgb, config)

    vlm_results = []
    for job in build_prompt_jobs(config, prompt_detections):
        result = call_openai_vlm(original_path, job["prompt"], config)
        if result["usage"] is not None:
            result["usage"]["image_tokens"] = image_tokens
            if result["usage"].get("prompt_tokens") is not None:
                result["usage"]["non_image_prompt_tokens"] = result["usage"]["prompt_tokens"] - image_tokens
        print(json.dumps(result["usage"], indent=2))
        result_path = config.output_dir / f"vlm_result_{job['name']}.json"
        save_json(result_path, result)

        vlm_results.append(
            {
                "name": job["name"],
                "result_path": str(result_path),
                "response_text": result["text"],
                "usage": usage_summary(result),
            }
        )

    payload = {
        "instruction": config.instruction,
        "camera_intrinsics": CALIBRATED_INTRINSICS,
        "pose_results": pose_results,
        "prompt_detections": prompt_detections,
        "images": {"original": str(original_path)},
        "vlm_results": vlm_results,
    }

    save_json(config.output_dir / "result.json", payload)
    print(json.dumps(payload["vlm_results"], indent=2))
    return payload


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
