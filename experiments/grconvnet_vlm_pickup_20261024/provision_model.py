"""Fetch pinned official GR-ConvNet code/weights for the authorized local trial.

No training or environment mutation. Missing image libraries may be installed
separately into the experiment's local dependency directory, never the venv.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VENDOR = HERE / "vendor"
SOURCE = VENDOR / "grconvnet_source"
MODEL_DIR = ROOT / "outputs/grconvnet_vlm_pickup_20261024/model"
REPO = "skumra/robotic-grasping"
CHECKPOINT = "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98"
FILES = ("LICENSE", "inference/models/grconvnet3.py", "inference/models/grasp_model.py",
         "inference/post_process.py", "utils/data/camera_data.py",
         "utils/dataset_processing/image.py", "utils/dataset_processing/grasp.py")


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "local-grconvnet-trial"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def convert_trusted_checkpoint():
    """Convert the hash-verified legacy official Module pickle to safe tensors.

    This is an explicit provisioning operation, not an inference/test side
    effect. PyTorch's weights-only loader cannot decode this old storage format.
    Only the pinned official artifact and pinned architecture are accepted.
    """
    import sys
    import torch
    sys.path.insert(0, str(SOURCE))
    from inference.models.grconvnet3 import GenerativeResnet
    provenance_path = VENDOR / "provenance.json"
    lock = json.loads(provenance_path.read_text())
    original = ROOT / lock["checkpoint"]["relative_path"]
    if sha(original.read_bytes()) != lock["checkpoint"]["sha256"]:
        raise ValueError("Official checkpoint changed before conversion")
    model = torch.load(original, map_location="cpu", weights_only=False)
    if not isinstance(model, GenerativeResnet):
        raise ValueError("Unexpected official model architecture")
    safe_path = MODEL_DIR / "cornell_rgbd_grconvnet3_ch32_epoch19_state_dict.pt"
    torch.save(model.state_dict(), safe_path)
    architecture = {"input_channels": model.conv1.in_channels,
                    "output_channels": model.pos_output.out_channels,
                    "channel_size": model.conv1.out_channels,
                    "dropout": bool(model.dropout), "prob": float(model.dropout_pos.p)}
    lock["safe_state_dict"] = {"relative_path": str(safe_path.relative_to(ROOT)),
                               "sha256": sha(safe_path.read_bytes()), "architecture": architecture,
                               "conversion": "Explicit trusted official Module pickle load, CPU; then state_dict saved. Inference loads only this state_dict with weights_only=True.",
                               "reason": "PyTorch weights-only loader rejected the legacy checkpoint's persistent storage format."}
    provenance_path.write_text(json.dumps(lock, indent=2)+"\n")
    print(json.dumps(lock["safe_state_dict"], indent=2))


def main():
    VENDOR.mkdir(parents=True, exist_ok=True)
    lock_path = VENDOR / "provenance.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        commit = lock["upstream_commit"]
    else:
        commit = json.loads(fetch(f"https://api.github.com/repos/{REPO}/commits/master"))["sha"]
        lock = {"upstream_repository": f"https://github.com/{REPO}",
                "upstream_commit": commit, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Official Cornell RGB-D pretrained checkpoint referenced by released OWG; no training.",
                "paper_difference": "Published OWG describes Jacquard-pretrained proposals; its released PyBullet code defaults to this Cornell RGB-D checkpoint.",
                "source_files": {}}
    for relative in FILES:
        url = f"https://raw.githubusercontent.com/{REPO}/{commit}/{relative}"
        destination = SOURCE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = destination.read_bytes() if destination.exists() else fetch(url)
        destination.write_bytes(data)
        lock["source_files"][relative] = {"url": url, "sha256": sha(data), "size_bytes": len(data)}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODEL_DIR / "cornell_rgbd_grconvnet3_ch32_epoch19.pt"
    url = f"https://raw.githubusercontent.com/{REPO}/{commit}/{CHECKPOINT}"
    data = destination.read_bytes() if destination.exists() else fetch(url)
    destination.write_bytes(data)
    lock["checkpoint"] = {"url": url, "relative_path": str(destination.relative_to(ROOT)),
                          "sha256": sha(data), "size_bytes": len(data)}
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"commit": commit, "checkpoint": str(destination),
                      "bytes": len(data), "sha256": sha(data)}, indent=2))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--convert-trusted-checkpoint"]:
        convert_trusted_checkpoint()
    else:
        main()
