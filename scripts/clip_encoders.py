"""Frozen local image/text encoder loading shared by NIST comparisons.

No benchmark layouts, likelihood scoring or model downloads.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModel, AutoProcessor

HERE = Path(__file__).resolve().parent
TEMPLATE = "a photo of a {name}."


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def asset_record(method):
    config = read(HERE / "encoder_assets.json")
    assert config["text_template"] == TEMPLATE
    asset = config["models"][method]
    directory = Path(asset["directory"])
    files = {name: {"bytes": (directory / name).stat().st_size,
                    "sha256": digest(directory / name)} for name in asset["files"]}
    # The freshly downloaded checkpoint also has independently recorded hashes.
    provenance = directory / "asset_provenance.json"
    if provenance.exists():
        download = read(provenance)
        assert download["revision"] == asset["revision"]
        assert download["files"] == files
    return dict(**asset, file_hashes=files,
                total_bytes=sum(item["bytes"] for item in files.values()))


def load_model(asset, device):
    start = perf_counter()
    processor = AutoProcessor.from_pretrained(asset["directory"], local_files_only=True)
    model, loading = AutoModel.from_pretrained(
        asset["directory"], local_files_only=True,
        use_safetensors=asset["use_safetensors"], dtype=torch.float32,
        attn_implementation="eager", output_loading_info=True)
    for key in ("missing_keys", "mismatched_keys", "error_msgs"):
        assert not loading.get(key), (key, loading[key])
    unexpected = set(loading.get("unexpected_keys", []))
    legacy = {"text_model.embeddings.position_ids", "vision_model.embeddings.position_ids"}
    assert unexpected <= legacy if type(model).__name__ == "CLIPModel" else not unexpected
    if unexpected:
        # The cached CLIP weights persist deterministic buffers; v5 regenerates them.
        for embeddings in (model.text_model.embeddings, model.vision_model.embeddings):
            positions = embeddings.position_ids
            assert torch.equal(positions, torch.arange(positions.shape[-1]).expand_as(positions))
    loading = {key: sorted(value) if isinstance(value, set) else value
               for key, value in loading.items()}
    model.eval().requires_grad_(False).to(device)
    synchronize(device)
    return model, processor, dict(model_load_seconds=perf_counter() - start,
                                 model_class=type(model).__name__, loading_info=loading,
                                 deterministic_legacy_buffers_regenerated=sorted(unexpected))


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def feature_tensor(output):
    # Transformers 5 returns projected pooling outputs from these helper methods.
    return output.pooler_output if hasattr(output, "pooler_output") else output
