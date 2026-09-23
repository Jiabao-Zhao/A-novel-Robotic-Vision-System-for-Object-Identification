"""Exercise original SAM-6D semantic assignment with saved scene descriptors.

Run with: python -m scripts.check_sam6d_multiobject_matching

This checks multi-CAD matching behavior, not segmentation or pose accuracy.
No SAM, DINO, renderer, or registration model is run. The cached descriptors
are our DINOv2-L/14 localization inputs and 42-view templates, not a faithful
reproduction of the full SAM-6D pipeline.
"""

import ast
import hashlib
import json
from pathlib import Path
import runpy
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/cache/sam6d_matching_source"
FEATURE_RUN = ROOT / "outputs/large_encoder_wrist_association_2026-09-11/20260917T173743688156Z"
SCENE = ROOT / "experiments/wrist_association_2026-09-11"
OUTPUT = ROOT / "outputs/sam6d_multiobject_check_20260918/results.json"
TARGETS = ["red_block", "blue_block"]


def main():
    started = perf_counter()
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Reuse the existing loader for the literal upstream cosine-scoring classes.
    upstream = runpy.run_path(str(ROOT / "tests/test_wrist_sam6d_patch.py"))
    namespace = upstream["upstream_scoring_classes"]()
    path = CACHE / "model/detector.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "Instance_Segmentation_Model")
    names = {"compute_semantic_score", "best_template_pose"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == names
    cls.body, cls.bases, cls.decorator_list, cls.keywords = methods, [], [], []
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    matcher = namespace["Instance_Segmentation_Model"]()
    matcher.matching_config = SimpleNamespace(
        metric=namespace["PairwiseSimilarity"](),
        aggregation_function="avg_5", confidence_thresh=0.2)

    manifest = json.loads((FEATURE_RUN / "inputs.json").read_text())
    with np.load(FEATURE_RUN / "large/features.npz") as saved:
        features = torch.tensor(saved["cls_features"], dtype=torch.float32, device=device)
    get_feature = lambda key: features[manifest[key]["feature_index"]]
    object_ids = sorted(k.split("/")[-1] for k in manifest
                        if k.startswith("observed/localization/"))
    observed = torch.stack([get_feature(f"observed/localization/{oid}") for oid in object_ids])
    cad_ids = list(json.loads((SCENE / "manifest.json").read_text())["targets"])
    cases = {}
    for name, ids in (("red_only", TARGETS[:1]), ("blue_only", TARGETS[1:]),
                      ("red_and_blue_together", TARGETS), ("all_six_together", cad_ids)):
        references = torch.stack([torch.stack([
            get_feature(f"blenderproc42/{cid}/{view}") for view in range(1, 43)
        ]) for cid in ids])
        matcher.ref_data = {"descriptors": references}
        with torch.inference_mode():
            selected, assigned, scores, views = matcher.compute_semantic_score(observed)
            per_view = matcher.matching_config.metric(observed, references)
            aggregate = per_view.topk(5, dim=-1).values.mean(-1)
        cases[name] = {
            "cad_ids": ids, "comparison_shape": list(per_view.shape),
            "top5_scores": aggregate.tolist(),
            "assignments": [
                {"object_id": object_ids[i], "cad_id": ids[j], "score": score,
                 "best_template_index_1based": view + 1}
                for i, j, score, view in zip(selected.tolist(), assigned.tolist(),
                                             scores.tolist(), views.tolist())],
        }
    joint = np.array(cases["red_and_blue_together"]["top5_scores"])
    separate = np.concatenate([cases[n]["top5_scores"] for n in ("red_only", "blue_only")], axis=1)
    np.testing.assert_allclose(joint, separate, rtol=0, atol=0)
    # Ground truth is attached only after the predictions above have been made.
    identity = {r["object_id"]: r["simulator_instance"]
                for r in json.loads((SCENE / "identity_audit.json").read_text())}
    for case in cases.values():
        for prediction in case["assignments"]:
            prediction["ground_truth_cad_id"] = identity[prediction["object_id"]]
    result = {
        "scope": __doc__.strip(), "device": device,
        "upstream_commit": "1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (path, CACHE / "model/loss.py", CACHE / "model/utils.py")},
        "feature_run": str(FEATURE_RUN.relative_to(ROOT)), "object_ids": object_ids,
        "joint_vs_separate_score_max_difference": float(np.max(np.abs(joint - separate))),
        "segmentation_rerun": False, "pose_estimation_run": False,
        "cases": cases, "elapsed_seconds": perf_counter() - started,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
