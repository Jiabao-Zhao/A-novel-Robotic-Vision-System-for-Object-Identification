"""Illustrate saved DINOv2 tokens for one T-LESS object; no model inference."""

import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "outputs/bop_localization_20260917/scene_000007_000003"
TEMPLATES = ROOT / "outputs/surface_verification_20260917/stage4_tless"
OUTPUT = ROOT / "outputs/dino_token_example_20260920"
CAD, OBJECT = "obj_000018", "object_001"
QUERY_RC = (4, 3)  # Fixed illustrative patch on the observed outer circular face.


def read_json(path):
    return json.loads(path.read_text())


def foreground(mask):
    """Reconstruct the saved binary crop/letterbox and strict >50% patch gate."""
    yy, xx = np.nonzero(mask)
    crop = mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    height, width = crop.shape
    scale = 224 / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(crop.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((224, 224), np.uint8)
    y, x = (224 - size[1]) // 2, (224 - size[0]) // 2
    canvas[y:y + size[1], x:x + size[0]] = resized
    return (canvas.reshape(16, 14, 16, 14).mean(axis=(1, 3)) > .5).reshape(-1)


def unit(features):
    features = np.asarray(features, np.float32)
    return features / np.linalg.norm(features, axis=-1, keepdims=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pair = next(p for p in read_json(SCENE / "pair_scores.json")
                if p["cad_id"] == CAD and p["object_id"] == OBJECT)
    scores = np.array(pair["cls_view_cosines"])
    selected = [int(scores.argmax()) + 1, int(scores.argmin()) + 1]
    oi = read_json(SCENE / "inputs/features/index.json")["keys"].index(OBJECT)
    ti = read_json(TEMPLATES / "template_features/index.json")
    assert ti["model"] == "dinov2_vitl14"
    indices = [ti["keys"].index(f"{CAD}/{v}") for v in selected]
    with np.load(SCENE / "inputs/features/features.npz") as cache:
        query_cls = unit(cache["cls_features"][oi])
        query = unit(cache["patch_tokens"][oi].reshape(256, 1024))
    with np.load(TEMPLATES / "template_features/features.npz") as cache:
        reference_cls = unit(cache["cls_features"][indices])
        references = unit(cache["patch_tokens"][indices].reshape(2, 256, 1024))
    np.testing.assert_allclose(reference_cls @ query_cls, scores[np.array(selected) - 1], atol=2e-6)
    observed = plt.imread(SCENE / "inputs" / f"{OBJECT}.png")
    images = [observed] + [plt.imread(TEMPLATES / "templates42" / f"{CAD}_view_{v:02}.png")
                           for v in selected]
    query_fg = foreground(cv2.imread(str(SCENE / "masks" / f"{OBJECT}.png"), 0) > 0)
    view_records = {r["view_index_1based"]: r for r in read_json(SCENE / "all_view_scores.json")
                    if r["cad_id"] == CAD and r["object_id"] == OBJECT}
    patch_index = QUERY_RC[0] * 16 + QUERY_RC[1]
    assert query_fg[patch_index]
    details, maps, masks = [], [], []
    for v, reference in zip(selected, references, strict=True):
        rgba = cv2.imread(str(TEMPLATES / "templates42/renders" / CAD / f"view_{v:02}_rgba.png"), -1)
        fg = foreground(rgba[..., 3] > 0)
        masked = (query * query_fg[:, None]) @ (reference * fg[:, None]).T
        patch_score = float(np.clip(masked.max(axis=1).sum() / (query_fg.sum() + 1e-6), 0, 1))
        np.testing.assert_allclose(patch_score, view_records[v]["patch"], atol=2e-6)
        similarity = reference @ query[patch_index]
        best = int(np.where(fg, similarity, -np.inf).argmax())
        maps.append(similarity.reshape(16, 16))
        masks.append(fg.reshape(16, 16))
        details.append({"view_1based": v, "cls_cosine": float(scores[v - 1]),
                        "patch_score": patch_score, "query_patch_best_cosine": float(similarity[best]),
                        "matching_patch_rc_zero_based": list(divmod(best, 16))})

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12, 7.3), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(2.2, 1))
    labels = ["Observed object\nSaved localization-masked RGB"] + [
        f"CAD 18, view {d['view_1based']}\nCLS {d['cls_cosine']:.3f} | patch {d['patch_score']:.3f}"
        for d in details]
    for col, (picture, label) in enumerate(zip(images, labels, strict=True)):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(picture, interpolation="nearest")
        ax.set_title(label)
        ax.axis("off")
    ax = fig.add_subplot(grid[1, :])
    ax.plot(np.arange(1, 43), scores, "o-", color="#526b8d", markersize=3)
    for d, color in zip(details, ("#13876a", "#bd5347"), strict=True):
        ax.scatter(d["view_1based"], d["cls_cosine"], color=color, s=65, zorder=3)
        ax.annotate(f"view {d['view_1based']}: {d['cls_cosine']:.3f}",
                    (d["view_1based"], d["cls_cosine"]), xytext=(9, 9), textcoords="offset points")
    ax.set(xlabel="Saved CAD template index (not an angle in degrees)", ylabel="Raw CLS cosine", ylim=(0, .72))
    ax.grid(alpha=.2)
    fig.suptitle("DINOv2-L/14: same observation and CAD, different rendered views\n"
                 "T-LESS scene 7, frame 3 | localization mask IoU 0.828 | 224 x 224 inputs", fontsize=14)
    fig.savefig(OUTPUT / "cad_view_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout="constrained")
    axes[0].imshow(observed, interpolation="nearest")
    row, col = QUERY_RC
    axes[0].add_patch(Rectangle((col * 14 - .5, row * 14 - .5), 14, 14,
                               linewidth=2.5, edgecolor="#ed2546", facecolor="none"))
    axes[0].set_title("One observed patch (red square)\n14 x 14 input pixels -> 1,024 numbers")
    for ax, picture, values, fg, detail in zip(axes[1:], images[1:], maps, masks, details, strict=True):
        ax.imshow(picture, interpolation="nearest")
        heat = ax.imshow(np.ma.masked_where(~fg, values), extent=(-.5, 223.5, 223.5, -.5),
                         cmap="viridis", vmin=0, vmax=1, alpha=.8, interpolation="nearest")
        r, c = detail["matching_patch_rc_zero_based"]
        ax.add_patch(Rectangle((c * 14 - .5, r * 14 - .5), 14, 14,
                              linewidth=2.5, edgecolor="#ed2546", facecolor="none"))
        ax.set_title(f"CAD view {detail['view_1based']}: patch cosine map\n"
                     f"Best foreground match: {detail['query_patch_best_cosine']:.3f}")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(heat, ax=axes[1:], shrink=.67, label="Cosine to the selected observed patch")
    fig.suptitle("Actual saved patch features: compare one query with every CAD foreground patch\n"
                 "Both maps use the same 0-1 scale. Red CAD square = best match; not verified correspondence.", fontsize=12)
    fig.savefig(OUTPUT / "patch_token_similarity.png", dpi=150)
    plt.close(fig)
    metadata = {"model": "dinov2_vitl14", "scene": 7, "frame": 3, "cad_id": CAD, "object_id": OBJECT,
                "selection": "Illustrative large view-score gap among baseline masks with IoU >= 0.82; not aggregate evidence",
                "verified_exact_view": False, "encoder_rerun": False, "cls_dimension": 1024,
                "patch_grid": [16, 16], "query_patch_rc_zero_based": list(QUERY_RC),
                "views": details, "all_42_cls_cosines": scores.tolist()}
    (OUTPUT / "example.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"model": metadata["model"], "views": details, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
