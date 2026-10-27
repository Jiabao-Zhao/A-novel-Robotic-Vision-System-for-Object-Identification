"""No-adapter NIDS-Net BOP matching on the trial's common input images.

The official BOP code pools raw DINOv2 patch tokens using a bilinear soft mask,
then selects a template view by maximum nonnegative cosine. It averages this
score with directional patch appearance for final proposal confidence. Image
encoding is outside this module. See nids_source_audit.json for source details.
"""

import math

import numpy as np


def _unavailable(reason, **extra):
    return {"status": "unavailable", "reason": reason, **extra}


def _bilinear_weights(mask, grid_side):
    """Match torch.interpolate(mode='bilinear', align_corners=False)."""
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    return functional.interpolate(tensor, size=(grid_side, grid_side), mode="bilinear",
                                  align_corners=False)[0, 0].numpy()


def nids_embedding(patch_tokens, mask224):
    """Return a raw foreground-average embedding and auditable mask metadata.

    patch_tokens are the unmodified x_norm_patchtokens, [N,C] or [S,S,C].
    mask224 is a 224-by-224 binary foreground mask aligned with the encoded
    image. No per-token L2 normalization or foreground threshold is applied.
    """
    import torch

    tokens = np.asarray(patch_tokens, dtype=np.float32)
    mask = np.asarray(mask224, dtype=np.float32)
    if mask.shape != (224, 224):
        raise ValueError("mask224 must have shape (224, 224)")
    if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError("mask224 must contain finite foreground weights in [0, 1]")
    if tokens.ndim == 2:
        side = math.isqrt(tokens.shape[0])
        if side * side != tokens.shape[0]:
            raise ValueError("patch token count must be a square grid")
        tokens = tokens.reshape(side, side, tokens.shape[-1])
    elif tokens.ndim != 3 or tokens.shape[0] != tokens.shape[1]:
        raise ValueError("patch_tokens must have shape [N,C] or [S,S,C]")
    side = tokens.shape[0]
    if side == 0 or tokens.shape[-1] == 0:
        raise ValueError("patch_tokens must be nonempty")
    if not np.isfinite(tokens).all():
        return _unavailable("nonfinite_patch_tokens", embedding=None)
    weights = _bilinear_weights(mask, side)
    tensor_weights = torch.from_numpy(weights)
    weight_sum = float(tensor_weights.sum())
    metadata = {
        "patch_grid": [side, side],
        "embedding_dimensions": int(tokens.shape[-1]),
        "mask_interpolation": "bilinear_align_corners_false_no_antialias",
        "mask_weight_sum": weight_sum,
        "positive_weight_patch_count": int(np.count_nonzero(weights > 0.0)),
        "foreground_pixel_count": int(np.count_nonzero(mask > 0.0)),
        "foreground_patch_weights": weights.ravel().tolist(),
        "tokenwise_l2_normalization": False,
        "pooled_l2_normalization": "deferred_to_cosine_matching",
        "adapter": False,
        "reduction_precision": "torch_cpu_float32",
    }
    if weight_sum <= 0.0:
        return _unavailable("empty_resized_foreground_mask", embedding=None, metadata=metadata)
    pooled = (torch.from_numpy(tokens) * tensor_weights[..., None]).sum(dim=(0, 1)) / tensor_weights.sum()
    norm = float(torch.linalg.vector_norm(pooled))
    metadata["raw_foreground_mean_norm"] = norm
    if not np.isfinite(norm) or norm < 1e-12:
        return _unavailable("zero_or_nonfinite_foreground_embedding", embedding=None, metadata=metadata)
    return {"status": "available", "embedding": pooled.tolist(), "metadata": metadata}


def nids_score(query_embedding, template_embeddings):
    """FFA-only diagnostic: maximum clamped cosine over the CAD view pool.

    Inputs are dictionaries returned by nids_embedding. Identity labels and
    simulator pose information are neither needed nor accepted by this API.
    Missing views are explicit unavailable scores, rather than silently removed.
    The caller handles ties between scene crops without view-order tie breaking.
    """
    import torch
    import torch.nn.functional as functional

    if query_embedding["status"] != "available":
        return _unavailable("query_embedding_unavailable", score=None)
    if not template_embeddings:
        return _unavailable("empty_template_pool", score=None)
    invalid = [i for i, row in enumerate(template_embeddings) if row["status"] != "available"]
    if invalid:
        return _unavailable("template_embedding_unavailable", score=None, unavailable_view_indices=invalid)
    query = np.asarray(query_embedding["embedding"], dtype=np.float32)
    templates = np.asarray([row["embedding"] for row in template_embeddings], dtype=np.float32)
    if query.ndim != 1 or templates.ndim != 2 or templates.shape[1] != query.size:
        raise ValueError("query and template embeddings must have matching feature dimensions")
    if not np.isfinite(query).all() or not np.isfinite(templates).all():
        return _unavailable("nonfinite_embedding", score=None)
    query_norm = float(np.linalg.norm(query))
    template_norms = np.linalg.norm(templates, axis=1)
    if query_norm < 1e-12 or np.any(template_norms < 1e-12):
        return _unavailable("zero_embedding_norm", score=None)
    query_tensor = functional.normalize(torch.from_numpy(query), dim=-1)
    reference_tensor = functional.normalize(torch.from_numpy(templates), dim=-1)
    cosine = functional.cosine_similarity(query_tensor[None], reference_tensor, dim=-1).numpy()
    clamped = np.clip(cosine, 0.0, 1.0)
    score = float(clamped.max())
    winners = np.flatnonzero(np.abs(clamped - score) <= 1e-12).tolist()
    return {
        "status": "available",
        "score": score,
        "best_view_index": int(winners[0]),
        "tied_best_view_indices": winners,
        "per_view_cosine": cosine.tolist(),
        "per_view_score": clamped.tolist(),
        "aggregation": "max_clamped_cosine",
        "template_count": int(len(template_embeddings)),
        "adapter": False,
    }


def nids_appearance(queryrawpatches, qmask224, refrawpatches, refmask224):
    """Official BOP query-to-reference patch score at one selected CAD view.

    Query foreground weights are bilinear samples. Reference foreground uses
    strict >0.5 coverage of each 14-by-14 pixel block. The official denominator
    counts nonzero channel sums of normalized query tokens, not mask pixels.
    """
    import torch
    import torch.nn.functional as functional

    query = np.asarray(queryrawpatches, dtype=np.float32).reshape(-1, np.asarray(queryrawpatches).shape[-1])
    reference = np.asarray(refrawpatches, dtype=np.float32).reshape(-1, np.asarray(refrawpatches).shape[-1])
    qmask = np.asarray(qmask224, dtype=np.float32)
    rmask = np.asarray(refmask224, dtype=np.float32)
    if query.shape[0] != 256 or reference.shape[0] != 256:
        raise ValueError("BOP 224-by-224 appearance requires 256 ViT/14 patch tokens")
    if query.shape[1] != reference.shape[1] or qmask.shape != (224, 224) or rmask.shape != (224, 224):
        raise ValueError("query/reference dimensions or 224-by-224 masks do not match")
    for array in [query, reference, qmask, rmask]:
        if not np.isfinite(array).all():
            return _unavailable("nonfinite_appearance_input", score=None)
    for mask in [qmask, rmask]:
        if np.any((mask < 0) | (mask > 1)):
            raise ValueError("appearance masks must be finite foreground weights in [0,1]")
    with torch.no_grad():
        qweights = functional.interpolate(torch.from_numpy(qmask)[None, None], size=(16, 16),
                                         mode="bilinear", align_corners=False).reshape(-1)
        coverage = functional.avg_pool2d(torch.from_numpy(rmask)[None, None], kernel_size=14, stride=14).reshape(-1)
        reference_selected = coverage > 0.5
        if not torch.any(qweights > 0) or not torch.any(reference_selected):
            return _unavailable("empty_appearance_foreground", score=None)
        query = functional.normalize(torch.from_numpy(query) * qweights[:, None], dim=-1)
        reference = functional.normalize(torch.from_numpy(reference) * reference_selected[:, None], dim=-1)
        similarities = query @ reference.T
        best_per_query = similarities.max(dim=-1).values
        query_count = int(torch.count_nonzero(query.sum(dim=-1)))
        raw_score = float(best_per_query.sum() / (query_count + 1e-6))
    return {
        "status": "available",
        "score": float(np.clip(raw_score, 0.0, 1.0)),
        "unclamped_score": raw_score,
        "query_nonzero_channel_sum_count": query_count,
        "query_positive_mask_patch_count": int(torch.count_nonzero(qweights > 0)),
        "reference_majority_foreground_patch_count": int(torch.count_nonzero(reference_selected)),
        "query_mask_rule": "bilinear_soft_weight_then_token_l2",
        "reference_mask_rule": "avgpool14_strict_greater_than_half_then_token_l2",
        "direction": "mean_query_max_reference",
        "denominator_epsilon": 1e-6,
        "appearance_arithmetic": "torch_cpu_float32",
    }


def nids_match(query_record, reference_records):
    """Full BOP no-adapter proposal confidence on controlled common images.

    Each input record has patch_tokens and mask224. The FFA-only score is also
    returned as a prespecified diagnostic. Appearance is evaluated at the first
    maximum FFA view, matching torch.max's deterministic view selection.
    """
    query_embedding = nids_embedding(query_record["patch_tokens"], query_record["mask224"])
    reference_embeddings = [nids_embedding(row["patch_tokens"], row["mask224"]) for row in reference_records]
    semantic = nids_score(query_embedding, reference_embeddings)
    if semantic["status"] != "available":
        return _unavailable("ffa_score_unavailable", score=None, ffa=semantic)
    view = semantic["best_view_index"]
    reference = reference_records[view]
    appearance = nids_appearance(query_record["patch_tokens"], query_record["mask224"],
                                 reference["patch_tokens"], reference["mask224"])
    if appearance["status"] != "available":
        return _unavailable("appearance_score_unavailable", score=None, ffa=semantic, appearance=appearance)
    return {
        "status": "available",
        "score": (semantic["score"] + appearance["score"]) / 2.0,
        "ffa_only_score": semantic["score"],
        "appearance_score": appearance["score"],
        "best_view_index": view,
        "aggregation": "mean_max_ffa_cosine_and_best_ffa_view_patch_appearance",
        "ffa": semantic,
        "appearance": appearance,
        "adapter": False,
    }
