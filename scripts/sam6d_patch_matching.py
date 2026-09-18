"""SAM-6D ISM visual scoring, isolated from proposal filtering and geometry.

Audited against JiehongLin/SAM-6D commit
1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2, model/{dinov2,loss,detector}.py.
The caller supplies unchanged descriptor inputs and resized object masks.

Adapted scoring arithmetic is distributed under the upstream MIT License:
Copyright (c) 2023 Van Nguyen Nguyen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn.functional as F


def semantic_score(observed_cls, cad_cls):
    """Upstream PairwiseSimilarity -> avg_5; best single CLS view for appearance."""
    query = F.normalize(torch.as_tensor(observed_cls, dtype=torch.float32), dim=-1)
    reference = F.normalize(torch.as_tensor(cad_cls, dtype=torch.float32), dim=-1)
    if query.ndim != 1 or reference.ndim != 2 or reference.shape[0] < 5:
        raise ValueError("Expected one query and at least five CAD view descriptors.")
    raw = F.cosine_similarity(query[None], reference, dim=-1)
    scores = raw.clamp(0, 1)
    best = int(scores.argmax())
    top5 = torch.topk(scores, k=5).values
    return {"cls_view_cosines": raw.tolist(), "clamped_cls_view_scores": scores.tolist(),
            "semantic_score": float(top5.mean()), "max_cls_score": float(scores[best]),
            "best_view_index_1based": best + 1, "top5_scores": top5.tolist()}


def masked_patch_descriptors(tokens, occupancy):
    """Strict >0.5 foreground gate; retain zero descriptors for rejected patches."""
    tokens = torch.as_tensor(tokens, dtype=torch.float32).reshape(256, -1)
    weights = torch.as_tensor(occupancy, dtype=torch.float32).reshape(256)
    return F.normalize(tokens * (weights > .5)[:, None], dim=-1)


def appearance_score(query, reference):
    """Literal compute_straight/compute_visible_ratio arithmetic on masked tokens.

Keep the upstream sum-based nonzero count, epsilon, zero background rows and
columns, final clipping, and strict visibility cutoff. No occupancy weighting.
The reverse visible ratio is diagnostic only in this visual experiment.
"""
    similarities = query @ reference.T
    maxima = similarities.max(dim=-1).values
    query_count = torch.count_nonzero(query.sum(dim=-1))
    score = (maxima.sum() / (query_count + 1e-6)).clamp(0, 1)
    reverse_maxima = similarities.max(dim=0).values
    reverse_valid = torch.count_nonzero(reverse_maxima)
    visible = torch.count_nonzero(reverse_maxima * (reverse_maxima > .5)) / (reverse_valid + 1e-6)
    return {"appearance_score": float(score), "observed_patch_count": int(query_count),
            "cad_patch_count": int(torch.count_nonzero(reference.sum(dim=-1))),
            "reverse_nonzero_match_count": int(reverse_valid), "visible_ratio_diagnostic": float(visible),
            "observed_patch_maxima": maxima.tolist()}


def combined_visual_score(semantic, appearance):
    """SAM-6D's equally weighted semantic/appearance terms, excluding geometry."""
    return .5 * semantic + .5 * appearance
