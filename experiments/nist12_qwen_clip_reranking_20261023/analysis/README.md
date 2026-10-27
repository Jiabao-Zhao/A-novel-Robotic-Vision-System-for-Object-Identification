# CLIP top-five retrieval + DINO selection

Eight retained N00–N07 simulation layouts contain twelve products each. All images use the approved middle lighting condition with original colors preserved. Qwen3.5-9B generates attributes and descriptive phrases; frozen CLIP ViT-B/32 retrieves five crops, then the unchanged DINOv2 global/foreground-patch score makes the final selection. DINO alone ranks all twelve crops.

| Method | Correct /96 | Wrong | Tied | Unavailable | Fixed DINO errors | Introduced errors |
|---|---:|---:|---:|---:|---:|---:|
| DINO alone | 65 | 31 | 0 | 0 | — | — |
| CLIP full → DINO | 66 | 30 | 0 | 0 | 1 | 0 |
| CLIP shape text → DINO | 65 | 31 | 0 | 0 | 1 | 1 |
| CLIP full only | 55 | 41 | 0 | 0 | 9 | 19 |
| CLIP shape text only | 57 | 39 | 0 | 0 | 9 | 17 |

Full-attribute CLIP→DINO is the primary comparison. SHAPE-text is a separately frozen control and retains RGB image features. CLIP-only selection is diagnostic.

| Product | DINO /8 | CLIP full→DINO /8 | CLIP shape→DINO /8 | Full recall@5 /8 | Shape recall@5 /8 |
|---|---:|---:|---:|---:|---:|
| Large gear | 8 | 8 | 8 | 8 | 8 |
| M16 nut | 8 | 8 | 8 | 8 | 8 |
| 16 mm round pin | 5 | 5 | 5 | 8 | 8 |
| 16 mm rectangular pin | 6 | 6 | 6 | 8 | 8 |
| Waterproof | 6 | 7 | 7 | 8 | 8 |
| USB | 8 | 8 | 8 | 8 | 8 |
| D-sub | 7 | 7 | 7 | 8 | 8 |
| BNC | 8 | 8 | 7 | 8 | 7 |
| 12 mm round pin | 2 | 2 | 2 | 7 | 8 |
| 8 mm round pin | 3 | 3 | 3 | 8 | 8 |
| 12 mm rectangular pin | 1 | 1 | 1 | 8 | 8 |
| 8 mm rectangular pin | 3 | 3 | 3 | 8 | 8 |

## Errors fixed or introduced

- **N07 · Waterproof · fixed_error:** DSUB_Male → Waterproof_Male; true crop in top five: True.

## Interpretation and limits

The hybrid is a filter, not a new DINO score: if CLIP retains the unique DINO winner, the final answer must remain identical. An error can be rescued only when CLIP removes the wrong DINO winner while retaining a better surviving match. A correct match is lost when retrieval removes it, or evidence is unavailable. This invariant is checked independently for every task and both text conditions.

The four CLIP terms are image–image, text–text, crop-image/CAD-text and crop-text/CAD-image cosine similarities, equally averaged per CAD view and maximized over all twenty views. The five highest distinct scene crops are shortlisted. Ties at the fifth/sixth boundary are unavailable; final ties remain unresolved. No per-case fallback, score fusion, assignment, weight fitting or tuning is used.

CLIP and Qwen receive the same native raw RGB files. DINO uses exact isolated-CAD silhouettes and depth-plane foreground masks with white, aspect-preserving 224-pixel preprocessing. Both primary methods reuse the exact same DINO candidate scores and CAD-view mappings; adding CLIP does not change this preprocessing.

These are reused known-CAD simulator scenes with one instance of each product and previously accepted depth-localizer rectangles. They do not measure localization quality, absent targets, duplicate instances, unseen CADs, real-image robustness, robot execution or assembly success. The relit twelve-object condition differs from the earlier eight-object J study; its 58/64 result is not this baseline. Scores are uncalibrated and all pretrained weights remain unchanged.

Evidence: [all 96 tasks and candidate images/text](index.html), [per-object CSV](per_object.csv), [all decisions](decisions.csv), [paired changes](paired_changes.csv), [within-family errors](within_family.csv), [verification](verification.json).

![Comparison](comparison.png)

![Every changed primary decision](changed_cases.png)

![One full pipeline example](pipeline_case.png)
