# GPT-5.2 pilot: findings and endpoint audit

The authorized pilot completed: 32 frozen scenes, 64 successful GPT-5.2 responses, and the matching 64 saved GPT-4.1-mini controls. No GPT-4o calls, new captures, explanation-first trials, held-out tests, or production changes were made. The first compatibility request was rejected before generation; its protocol/error are preserved separately. The successful image probe is included among the 64 responses.

## Classification and usable scores are different outcomes

| Model / format | Correct / 32 | Accuracy | Valid raw scores | Correctness AUROC on scored cases |
| --- | ---: | ---: | ---: | ---: |
| GPT-4.1-mini / label only | 29 | 90.625% | 32/32 | 0.7184 |
| GPT-4.1-mini / label then explanation | 29 | 90.625% | 32/32 | 0.7471 |
| GPT-5.2 / label only | 30 | 93.750% | 17/32 | 0.8667 |
| GPT-5.2 / label then explanation | 31 | 96.875% | 20/32 | 0.7895 |

On these exact scenes, moving from mini to GPT-5.2 corrected two errors and introduced one with label-only; it corrected three and introduced one with label-then-explanation. Within GPT-5.2, explanation corrected one error and introduced none. This is a small exploratory improvement, not established superiority.

The 12 other-object captures were all correct for all four model/format combinations; they contain no errors from which to assess discrimination. The remaining cohort has only two correlated states per original target.

## Important: 27 GPT-5.2 decision logprobs were positive

The actual provider returned tiny positive log probabilities for 15 label-only and 12 label-then-explanation decisions. Their values ranged from `0.00000762939453125` to `0.00006103515625`. A valid log probability cannot be positive: exponentiating these values would produce a likelihood greater than one.

For example, `task_03_bbq_sauce_state_01_label_then_evidence.json` contains the original decision token `E` with logprob `0.00006103515625`. This is in the saved provider response, not caused by summing explanation or whitespace tokens. All decisions in this pilot consisted of one decision-bearing token.

The existing parser rejected these scores while preserving their generated predictions and original token values. **Nothing was clamped to one, renormalized, or replaced with model self-reported confidence.** All 27 affected classifications happened to be correct. The cause could be numerical precision in provider computation/serialization, but that cause has not been established; no speculative correction was applied.

Consequences:

- Maximum autonomous coverage under the unchanged gate is 53.125% for GPT-5.2 label-only and 62.5% for label-then-explanation, even with a permissive threshold. The remaining scenes defer due to unavailable confidence.
- AUROC is computed on different available subsets, so the model AUROCs are not a clean like-for-like confidence-quality comparison.
- Lower GPT-5.2 AURC must not be interpreted as automatically better: its curves end at substantially lower coverage. The report's AURC integrates only the operationally available coverage domain.
- Pooled mini-versus-GPT-5.2 comparisons have no shared nonzero threshold-attainable coverage: mini's high-score ties begin above GPT-5.2's maximum coverage. No artificial tie-breaking or interpolation was used to manufacture matched-coverage results.
- Within GPT-5.2, at 15/32 accepted scenes (46.875% coverage), label-only falsely accepts one versus zero for explanation. At 16/32, both falsely accept one; at 17/32, the counts are two versus one. These are exploratory curve points, not chosen deployment thresholds.

## Tomato sauce and alphabet soup

| Target | Mini label | Mini label + explanation | GPT-5.2 label | GPT-5.2 label + explanation |
| --- | ---: | ---: | ---: | ---: |
| Tomato sauce | 0/2 | 1/2 | 2/2 | 2/2 |
| Alphabet soup | 1/2 | 1/2 | 0/2 | 1/2 |

GPT-5.2's alphabet-soup error on state 1 remains wrong with explanation and has raw likelihood `0.9932669404503173`. The explanation condition correctly identifies state 2 with likelihood `0.8927258005064531`. Thus the incorrect choice can still receive a higher score than the correct choice. Label-only is wrong on both soup scenes, with likelihoods `0.9769058277839734` and `0.9999084514564487`.

Scores still concentrate near one: the median available raw likelihood is approximately `0.9999466` for GPT-5.2 label-only and `0.9999619` for label-then-explanation. A newer model did not solve the original near-one likelihood concentration, nor establish calibrated confidence.

## Compatibility, verification, and conclusion

Actual endpoint behavior required `top_logprobs=5`, after rejecting 20. These alternatives are diagnostic-only; the gate score remains `exp(sum(original decision-token logprobs))`. The pinned model was `gpt-5.2-2025-12-11`, temperature 0, reasoning effort none, detail high, and 128 completion-token budget. All 64 responses ended with `stop` and reported zero reasoning tokens. No system fingerprint was supplied.

All raw response hashes, generated-label-to-object mappings, and valid raw likelihood computations were checked. Cached resume completed with the API call function explicitly disabled and left `records.json` byte-identical. The complete test suite passed 268 tests; the five new pilot tests were rerun after adding the observed positive-logprob case and passed.

**Conclusion:** GPT-5.2 with label-then-explanation improved classification modestly on this pilot, especially tomato sauce, but its current returned logprob behavior makes it unsuitable as a drop-in replacement for the unchanged confidence gate. Retain it as an experimental model until the positive-logprob issue is understood. Do not promote the new model, pick a threshold, or claim calibrated confidence from these results.

[Full generated report and protocol](REPORT.md) · [Endpoint audit](API_AUDIT.json) · [Matched coverage](matched_coverage.csv) · [Per-association results](associations.csv)
