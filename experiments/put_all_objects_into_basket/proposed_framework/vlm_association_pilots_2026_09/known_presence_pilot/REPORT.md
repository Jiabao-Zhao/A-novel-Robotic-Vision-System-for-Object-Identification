# Known-presence / no-N pilot

Ten frozen LIBERO-Object calibration scenes, initial state 0 per target, selected by index rather than accuracy. Twenty fresh requests (checkpoints reused on resume). No validation/test outcomes read. This pilot changes only the presence/N prompt, not visual labels or images.

## Prompt change

Replace:

> N means none of the localized candidates corresponds to the target.

with:

> The target is guaranteed to correspond to one of the localized candidates.

The output list changes from A, B, C, D, E, F, G, N to A, B, C, D, E, F, G. Every target is confirmed in the frozen localized candidate set. The specific ground-truth ID is never disclosed. Candidate IDs/mapping/order, crops and geometry stay fixed. This combines an explicit presence assertion with removal of N; it cannot isolate their separate effects.

Model: `gpt-4.1-mini-2025-04-14`; temperature 0; image detail high; identical direct-letter 1792×896 contact sheets from frozen 768×768 RGB. Same system prompt, max_completion_tokens=8, logprobs=True, top_logprobs=20. Pair order alternates by task index. No logit bias or constrained decoding.

Score = exp(sum of generated decision-bearing token log probabilities). Separable formatting is excluded. No sequence-length or candidate normalization. Top alternatives are archived only. Missing probabilities/invalid outputs remain unavailable, not zero. Production prompt, HITL, final_object_id, CAD and thresholds are untouched.

## Paired results

| Target | With N: choice / correct | Raw likelihood | No N: choice / correct | Raw likelihood | Change |
| --- | --- | ---: | --- | ---: | ---: |
| alphabet soup | E / True | 0.617616693418 | F / False | 0.980871487998 | 0.36325479458 |
| cream cheese | F / True | 0.99999956798 | F / True | 1 | 4.32019894414e-07 |
| salad dressing | C / True | 1 | C / True | 1 | 0 |
| bbq sauce | E / True | 1 | E / True | 1 | 0 |
| ketchup | C / True | 1 | C / True | 1 | 0 |
| tomato sauce | C / True | 1 | E / False | 1 | 0 |
| butter | G / True | 1 | G / True | 1 | 0 |
| milk | B / True | 1 | B / True | 0.999999806387 | -1.93612630706e-07 |
| chocolate pudding | G / True | 0.999999687184 | G / True | 1 | 3.12816276771e-07 |
| orange juice | C / True | 1 | C / True | 1 | 0 |

## Summary

```json
{
  "with_n": {
    "n": 10,
    "correct": 10,
    "scores_available": 10,
    "mean": 0.9617615948581901,
    "median": 1.0,
    "min": 0.6176166934180721,
    "max": 1.0,
    "exactly_one": 7,
    "wrong_ge_099": 0
  },
  "known_present_no_n": {
    "n": 10,
    "correct": 8,
    "scores_available": 10,
    "mean": 0.998087129438527,
    "median": 1.0,
    "min": 0.9808714879979001,
    "max": 1.0,
    "exactly_one": 8,
    "wrong_ge_099": 1
  }
}
```

## Interpretation limits

Higher generated-label likelihood is not necessarily better recognition or calibrated correctness. One response per condition cannot separate prompt effects from API repeatability variation. Ten scenes with one state per target are a quick score-sensitivity check, not a reliable threshold calibration or generalization experiment. Additional states test pose/occlusion robustness; they do not add semantic categories. No extra states or absent-target trials were run.

inputs.json preserves both exact prompts, mappings and image/localization/GT/settings/source hashes. responses/ preserves raw provider responses, tokens and alternatives. records.json, summary.json and paired_scores.csv preserve full numeric precision; this table is display-rounded.

```sh
python -m scripts.run_vlm_presence_pilot check
python -m scripts.run_vlm_presence_pilot run
python -m scripts.run_vlm_presence_pilot report
```

`run` resumes only missing requests; `report` is offline. No automatic full experiment.

API reference: [OpenAI Chat Completions](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create). The official SDK token logprobs/top alternatives structure was checked; scoring reuses the existing parser.
