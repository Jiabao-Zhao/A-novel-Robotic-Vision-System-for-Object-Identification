# Short-evidence explanation pilot

Ten frozen LIBERO-Object state-0 calibration scenes, 30 fresh responses. No-N/known-presence policy and direct A-G visual labels fixed across all three conditions. Scores below include only the generated decision label, not explanation likelihood.

## Results

| Format | Correct / 10 | Mean label likelihood | Range | AUROC | Wrong >=0.99 | Exact-one scores |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| label_only | 9 | 0.975450069 | 0.816551618 - 1 | 1 | 0 | 8 |
| label_then_evidence | 9 | 0.955985311 | 0.561564806 - 1 | 1 | 0 | 5 |
| evidence_then_label | 7 | 1 | 1 - 1 | 0.5 | 3 | 10 |

AUROC is undefined if there are no scored errors or no scored correct predictions. A lower likelihood or a wider range alone is not evidence of a better deferral signal.

## Every target

| Target | Ground truth | Label only: choice / correct / score | Label then evidence | Evidence then label |
| --- | --- | --- | --- | --- |
| alphabet soup | object_005 | F / False / 0.816551618 | B / False / 0.561564806 | B / False / 1 |
| cream cheese | object_006 | F / True / 1 | F / True / 0.999794812 | F / True / 1 |
| salad dressing | object_003 | C / True / 1 | C / True / 1 | C / True / 1 |
| bbq sauce | object_005 | E / True / 1 | E / True / 0.99999623 | E / True / 1 |
| ketchup | object_003 | C / True / 1 | C / True / 1 | C / True / 1 |
| tomato sauce | object_003 | C / True / 0.937949068 | C / True / 0.999998495 | E / False / 1 |
| butter | object_007 | G / True / 1 | G / True / 1 | G / True / 1 |
| milk | object_002 | B / True / 1 | B / True / 1 | B / True / 1 |
| chocolate pudding | object_007 | G / True / 1 | G / True / 0.998498768 | E / False / 1 |
| orange juice | object_003 | C / True / 1 | C / True / 1 | C / True / 1 |

## False acceptances at exactly matched coverage

Only attainable thresholds; tied scores are not split. No operating threshold is selected. Unavailable scores count as deferred. These are exploratory calibration-scene comparisons.

| Explanation format | Coverage | Label-only false accepts | Explanation false accepts | Label-only accuracy | Explanation accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| label_then_evidence | 80% | 0 | 0 | 1 | 1 |
| label_then_evidence | 90% | 0 | 0 | 1 | 1 |
| label_then_evidence | 100% | 1 | 1 | 0.9 | 0.9 |
| evidence_then_label | 100% | 1 | 3 | 0.9 | 0.7 |

## Alphabet soup and tomato sauce: actual output text

These are model-generated visual claims, not verified descriptions or ground-truth evidence.

### alphabet soup — evidence_then_label

Correct: False; label likelihood: 1

```text
The can in B shows a label with images of alphabet soup ingredients and typical soup can design.  
B
```

### alphabet soup — label_only

Correct: False; label likelihood: 0.816551618

```text
F
```

### alphabet soup — label_then_evidence

Correct: False; label likelihood: 0.561564806

```text
B
The can in B shows a label with orange shapes typical of alphabet soup cans.
```

### tomato sauce — evidence_then_label

Correct: False; label likelihood: 1

```text
The bottle labeled "BBQ Sauce" is the closest match to tomato sauce among the candidates.
E
```

### tomato sauce — label_only

Correct: True; label likelihood: 0.937949068

```text
C
```

### tomato sauce — label_then_evidence

Correct: True; label likelihood: 0.999998495

```text
C
The can in C has a red and green label typical of tomato sauce packaging.
```

## Controlled protocol and limitations

Same `gpt-4.1-mini-2025-04-14`, temperature 0, detail high, frozen RGB/localization/GT/candidate order and crops. All arms use a shared format-neutral system instruction and max_completion_tokens=128 to accommodate the explanation; hence the fresh label-only control is the valid comparison, not a historical 8-token response. No API history from other conditions, logit bias, constrained decoding, or self-reported confidence. The 20-word evidence request is identical in both explanation arms, with order reversed. Word-budget overruns are logged, never rerun.

Parser locates the designated first/last label line and sums ONLY overlapping original token logprobs. Standalone whitespace/punctuation is excluded. If a label shares a token with formatting, that token is indivisible and included. Missing/sentinel probabilities, malformed output or truncation do not create a score. All provider tokens/alternatives remain in responses/.

Evidence-first likelihood is conditioned on the already-generated explanation. It may express consistency with that explanation rather than correct object identity. Evidence written after the label cannot retroactively modify the earlier token probability, although asking for it changes the input prompt. Neither score is calibrated correctness probability.

Only ten previously inspected calibration scenes, one response per condition: insufficient for robust model/threshold selection or causal claims beyond this pilot. Fixed temperature does not quantify API repeatability. Each condition's own prediction defines correctness; accuracy changes and score discrimination are reported separately. No validation/test outcomes read, no production/HITL/CAD/planner edits, no automatic follow-up experiment.

inputs.json records exact prompts, settings, image/localization/GT/source hashes. records.json/associations.csv save explanations and decision tokens. summary.json contains correct/incorrect distributions and paired transitions. threshold_curves.csv includes accuracy, coverage, false accepts and risk; AURC uses existing trapezoidal tied-endpoint convention, not attainable within-tie interpolation.

```sh
python -m scripts.run_vlm_explanation_pilot check
python -m scripts.run_vlm_explanation_pilot run
python -m scripts.run_vlm_explanation_pilot report
```

`run` resumes missing calls; `report` is offline. [Official output-token logprob API](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create).
