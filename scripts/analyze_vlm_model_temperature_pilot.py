"""Offline analysis of the bounded model/temperature pilot; no API calls."""

from collections import Counter, defaultdict
import csv
import json
import math

from scripts.analyze_vlm_uncertainty_scores import (
    risk_coverage_curve, score_distribution, threshold_metrics,
)
from scripts.run_vlm_model_temperature_pilot import (
    CONDITIONS, MODELS, OUTPUT_ROOT, REPEATS, TEMPERATURES,
    response_record,
)


SCORE = "raw_sequence_likelihood"
DISPLAY_THRESHOLDS = (.5, .7, .8, .9, .95, .99, .999, .9999, .99999, .999999, 1.0)


def validate_records(payload):
    records, spec = payload["records"], payload["protocol"]
    scenes = {s["sample_id"]: s for s in spec["samples"]}
    expected = {(s, m, t, r) for s in scenes for m, t in CONDITIONS for r in range(REPEATS)}
    keyed = {}
    for row in records:
        key = (row["sample_id"], row["model"], row["temperature"], row["repeat"])
        if key in keyed:
            raise ValueError(f"Duplicate call: {key}")
        keyed[key] = row
        scene = scenes[row["sample_id"]]
        if any(row.get(field) != value for field, value in scene.items()):
            raise ValueError("Input differs across paired conditions.")
        if row["partition"] != "calibration" or row["requested_model"] != row["model"]:
            raise ValueError("Unexpected split or returned model.")
        rebuilt = response_record(scene, row["requested_model"], row["temperature"],
                                  row["repeat"], row["order_index"], row["provider_response"])
        for field in ("choice", "predicted_object_id", "correct", "raw_log_probability",
                      SCORE, "decision_tokens", "first_position_choice_logprobs"):
            if row[field] != rebuilt[field]:
                raise ValueError(f"Saved {field} differs from raw API response.")
    if set(keyed) != expected:
        raise ValueError("Incomplete condition/repeat grid.")
    return keyed


def distribution(values):
    import numpy as np

    if not values:
        return None
    return {"n": len(values), "mean": float(np.mean(values)),
            **{k: float(np.quantile(values, q)) for k, q in
               (("min", 0), ("q10", .1), ("q25", .25), ("median", .5),
                ("q75", .75), ("q90", .9), ("max", 1))}}


def same_token_rows(a, b, comparison):
    """Compare only returned exact tokens at the identical prefix position."""
    pa, pb = a["first_position_choice_logprobs"], b["first_position_choice_logprobs"]
    return [{"comparison": comparison, "sample_id": a["sample_id"],
             "model_a": a["model"], "model_b": b["model"],
             "temperature_a": a["temperature"], "temperature_b": b["temperature"],
             "repeat_a": a["repeat"], "repeat_b": b["repeat"],
             "token": token, "logprob_a": pa[token], "logprob_b": pb[token],
             "logprob_delta_b_minus_a": pb[token] - pa[token],
             "probability_delta_b_minus_a": math.exp(pb[token]) - math.exp(pa[token])}
            for token in sorted(pa.keys() & pb.keys())]


def gap_comparison(low, high):
    """For T=.5 versus T=1, compare log odds of the same two semantic choices.

    At fixed logits, post-temperature log odds would have ratio 2; unchanged
    returned log odds would have ratio 1. This is a diagnostic, not a guarantee
    of API internals: hosted-model numerical variability can affect the gaps.
    """
    a, b = low["first_position_choice_logprobs"], high["first_position_choice_logprobs"]
    ordered = sorted(b, key=b.get, reverse=True)
    if not ordered:
        return None
    first = ordered[0]
    second = next((t for t in ordered[1:] if t.strip() != first.strip()), None)
    if second is None or first not in a or second not in a:
        return None
    gap_one = b[first] - b[second]
    if gap_one <= 1e-8:
        return None
    gap_half = a[first] - a[second]
    return {"sample_id": low["sample_id"], "model": low["model"],
            "repeat": low["repeat"], "first_token": first, "second_token": second,
            "gap_t05": gap_half, "gap_t1": gap_one, "ratio": gap_half / gap_one,
            "absolute_error_unchanged": abs(gap_half - gap_one),
            "absolute_error_temperature_scaled": abs(gap_half - 2 * gap_one)}


def summarize_condition(rows):
    available = [r for r in rows if r[SCORE] is not None]
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["sample_id"]].append(row)
    curve = risk_coverage_curve(rows, SCORE)
    return {
        "calls": len(rows), "scenes": len(by_scene),
        "correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "score_availability": len(available) / len(rows),
        "likelihood": distribution([r[SCORE] for r in available]),
        "log_probability": distribution([r["raw_log_probability"] for r in available]),
        "by_correctness": score_distribution(rows, SCORE),
        "aurc": curve["aurc_over_operational_coverage"],
        "exact_one_count": sum(r[SCORE] == 1 for r in rows),
        "score_ge_099_count": sum(r[SCORE] >= .99 for r in available),
        "wrong_ge_099_count": sum(not r["correct"] and r[SCORE] >= .99 for r in available),
        "wrong_exact_one_count": sum(not r["correct"] and r[SCORE] == 1 for r in rows),
        "scenes_with_prediction_changes": sum(len({r["choice"] for r in rs}) > 1 for rs in by_scene.values()),
        "mean_decision_tokens": sum(r["decision_token_count"] for r in rows) / len(rows),
        "thresholds": [threshold_metrics(rows, SCORE, t) for t in DISPLAY_THRESHOLDS],
    }, curve["points"]


def paired_summary(pairs):
    """Bootstrap scenes, retaining all repeated calls within each scene."""
    import numpy as np

    by_scene = defaultdict(list)
    for a, b in pairs:
        by_scene[a["sample_id"]].append(int(b["correct"]) - int(a["correct"]))
    deltas = np.array([np.mean(values) for values in by_scene.values()])
    rng = np.random.default_rng(20260906)
    bootstrap = rng.choice(deltas, (4000, len(deltas)), replace=True).mean(axis=1)
    same = [(a, b) for a, b in pairs if a["choice"] is not None and a["choice"] == b["choice"]
            and a[SCORE] is not None and b[SCORE] is not None]
    return {"paired_calls": len(pairs), "scenes": len(by_scene),
            "prediction_changes": sum(a["choice"] != b["choice"] for a, b in pairs),
            "accuracy_delta_b_minus_a": float(deltas.mean()),
            "scene_bootstrap_95_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
            "same_choice_pairs": len(same),
            "same_choice_score_delta": distribution([b[SCORE] - a[SCORE] for a, b in same])}


def analyze(payload):
    keyed = validate_records(payload)
    records = payload["records"]
    ids = sorted({r["sample_id"] for r in records})
    summary = {"conditions": [], "paired_comparisons": {}, "same_token_summary": {},
               "temperature_gap_diagnostics": {}, "class_results": []}
    curves, tokens, gaps = [], [], []
    for model, temperature in CONDITIONS:
        rows = [r for r in records if r["model"] == model and r["temperature"] == temperature]
        stats, curve = summarize_condition(rows)
        summary["conditions"].append({"model": model, "temperature": temperature, **stats})
        curves.extend({"model": model, "temperature": temperature, **p} for p in curve)
        for target in sorted({r["target_description"] for r in rows}):
            subset = [r for r in rows if r["target_description"] == target]
            summary["class_results"].append({"model": model, "temperature": temperature,
                "target": target, "n": len(subset), "correct": sum(r["correct"] for r in subset),
                "wrong_ge_099": sum(not r["correct"] and r[SCORE] is not None and r[SCORE] >= .99 for r in subset)})
    comparisons = {}
    for model in MODELS:
        for t in TEMPERATURES[1:]:
            comparisons[f"{model}: T0 -> T{t:g}"] = [
                (keyed[s, model, 0., r], keyed[s, model, t, r]) for s in ids for r in range(REPEATS)]
        for t in TEMPERATURES:
            comparisons[f"{model}: repeated T{t:g}"] = [
                (keyed[s, model, t, 0], keyed[s, model, t, r]) for s in ids for r in range(1, REPEATS)]
        for s in ids:
            for r in range(REPEATS):
                gap = gap_comparison(keyed[s, model, .5, r], keyed[s, model, 1., r])
                if gap is not None:
                    gaps.append(gap)
        selected_gaps = [g for g in gaps if g["model"] == model]
        summary["temperature_gap_diagnostics"][model] = {
            "available_pairs": len(selected_gaps), "possible_pairs": len(ids) * REPEATS,
            **{field: distribution([g[field] for g in selected_gaps]) for field in
               ("ratio", "absolute_error_unchanged", "absolute_error_temperature_scaled")}}
    for t in TEMPERATURES:
        comparisons[f"model: mini -> 4o at T{t:g}"] = [
            (keyed[s, MODELS[0], t, r], keyed[s, MODELS[1], t, r]) for s in ids for r in range(REPEATS)]
    for name, pairs in comparisons.items():
        summary["paired_comparisons"][name] = paired_summary(pairs)
        matched = [item for a, b in pairs for item in same_token_rows(a, b, name)]
        tokens.extend(matched)
        summary["same_token_summary"][name] = {
            "paired_calls": len(pairs), "common_exact_token_entries": len(matched),
            "absolute_logprob_delta": distribution([abs(x["logprob_delta_b_minus_a"]) for x in matched]),
            "absolute_probability_delta": distribution([abs(x["probability_delta_b_minus_a"]) for x in matched])}
    summary["audit"] = {
        "calls": len(records), "scenes": len(ids), "all_inputs_match_frozen_prompts_and_images": True,
        "finish_reasons": dict(Counter(r["provider_response"]["choices"][0]["finish_reason"] for r in records)),
        "fingerprints": {m: dict(Counter(r["provider_response"].get("system_fingerprint", "unavailable")
                                          for r in records if r["model"] == m)) for m in MODELS},
        "invalid_choices": sum(r["choice"] is None for r in records),
        "unavailable_scores": sum(r[SCORE] is None for r in records),
        "full_vs_decision_differences": sum(r["full_response_log_probability"] != r["raw_log_probability"] for r in records),
        "usage": {m: {k: sum(r["provider_response"].get("usage", {}).get(k, 0)
                              for r in records if r["model"] == m)
                       for k in ("prompt_tokens", "completion_tokens")} for m in MODELS}}
    return summary, curves, tokens, gaps


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figures(records, curves, gaps, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {0.: "#0072B2", .5: "#D55E00", 1.: "#009E73"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, model in enumerate(MODELS):
        for t in TEMPERATURES:
            subset = [r for r in records if r["model"] == model and r["temperature"] == t and r[SCORE] is not None]
            ordered = sorted(r[SCORE] for r in subset)
            axes[i, 0].step(ordered, np.arange(1, len(ordered)+1)/len(ordered), where="post", label=f"T={t:g}", color=colors[t])
            ps = [p for p in curves if p["model"] == model and p["temperature"] == t]
            axes[i, 1].plot([p["autonomous_coverage"] for p in ps], [p["autonomous_accuracy"] for p in ps], ".-", color=colors[t], label=f"T={t:g}")
            for correct in (True, False):
                values = [r[SCORE] for r in subset if r["correct"] == correct]
                transformed = -np.log10(np.maximum(1-np.array(values), 1e-15))
                axes[i, 2].hist(transformed, bins=np.arange(0, 16), histtype="step",
                                color=colors[t], linestyle="-" if correct else "--",
                                label=f"T={t:g} {'correct' if correct else 'wrong'}")
        axes[i, 0].set(title=f"{model}\nRaw score ECDF", xlabel="Raw likelihood", ylabel="Fraction of scored calls", xlim=(0, 1.01))
        axes[i, 1].set(title="Accuracy versus autonomous coverage", xlabel="Coverage", ylabel="Association accuracy", xlim=(0, 1.02), ylim=(0, 1.02))
        axes[i, 2].set(title="Score spread by correctness", xlabel="-log10(1-score); exact 1 displayed at 15", ylabel="Calls")
        for axis in axes[i]:
            axis.legend(fontsize=7)
            axis.grid(alpha=.2)
    fig.suptitle("Model / temperature pilot: 20 frozen calibration scenes, 3 calls per scene / condition")
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, model in zip(axes, MODELS):
        rows = [g for g in gaps if g["model"] == model]
        x = [g["gap_t1"] for g in rows]
        y = [g["gap_t05"] for g in rows]
        limit = max([1, *x, *y]) * 1.05
        axis.scatter(x, y, alpha=.65, s=22, label=f"Same-token pairs (n={len(rows)})")
        axis.plot([0, limit], [0, limit], color="#009E73", label="Unchanged log odds: y=x")
        axis.plot([0, limit/2], [0, limit], "--", color="#D55E00", label="Temperature-scaled: y=2x")
        axis.set(title=model, xlabel="Log-odds gap at T=1", ylabel="Same two tokens: gap at T=0.5",
                 xlim=(-.03 * limit, limit), ylim=(min([0, *y])-.05 * limit, limit))
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.suptitle("Returned first-position log probabilities: temperature diagnostic")
    fig.tight_layout()
    fig.savefig(root / "temperature_logodds.png", dpi=160)
    plt.close(fig)


def save_report(payload, summary, curves, tokens, gaps):
    root = OUTPUT_ROOT / "analysis"
    root.mkdir(exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(root / "threshold_curves.csv", curves)
    write_csv(root / "same_first_token_logprobs.csv", tokens)
    write_csv(root / "temperature_logodds.csv", gaps)
    fields = ("sample_id", "model", "temperature", "repeat", "target_description", "ground_truth_object_id",
              "predicted_object_id", "correct", "generated_output_text", "decision_token_count", "raw_log_probability", SCORE)
    write_csv(root / "associations.csv", [{f: r[f] for f in fields} for r in payload["records"]])
    write_csv(root / "class_results.csv", summary["class_results"])
    save_figures(payload["records"], curves, gaps, root)
    fmt = lambda v: "unavailable" if v is None else f"{v:.6g}"
    lines = ["# VLM model / temperature pilot", "",
        "20 fixed LIBERO calibration scenes (initial states 0 and 1 from each class). "
        "Two pinned models, three temperatures, three repeats: 360 fresh responses, "
        "not 360 independent scenes. Images, prompt text, target descriptions, candidate order, "
        "ground truth, and output representation are frozen from the compact-label ablation. "
        "No human corrections, CAD registration, planning, or robot execution. No test-set use "
        "or threshold selection; all curves are exploratory calibration-subset results.", "",
        "Score: exp(sum of provider log probabilities for generated decision-bearing tokens). "
        "Generated labels determine predictions; alternatives never change scores or predictions.", "",
        "| Model | T | Accuracy | Mean score | Median score | Exactly 1 | Wrong >=.99 | AUROC | AURC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in summary["conditions"]:
        lines.append(f"| {c['model']} | {c['temperature']:g} | {c['correct']}/{c['calls']} ({c['accuracy']:.1%}) | {fmt(c['likelihood']['mean'])} | {fmt(c['likelihood']['median'])} | {c['exact_one_count']} | {c['wrong_ge_099_count']} | {fmt(c['by_correctness']['correctness_ranking_auc'])} | {fmt(c['aurc'])} |")
    lines += ["", "## Score range and saturation", "",
        "| Model | T | Minimum | 10th percentile | Median | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for c in summary["conditions"]:
        d = c["likelihood"]
        lines.append(f"| {c['model']} | {c['temperature']:g} | {d['min']:.9f} | {d['q10']:.9f} | {d['median']:.9f} | {d['max']:.9f} |")
    lines += ["", "## Same-token temperature diagnostic", "",
        "Compare exact token text at output position zero, using only alternatives actually returned "
        "in both calls. Whitespace-prefixed variants remain distinct; a later token after an emitted "
        "newline is not treated as position zero. Missing top-k alternatives are not zero-filled. "
        "same_first_token_logprobs.csv also includes repeated-temperature controls.", "",
        "For T=.5 vs T=1, compare the log odds of the same two distinct semantic labels at position zero. "
        "If logits were fixed and returned probabilities temperature-scaled, the gap ratio would be 2; "
        "unchanged returned gaps would give 1. Hosted-model variations prevent interpreting this as "
        "an API implementation guarantee. Missing pairs are excluded with availability recorded.", "",
        "| Model | Available pairs | Median gap ratio (.5 / 1) | Median error: unchanged gap | Median error: doubled gap |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for model, g in summary["temperature_gap_diagnostics"].items():
        median = lambda field: None if g[field] is None else g[field]["median"]
        lines.append(f"| {model} | {g['available_pairs']}/{g['possible_pairs']} | {fmt(median('ratio'))} | {fmt(median('absolute_error_unchanged'))} | {fmt(median('absolute_error_temperature_scaled'))} |")
    lines += ["", "Same-token variability control below reports the median absolute logprob change "
        "over common exact token entries (including low-probability alternatives). These are "
        "descriptive matched-token summaries, not independent statistical observations.", "",
        "| Comparison | Matched token entries | Median absolute logprob change |",
        "| --- | ---: | ---: |"]
    for name, item in summary["same_token_summary"].items():
        if not name.startswith("model:"):
            d = item["absolute_logprob_delta"]
            lines.append(f"| {name} | {item['common_exact_token_entries']} | {fmt(d['median'] if d else None)} |")
    lines += ["", "## Paired changes (B minus A)", "",
        "Accuracy intervals use a paired bootstrap over scenes, preserving repeated calls as a cluster. "
        "They remain exploratory and do not quantify transfer to new object classes.", "",
        "| Comparison | Changed predictions | Accuracy delta | Scene-bootstrap 95% interval |",
        "| --- | ---: | ---: | --- |"]
    for name, p in summary["paired_comparisons"].items():
        lines.append(f"| {name} | {p['prediction_changes']}/{p['paired_calls']} | {p['accuracy_delta_b_minus_a']:+.1%} | {p['scene_bootstrap_95_interval']} |")
    lines += ["", "## Descriptive thresholds (not selected operating points)", "",
        "| Model | T | Threshold | Coverage | Autonomous accuracy | False accepts / all calls |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for c in summary["conditions"]:
        for point in c["thresholds"]:
            if point["threshold"] in (.7, .9, .95, .99):
                lines.append(f"| {c['model']} | {c['temperature']:g} | {point['threshold']:g} | {point['autonomous_coverage']:.1%} | {fmt(point['autonomous_accuracy'])} | {point['false_autonomous_acceptance_count']}/{point['total_trials']} |")
    lines += ["", "## Limitations and audit", "",
        "A larger score range is not evidence of better correctness ranking. AUROC is undefined if "
        "only one correctness outcome occurs. AURC uses tied threshold endpoints and trapezoidal "
        "interpolation from an empty-coverage origin; interpolated coverage inside a tied group "
        "is not achievable by a threshold. Unavailable scores count as deferred. "
        "No class-conditional score adjustment is used. No absent-target scenes are present. "
        "The paper did not specify its GPT-4o snapshot or temperature: this is not an exact replication. "
        "The first scene checks every condition before bulk work. Calls are interleaved and raw "
        "fingerprints/timestamps retained; temperature-zero and cross-condition variation can "
        "include server-side numerical differences. No production settings changed.", "",
        "Audit: " + json.dumps(summary["audit"]), "",
        "Run: `python -m scripts.run_vlm_model_temperature_pilot run` (resumes saved calls). "
        "Analyze without inference: `python -m scripts.analyze_vlm_model_temperature_pilot`.", "",
        "Official references checked September 6, 2026: "
        "[Chat Completions parameters](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create), "
        "[GPT-4o snapshots](https://developers.openai.com/api/docs/models/gpt-4o), "
        "[GPT-4.1-mini snapshot](https://developers.openai.com/api/docs/models/gpt-4.1-mini). "
        "The API reference describes temperature as a sampling parameter but does not explicitly "
        "specify pre/post-temperature semantics for returned logprobs. The gap comparison here "
        "tests observed behavior in this run, not a documented universal guarantee.", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    payload = json.loads((OUTPUT_ROOT / "records.json").read_text())
    result = analyze(payload)
    save_report(payload, *result)
    for c in result[0]["conditions"]:
        print(f"{c['model']} T={c['temperature']:g}: {c['correct']}/{c['calls']} correct; "
              f"mean likelihood={c['likelihood']['mean']:.9f}; "
              f"AUC={c['by_correctness']['correctness_ranking_auc']}")
    print("Audit:", json.dumps(result[0]["audit"]))
    print("Report:", OUTPUT_ROOT / "analysis" / "REPORT.md")


if __name__ == "__main__":
    main()
