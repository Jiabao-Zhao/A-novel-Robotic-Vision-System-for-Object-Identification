"""Analyze paired output-format calls; only development splits select thresholds."""

from collections import Counter
from datetime import datetime
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from scripts.analyze_vlm_uncertainty_scores import (
    risk_coverage_curve, score_distribution, select_development_operating_point,
    threshold_metrics,
)
from scripts.run_vlm_output_format_ablation import OUTPUT_ROOT, CONDITIONS, FROZEN_ROOT


SCORE = "raw_sequence_likelihood"
SPLITS = ("calibration", "validation", "test")


def validate_pairs(records):
    grouped = {}
    for row in records:
        key = (row["sample_id"], row["condition"])
        if key in grouped:
            raise ValueError(f"Duplicate response: {key}")
        grouped[key] = row
        decision = row["decision_tokens"]
        if row[SCORE] is not None:
            expected = math.exp(math.fsum(item["logprob"] for item in decision))
            if expected != row[SCORE]:
                raise ValueError("Sequence probability does not equal saved token product.")
        if len(decision) != row["decision_token_count"]:
            raise ValueError("Decision token count mismatch.")
        expected_correct = row["choice"] is not None and row["predicted_object_id"] == row["ground_truth_object_id"]
        if row["correct"] != expected_correct:
            raise ValueError("Correctness does not match the frozen ground-truth ID.")
        if row["choice"] is not None and row["candidate_map"][row["choice"]] != row["predicted_object_id"]:
            raise ValueError("Prediction differs from the generated choice.")
    pairs = []
    for sample_id in sorted({row["sample_id"] for row in records}):
        if any((sample_id, condition) not in grouped for condition in CONDITIONS):
            raise ValueError(f"Incomplete pair: {sample_id}")
        a, b = (grouped[sample_id, condition] for condition in CONDITIONS)
        for field in ("image_sha256", "localization_sha256", "target_description",
                      "ground_truth_object_id", "partition", "model"):
            if a[field] != b[field]:
                raise ValueError(f"Paired {field} mismatch for {sample_id}")
        if list(a["candidate_map"].values()) != list(b["candidate_map"].values()):
            raise ValueError(f"Candidate set or order changed for {sample_id}")
        pairs.append((a, b))
    return pairs


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return {"count": len(values), "mean": float(values.mean()),
            **{key: float(np.quantile(values, q)) for key, q in
               (("min", 0), ("q10", .1), ("q25", .25), ("median", .5), ("q75", .75), ("q90", .9), ("max", 1))}}


def metrics(rows):
    available = [row for row in rows if row[SCORE] is not None]
    curve = risk_coverage_curve(rows, SCORE)
    errors = [row for row in available if not row["correct"]]
    return {
        "n": len(rows), "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows)/len(rows),
        "score_availability": len(available)/len(rows),
        "mean_decision_token_count": float(np.mean([row["decision_token_count"] for row in rows])),
        "decision_token_count_histogram": dict(Counter(str(row["decision_token_count"]) for row in rows)),
        "likelihood": distribution([row[SCORE] for row in available]),
        "by_correctness": score_distribution(rows, SCORE),
        "aurc": curve["aurc_over_operational_coverage"],
        "exact_one_scores": sum(row[SCORE] == 1.0 for row in rows),
        "incorrect_with_score_ge_099": sum(row[SCORE] >= .99 for row in errors),
        "incorrect_with_score_equal_one": sum(row[SCORE] == 1.0 for row in errors),
    }


def paired_accuracy(pairs):
    delta = np.array([int(b["correct"]) - int(a["correct"]) for a,b in pairs])
    improved, worsened = int((delta == 1).sum()), int((delta == -1).sum())
    rng = np.random.default_rng(20260904)
    # Pair bootstrap describes these scenes and fixed task mix, not unseen classes.
    boot = rng.choice(delta, size=(5000,len(delta)), replace=True).mean(axis=1)
    return {
        "n": len(pairs), "same_prediction": sum(a["predicted_object_id"] == b["predicted_object_id"] for a,b in pairs),
        "both_correct": sum(a["correct"] and b["correct"] for a,b in pairs),
        "both_wrong": sum(not a["correct"] and not b["correct"] for a,b in pairs),
        "compact_wrong_id_correct": improved, "compact_correct_id_wrong": worsened,
        "accuracy_difference_id_minus_compact": float(delta.mean()),
        "paired_bootstrap_95_interval": np.quantile(boot,[.025,.975]).tolist(),
        "mcnemar_exact_two_sided_p": float(binomtest(improved, improved+worsened).pvalue) if improved+worsened else 1.0,
    }


def length_analysis(pairs):
    ids = [b for a,b in pairs if b["choice"] is not None and b["choice"] != "none" and b[SCORE] is not None]
    prefix = [math.exp(math.fsum(token["logprob"] for token in row["decision_tokens"][:-1])) for row in ids]
    suffix = [math.exp(row["decision_tokens"][-1]["logprob"]) for row in ids]
    same = [(a,b) for a,b in pairs if a["predicted_object_id"] == b["predicted_object_id"] and a[SCORE] is not None and b[SCORE] is not None]
    return {
        "tokenizations": dict(Counter(" | ".join(token["token"] for token in row["decision_tokens"]) for row in ids)),
        "object_id_prefix_likelihood": distribution(prefix),
        "object_id_final_token_likelihood": distribution(suffix),
        "object_id_prefix_penalty_over_one_percent": sum(value < .99 for value in prefix),
        "object_id_same_prediction_pairs": len(same),
        "same_prediction_id_likelihood_lower": sum(b[SCORE] < a[SCORE] for a,b in same),
        "same_prediction_id_likelihood_higher": sum(b[SCORE] > a[SCORE] for a,b in same),
        "same_prediction_id_likelihood_equal": sum(b[SCORE] == a[SCORE] for a,b in same),
        "same_prediction_likelihood_delta_id_minus_compact": distribution([b[SCORE]-a[SCORE] for a,b in same]),
        "interpretation": "Prefix/final-token decomposition is diagnostic only. The gate always uses the whole sequence product. Changed representation also changes conditional probabilities; sequence length alone is not an isolated causal treatment.",
    }


def analyze(records):
    pairs = validate_pairs(records)
    if len(pairs) != 500:
        raise ValueError("Full analysis requires 500 complete pairs.")
    summary = {"protocol": json.loads((OUTPUT_ROOT/"protocol.json").read_text()),
               "paired_accuracy": {}, "conditions": {}, "class_results": [],
               "length_analysis": length_analysis(pairs)}
    summary["execution_audit"] = {
        "complete_pairs": len(pairs), "identical_paired_inputs_verified": True,
        "returned_models": dict(Counter(row["model"] for row in records)),
        "system_fingerprints": dict(Counter(row["provider_response"].get("system_fingerprint", "unavailable") for row in records)),
        "finish_reasons": dict(Counter(row["provider_response"]["choices"][0]["finish_reason"] for row in records)),
        "candidate_counts_excluding_none": dict(Counter(str(len(row["candidate_map"])-1) for row in records)),
        "invalid_choices": sum(row["choice"] is None for row in records),
        "score_unavailable": sum(row[SCORE] is None for row in records),
        "pair_capture_gap_seconds": distribution([abs((datetime.fromisoformat(b["captured_at"])-datetime.fromisoformat(a["captured_at"])).total_seconds()) for a,b in pairs]),
        "total_prompt_tokens": sum(row["provider_response"].get("usage",{}).get("prompt_tokens",0) for row in records),
        "total_completion_tokens": sum(row["provider_response"].get("usage",{}).get("completion_tokens",0) for row in records),
    }
    curves = []
    for split in ("all", *SPLITS):
        subset = pairs if split == "all" else [(a,b) for a,b in pairs if a["partition"] == split]
        summary["paired_accuracy"][split] = paired_accuracy(subset)
    for condition in CONDITIONS:
        rows = [row for row in records if row["condition"] == condition]
        partitions = {split:[row for row in rows if row["partition"] == split] for split in SPLITS}
        if {key:len(value) for key,value in partitions.items()} != {"calibration":300,"validation":100,"test":100}:
            raise ValueError("Unexpected frozen splits.")
        results = {"all": metrics(rows), **{split:metrics(items) for split,items in partitions.items()}}
        points = {}
        for accuracy in (.9,.95):
            selected = select_development_operating_point(partitions["calibration"], partitions["validation"], SCORE, accuracy)
            if selected is not None:
                selected["test"] = threshold_metrics(partitions["test"], SCORE, selected["threshold"])
            points[str(accuracy)] = selected
        results["development_selected_operating_points"] = points
        summary["conditions"][condition] = results
        for split,items in {"all":rows, **partitions}.items():
            curves.extend({"condition":condition,"split":split,**point} for point in risk_coverage_curve(items,SCORE)["points"])
        for target in sorted({row["target_description"] for row in rows}):
            for split in ("all", "test"):
                subset = [row for row in rows if row["target_description"] == target and (split == "all" or row["partition"] == split)]
                selected = points["0.95"]
                summary["class_results"].append({"condition":condition,"target":target,"split":split,
                    "n":len(subset),"correct":sum(row["correct"] for row in subset),
                    "wrong_ge_099":sum(not row["correct"] and row[SCORE] is not None and row[SCORE]>=.99 for row in subset),
                    "gate_95": threshold_metrics(subset,SCORE,selected["threshold"]) if selected else None})
    # Historical control is secondary: it lacks detailed token outputs and was
    # captured at another time, so it cannot replace fresh paired condition A.
    historical = {row["sample_id"]:row for row in json.loads((FROZEN_ROOT/"association_records.json").read_text())["records"]}
    summary["historical_control_check"] = {"fresh_compact_prediction_changes": sum(a["predicted_object_id"] != historical[a["sample_id"]]["predicted_object_id"] for a,b in pairs),
        "historical_accuracy": sum(row["correct"] for row in historical.values())/500,
        "interpretation": "Temperature zero does not guarantee identical reruns; conclusions compare contemporaneous A/B pairs."}
    return summary, curves


def save_outputs(records, summary, curves):
    import csv

    analysis_root = OUTPUT_ROOT/"analysis"
    analysis_root.mkdir(exist_ok=True)
    (analysis_root/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    fields = ["sample_id","condition","partition","target_description","ground_truth_object_id",
              "predicted_object_id","correct","generated_output_text","decision_tokens",
              "decision_token_count","raw_log_probability",SCORE]
    with (analysis_root/"associations.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        writer.writerows({key:json.dumps(row[key]) if key=="decision_tokens" else row[key] for key in fields} for row in records)
    with (analysis_root/"threshold_curves.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    wrong=[{key:row[key] for key in fields} for row in records if not row["correct"] and row[SCORE] is not None and row[SCORE]>=.99]
    (analysis_root/"confidently_wrong.json").write_text(json.dumps(wrong,indent=2),encoding="utf-8")
    colors={"compact":"#0072B2","object_id":"#D55E00"}
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for axis,split in zip(axes[0],("validation","test")):
        for condition in CONDITIONS:
            points=[row for row in curves if row["condition"]==condition and row["split"]==split]
            axis.plot([row["autonomous_coverage"] for row in points], [row["autonomous_accuracy"] for row in points],".-",color=colors[condition],label=condition)
        axis.set(title=f"{split.capitalize()}: accuracy versus coverage",xlabel="Autonomous coverage",ylabel="Autonomous accuracy",xlim=(0,1.02),ylim=(.65,1.01))
        axis.axhline(.95,color="0.5",ls=":")
        axis.legend()
    for condition in CONDITIONS:
        for correct in (True,False):
            scores=[row[SCORE] for row in records if row["condition"]==condition and row["correct"]==correct and row[SCORE] is not None]
            transformed=-np.log10(np.maximum(1-np.array(scores),1e-15))
            axes[1,0].hist(transformed,bins=np.linspace(0,15,31),histtype="step",density=True,linestyle="-" if correct else "--",color=colors[condition],label=f"{condition}: {'correct' if correct else 'wrong'}")
    axes[1,0].set(title="All trials: raw sequence likelihood",xlabel="−log10(1 − likelihood); exact 1 capped at 15",ylabel="Density")
    axes[1,0].legend(fontsize=8)
    rows=[row for row in records if row["condition"]=="object_id" and row["choice"] not in (None,"none") and row[SCORE] is not None]
    prefix=[-math.fsum(token["logprob"] for token in row["decision_tokens"][:-1]) for row in rows]
    suffix=[-row["decision_tokens"][-1]["logprob"] for row in rows]
    axes[1,1].boxplot([prefix,suffix],tick_labels=["ID prefix tokens","Final ID token"],showfliers=True)
    axes[1,1].set(title="Object-ID output: token contributions",ylabel="Negative log likelihood (symlog)")
    axes[1,1].set_yscale("symlog",linthresh=1e-7)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    fig.suptitle("Paired GPT-4.1-mini output-format ablation | identical frozen LIBERO images")
    fig.tight_layout()
    fig.savefig(analysis_root/"format_comparison.png",dpi=170)
    plt.close(fig)

    lines=["# Paired VLM output-format ablation", "",
        "500 frozen LIBERO scenes; 1,000 fresh responses including a 20-scene pilot. "
        "Both formats use identical image bytes, localization, targets, candidate "
        "order and ground truth. GPT-4.1-mini-2025-04-14, temperature 0, image "
        "detail high, max_completion_tokens 8 and top_logprobs 20 are identical. "
        "Top alternatives are archived but never used for prediction or scoring.", "",
        "The only prompt changes are candidate `choice` values, the output-choice "
        "list, and N versus none. The images already display persistent object IDs. "
        "The production VLM, HITL, CAD, and planner are unchanged.", "",
        "## Accuracy and score ranking", "",
        "| Format | Split | Correct | Accuracy | AUROC | AURC | Mean decision tokens | Mean likelihood |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for condition in CONDITIONS:
        for split in ("all", *SPLITS):
            m=summary["conditions"][condition][split]
            lines.append(f"| {condition} | {split} | {m['correct']}/{m['n']} | {m['accuracy']:.4f} | {m['by_correctness']['correctness_ranking_auc']:.4f} | {m['aurc']:.4f} | {m['mean_decision_token_count']:.2f} | {m['likelihood']['mean']:.6f} |")
    lines.extend(["", "Raw sequence likelihood is not a calibrated probability of correct association. "
        "In particular, a provider-reported likelihood of 1 does not certify correctness.", "",
        "| Format | Mean likelihood: correct | Mean likelihood: wrong | Median likelihood: wrong | Wrong with likelihood >=0.99 | Wrong with likelihood exactly 1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for condition in CONDITIONS:
        m=summary["conditions"][condition]["all"]
        d=m["by_correctness"]
        lines.append(f"| {condition} | {d['correct']['mean']:.8f} | {d['incorrect']['mean']:.8f} | {d['incorrect']['median']:.8f} | {m['incorrect_with_score_ge_099']} | {m['incorrect_with_score_equal_one']} |")
    paired=summary["paired_accuracy"]["all"]
    lines.extend(["", f"Paired changes: compact-wrong/ID-correct = {paired['compact_wrong_id_correct']}; "
        f"compact-correct/ID-wrong = {paired['compact_correct_id_wrong']}. "
        f"ID-minus-compact accuracy difference = {paired['accuracy_difference_id_minus_compact']:.4f}; "
        f"paired bootstrap 95% interval = {paired['paired_bootstrap_95_interval']}; "
        f"exact McNemar p = {paired['mcnemar_exact_two_sided_p']:.6g}.", "",
        "## Independently selected operating points", "",
        "Each threshold maximizes validation coverage while meeting the requested "
        "accuracy on both calibration and validation. All thresholds come from "
        "these development splits; the previous production threshold is unused. "
        "Test curves are descriptive, not a basis for threshold selection.", "",
        "| Format | Development accuracy target | Fresh threshold | Validation accuracy / coverage | Test accuracy / coverage | Test false accepts | False accepts / all test | False accepts / accepted |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |"])
    for condition in CONDITIONS:
        for target,point in summary["conditions"][condition]["development_selected_operating_points"].items():
            if point is None:
                lines.append(f"| {condition} | {target} | unavailable | — | — | — | — | — |")
                continue
            v,t=point["validation"],point["test"]
            lines.append(f"| {condition} | {target} | {point['threshold']!r} | {v['autonomous_accuracy']:.4f} / {v['autonomous_coverage']:.4f} | {t['autonomous_accuracy']:.4f} / {t['autonomous_coverage']:.4f} | {t['false_autonomous_acceptance_count']} | {t['false_autonomous_acceptance_rate_all_trials']:.4f} | {t['false_autonomous_acceptance_rate_autonomous']:.4f} |")
    length=summary["length_analysis"]
    lines.extend(["", "## Token length and likelihood", "",
        "The recorded score is exp(sum of all decision-token logprobs), with no "
        "length normalization. Separable formatting tokens are excluded; underscores "
        "inside IDs are decision tokens. No candidate normalization is computed.", "",
        f"ID-prefix likelihood: {json.dumps(length['object_id_prefix_likelihood'])}", "",
        f"The ID prefix costs more than 1% probability in {length['object_id_prefix_penalty_over_one_percent']} ID outputs. "
        f"Among {length['object_id_same_prediction_pairs']} pairs with the same predicted object, "
        f"ID likelihood is lower in {length['same_prediction_id_likelihood_lower']}, "
        f"higher in {length['same_prediction_id_likelihood_higher']}, and equal in "
        f"{length['same_prediction_id_likelihood_equal']}.", "",
        length["interpretation"], "", "## Class-specific failures", "",
        "| Target | Format | Split | Correct / trials | Wrong with raw likelihood >=0.99 |",
        "| --- | --- | --- | ---: | ---: |"])
    for item in summary["class_results"]:
        lines.append(f"| {item['target']} | {item['condition']} | {item['split']} | {item['correct']}/{item['n']} | {item['wrong_ge_099']} |")
    lines.extend(["", "## Limitations and reproducibility", "",
        "These are the same ten object tasks with different initial states, not "
        "unseen object categories. Scene-bootstrap intervals assume scene-level "
        "independence and do not quantify transfer to new classes. A single output "
        "per condition and scene does not estimate within-scene API variability. "
        "Threshold ties are kept intact. AURC uses trapezoidal integration at "
        "tied threshold endpoints with the empty-coverage origin; its interpolation "
        "within a large tied group is a convention, not an attainable threshold.", "",
        f"Fresh compact predictions differ from historical saved predictions in "
        f"{summary['historical_control_check']['fresh_compact_prediction_changes']}/500 scenes. "
        "Historical scores are not substituted into the paired comparison.", "",
        "All 1,000 responses used the same pinned model and terminated normally, "
        "with seven localized candidates plus NONE, valid choices, and available "
        "decision logprobs. No scene has an absent ground-truth target, so NONE "
        "is supported but its detection accuracy is not evaluated by this dataset.", "",
        f"The API returned {len(summary['execution_audit']['system_fingerprints'])} backend fingerprints "
        "despite the fixed model snapshot. Fingerprints and timestamps are archived. "
        "An API rate-limit interruption was handled by resuming only missing responses "
        "with pacing; no successful response was replaced. Consequently, some pairs "
        "were separated in time: median gap "
        f"{summary['execution_audit']['pair_capture_gap_seconds']['median']:.1f} seconds, maximum "
        f"{summary['execution_audit']['pair_capture_gap_seconds']['max']:.1f} seconds. "
        "Server-side variation remains a limitation of this hosted-model ablation.", "",
        "Raw responses, exact prompts, token lists, hashes, API model/fingerprint, "
        "and usage are in ../responses/. Numeric tables and plots are in this folder.", "",
        "Run from the repository root in the existing WSL environment:", "",
        "```sh", "python -m scripts.run_vlm_output_format_ablation pilot",
        "python -m scripts.run_vlm_output_format_ablation full",
        "python -m scripts.analyze_vlm_output_format_ablation", "```", ""])
    (analysis_root/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    payload=json.loads((OUTPUT_ROOT/"full_records.json").read_text())
    summary,curves=analyze(payload["records"])
    save_outputs(payload["records"],summary,curves)
    for condition in CONDITIONS:
        values=summary["conditions"][condition]
        print(condition,json.dumps({"all":values["all"],"test":values["test"],"operating_points":values["development_selected_operating_points"]}))
    print("Paired accuracy:",json.dumps(summary["paired_accuracy"]))
    print("Token length:",json.dumps(summary["length_analysis"]))


if __name__ == "__main__":
    main()
