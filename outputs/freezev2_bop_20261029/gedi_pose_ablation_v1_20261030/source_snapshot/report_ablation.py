"""Verify saved ablation evidence and report fixed-cohort official pose recalls.

No model, inference, evaluator, or GT pose reader is launched by this reporter.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_cohort import local_path, validate_cohort, filter_prediction_csv, METRICS, TOOLKIT_REVISION

BASELINE = "outputs/freezev2_bop_20261029/classic_core_base_v2"
ARMS = ("baseline", "repair224", "repair476")


def fs(path):
    text = str(Path(path).absolute())
    return Path("\\\\?\\"+text) if os.name == "nt" and not text.startswith("\\\\?\\") else Path(text)


def key(row):
    return int(row["scene_id"]), int(row.get("im_id", row.get("image_id"))), int(row.get("obj_id", row.get("object_id")))


def same_numbers(left, right, label):
    if len(left) != len(right) or any(not math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12) for a,b in zip(left,right)):
        raise ValueError(label+" differs between CSV, ledger or candidate")


def pose_values(pose, *, candidate=False):
    rotation = pose["R"] if candidate else pose["R_m2c"]
    translation = pose["t"] if candidate else pose["t_m"]
    return [float(pose["score"]), *[float(x) for row in rotation for x in row],
            *[float(x)*1000 for x in translation]]


def csv_values(path):
    with fs(path).open(newline="",encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return sorted([(int(r["scene_id"]),int(r["im_id"]),int(r["obj_id"]),
                    float(r["score"]),*[float(x) for x in r["R"].split()],
                    *[float(x) for x in r["t"].split()],float(r["time"])) for r in rows])


def assert_csv_matches(path, predictions):
    actual = csv_values(path)
    expected = sorted([(*key(p),*pose_values(p),float(p["image_time_s"])) for p in predictions])
    if len(actual) != len(expected):
        raise ValueError("CSV and saved ledger pose counts differ")
    for a,b in zip(actual,expected):
        if a[:3] != b[:3]:raise ValueError("CSV and ledger target IDs differ")
        same_numbers(a[3:],b[3:],"Saved prediction")
    return len(actual)


class Auditor:
    def __init__(self):self.files={}

    def hash(self,path,expected=None):
        path=Path(path)
        if path.name.startswith("scene_gt"):
            raise ValueError("GT pose/visibility files are forbidden in the reporter")
        h=hashlib.sha256(fs(path).read_bytes()).hexdigest()
        if expected is not None and h != expected:raise ValueError("Changed saved evidence: "+str(path))
        self.files[str(path)]=h
        return h

    def read(self,path,expected=None):
        self.hash(path,expected)
        return json.loads(fs(path).read_text(encoding="utf-8-sig"))

    def candidate(self,folder,reference):
        folder=Path(folder);trace=self.read(folder/"diagnostic.json",reference.get("sha256",reference.get("diagnostic_sha256")))
        if trace.get("GT_pose_access",False):raise ValueError("Inference trace accessed GT poses")
        if trace["candidate"]["candidate_id"] != reference["candidate_id"] or trace["status"] != reference["status"]:
            raise ValueError("Candidate identity/status differs from image receipt")
        artifacts=trace.get("artifacts",trace.get("artifact_files",{}))
        for name,h in artifacts.items():
            if Path(name).name != name:raise ValueError("Candidate artifact escapes its directory")
            self.hash(folder/name,h)
        if trace["status"] == "complete":
            product=float(trace["coarse_feature_score"])*float(trace["refined_feature_score"])*float(trace["geometry_score"])
            same_numbers([product,trace["pose"]["score"]],[trace["joint_score"]]*2,"Joint score")
            if "original_CAD_rows" in trace:
                if (trace["original_CAD_rows"] != trace["score_detail"]["cad_count"]
                        or trace["retained_CAD_rows"] != trace["score_detail"]["feature_cad_count"]):
                    raise ValueError("Full-CAD and feature-row denominators were mixed")
                same_numbers([trace["geometry_score"]],
                    [trace["score_detail"]["geometry_inlier_points"]/trace["score_detail"]["cad_count"]],"Full-CAD geometry score")
                if trace["retained_sparse_rows"]!=trace["score_detail"]["sparse_observed_count"]:
                    raise ValueError("Retained sparse feature denominator differs")
                if trace["original_sparse_rows"]-trace["retained_sparse_rows"] != len(trace["removed_sparse_indices"]):
                    raise ValueError("Sparse GeDi row accounting differs")
        return trace

    def scores(self,receipt,expected_instances):
        final=local_path(receipt["score_file"])
        aggregate=self.read(final,receipt["score_sha256"])
        score_files=list(fs(final.parent).rglob("scores*.json"))
        # Native Windows extended paths are normalized before recording hashes.
        score_files=[Path(str(p).removeprefix("\\\\?\\")) for p in score_files]
        thresholds=[p for p in score_files if p.name != "scores_bop19.json"]
        if len(score_files)!=121 or len(thresholds)!=120:
            raise ValueError("Expected120 threshold scores plus1 aggregate BOP19 score")
        groups=Counter()
        for path in thresholds:
            value=self.read(path)
            if value.get("targets_count") != expected_instances:
                raise ValueError("Official score dropped or changed the target-instance denominator")
            recall=float(value["recall"])
            if not math.isfinite(recall) or not 0<=recall<=1:raise ValueError("Invalid official threshold recall")
            groups[path.parent.name.split("_",1)[0]]+=1
        if groups != {"error=vsd":100,"error=mssd":10,"error=mspd":10}:
            raise ValueError("Incomplete official VSD/MSSD/MSPD threshold families")
        result={name:float(aggregate[name]) for name in METRICS}
        for name,value in result.items():
            if not math.isfinite(value) or not 0<=value<=1:raise ValueError("Invalid official AR")
            same_numbers([value],[receipt["metrics"][name]],"Receipt official metric")
        return dict(metrics=result,threshold_score_files=120,all_score_files=121,targets_count_each=expected_instances,
                    gt_count_is_not_the_requested_instance_denominator=True)


def audit_image_ledgers(audit,items,targets,*,run_digest=None):
    expected={key(t):int(t["inst_count"]) for t in targets};found={};predictions=[];images=set()
    for path,original in items:
        ledger=audit.read(path,original)
        if ledger.get("GT_pose_access",False):raise ValueError("Image inference receipt accessed GT poses")
        if run_digest and ledger["identity"]["run"] != run_digest:raise ValueError("Image belongs to another run")
        scene,image=map(int,Path(path).parent.name.split("_"));images.add((scene,image))
        chosen={key(p):[] for p in ledger["predictions"]}
        for p in ledger["predictions"]:
            if key(p) in expected:chosen.setdefault(key(p),[]).append(p)
        for record in ledger["targets"]:
            obj=int(record.get("obj_id",record.get("object_id")));k=(scene,image,obj)
            if k not in expected:continue
            if k in found:raise ValueError("Duplicate cohort target ledger")
            count=int(record.get("inst_count",record.get("requested_instances")))
            if count != expected[k]:raise ValueError("Ledger changes official inst_count")
            selected=chosen.get(k,[])
            if len(selected) != int(record.get("predictions",record.get("valid_predictions"))):
                raise ValueError("Ledger pose count differs from retained predictions")
            candidates=[];by_id={}
            for reference in record.get("candidates",[]):
                directory=Path(str(reference["directory"]).replace("\\","/"))
                if directory.is_absolute() or ".." in directory.parts:raise ValueError("Escaping candidate directory")
                trace=audit.candidate(Path(path).parent/directory,reference)
                by_id[reference["candidate_id"]]=trace
                candidates.append({"candidate_id":reference["candidate_id"],"status":trace["status"],
                    "selected":reference["candidate_id"] in {p["candidate_id"] for p in selected},
                    "coarse_feature_score":trace.get("coarse_feature_score"),"refined_feature_score":trace.get("refined_feature_score"),
                    "geometry_score":trace.get("geometry_score"),"joint_score":trace.get("joint_score"),
                    "original_sparse_rows":trace.get("original_sparse_rows",trace.get("sparse_points")),
                    "retained_sparse_rows":trace.get("retained_sparse_rows",trace.get("sparse_points")),
                    "removed_sparse_indices":trace.get("removed_sparse_indices",[]),
                    "runtime_s":trace.get("runtime_s"),"failure":trace.get("reason"),
                    "mask_sha256":trace.get("artifacts",trace.get("artifact_files",{})).get("mask.png"),
                    "mask_source_sha256":trace["candidate"]["source_file_sha256"],
                    "diagnostic":str(Path(path).parent/directory/"diagnostic.json")})
            for p in selected:
                trace=by_id[p["candidate_id"]]
                if trace["status"]!="complete":raise ValueError("Selected pose has a failed candidate trace")
                if trace["pose"]["obj_id"]!=obj:raise ValueError("Selected candidate uses another CAD identity")
                same_numbers(pose_values(p),pose_values(trace["pose"],candidate=True),"Selected candidate pose")
                predictions.append(p)
            found[k]={"scene_id":scene,"im_id":image,"obj_id":obj,"requested_instances":count,
                "status":record["status"],"saved_predictions":len(selected),"candidates":candidates,
                "failure":record.get("failure",record.get("reason")),"suppressed":record.get("suppressed",[]),
                "rejected":record.get("rejected",[]),"image_active_runtime_s":ledger["runtime_s"]}
    if set(found)!=set(expected):raise ValueError("Missing cohort target ledgers, including failures")
    return [found[k] for k in sorted(found)],predictions


def audit_onboarding(audit,folder,required,rows):
    result={}
    for obj in sorted(required):
        directory=folder/f"onboarding/obj_{obj:06d}";path=directory/"onboarding.json"
        if not path.is_file():
            if any(r["obj_id"]==obj and r["status"]!="blocked" for r in rows):
                raise ValueError("Missing onboarding receipt for a processed CAD")
            result[str(obj)]={"status":"blocked"};continue
        value=audit.read(path);complete=audit.read(directory/"complete.json")
        audit.hash(directory/"arrays.npz",complete["sha256"])
        n=value["original_points"];m=value["retained_feature_points"];removed=value["removed_original_indices"]
        if n!=value["PCA_fit_rows"] or n!=value["ICP_geometry_rows"] or n!=value["geometry_score_denominator"]:
            raise ValueError("Full original CAD/PCA/ICP geometry denominator changed")
        if m+len(removed)!=n or len(set(removed))!=len(removed) or any(i<0 or i>=n for i in removed):
            raise ValueError("Invalid CAD finite-row accounting")
        result[str(obj)]={"status":"complete",**value}
    return result


def audit_evaluation(audit,path,cohort_path,cohort,targets,predictions,source_csv):
    receipt=audit.read(path)
    if receipt["status"]!="complete" or receipt["source_pins"]["toolkit_revision"]!=TOOLKIT_REVISION:
        raise ValueError("Official cohort evaluator has not completed with the pinned toolkit")
    pins=receipt["source_pins"]
    audit.hash(cohort_path,pins["cohort_sha256"])
    audit.hash(source_csv,pins["input_csv_sha256"])
    evaluated=local_path(receipt["evaluated_csv"]);audit.hash(evaluated,pins["evaluated_csv_sha256"])
    subset=audit.read(local_path(receipt["targets_path"]),pins["cohort_targets_sha256"])
    if subset!=sorted(targets,key=key) or receipt["requested_instances"]!=sum(t["inst_count"] for t in targets):
        raise ValueError("Evaluated cohort denominator differs from fixed targets")
    if pins["official_targets_sha256"]!=cohort["official_targets"]["sha256"]:
        raise ValueError("Official target source pin differs")
    if receipt.get("visib_gt_min")!=-1:raise ValueError("Unexpected evaluation visibility filter")
    assert_csv_matches(evaluated,predictions)
    scores=audit.scores(receipt,sum(t["inst_count"] for t in targets))
    return {**scores,"receipt":str(path),"receipt_sha256":audit.files[str(path)],
            "score_file":receipt["score_file"],"score_sha256":receipt["score_sha256"],
            "evaluation_runtime_s":receipt.get("runtime_s"),"source_pins":pins}


def report(root:Path,config_path:Path):
    root,config_path=local_path(root),local_path(config_path);audit=Auditor()
    config=audit.read(config_path);cohort_path=config_path.with_name("cohort.json");cohort=audit.read(cohort_path)
    out=local_path(config["output"]);baseline=root/BASELINE
    manifest=audit.read(out/"run_manifest.json");identity=manifest["identity"]
    audit.hash(config_path,identity["config_sha256"]);audit.hash(cohort_path,identity["cohort_sha256"])
    original_results=audit.read(baseline/"final_results.json",identity["baseline_final_results_sha256"])
    source=audit.read(baseline/"source_snapshot/source_manifest.json",identity["frozen_source_manifest_sha256"])
    for rel,h in source["files"].items():audit.hash(baseline/"source_snapshot"/rel,h)
    for name,h in identity["sources"].items():audit.hash(config_path.parent/name,h)
    official_path=root/cohort["official_targets"]["repo_relative_path"]
    targets=validate_cohort(cohort,audit.read(official_path,cohort["official_targets"]["sha256"]),"tless")
    if len(targets)!=7 or sum(t["inst_count"] for t in targets)!=7:raise ValueError("Expected the frozen seven-target diagnostic cohort")
    old_items={}
    for case in cohort["cases"]:
        item=case["baseline"];path=root/item["ledger_repo_relative_path"]
        if path in old_items and old_items[path]!=item["ledger_sha256"]:raise ValueError("Conflicting original ledger hashes")
        old_items[path]=item["ledger_sha256"]
    old_rows,old_poses=audit_image_ledgers(audit,list(old_items.items()),targets)
    if len(old_poses)!=3:raise ValueError("The frozen baseline cohort must contain exactly3 saved poses")
    old_csv=baseline/"datasets/tless/predictions/freezev2base_tless-test.csv"
    audit.hash(old_csv,original_results["datasets"]["tless"]["csv_sha256"])
    filtered,_=filter_prediction_csv(old_csv,targets)
    baseline_eval=audit_evaluation(audit,out/"baseline/evaluation/evaluation_receipt.json",cohort_path,cohort,targets,old_poses,old_csv)
    if fs(local_path(audit.read(out/"baseline/evaluation/evaluation_receipt.json")["evaluated_csv"])).read_bytes()!=filtered:
        raise ValueError("Baseline subset CSV differs from the unchanged full-run CSV")
    results={"baseline":{"rows":old_rows,"saved_predictions":len(old_poses),"evaluation":baseline_eval,
                         "runtime_scope":"Original full-scene timings include work outside the diagnostic cohort"}}
    for arm in ("repair224","repair476"):
        folder=out/arm;summary=audit.read(folder/"inference_summary.json")
        items=[(p,None) for p in sorted((folder/"images").glob("*/complete.json"))]
        rows,predictions=audit_image_ledgers(audit,items,targets,run_digest=identity["digest"])
        summary_rows={key(t):t for t in summary["targets"]}
        if len(summary["targets"])!=7 or set(summary_rows)!={key(t) for t in targets}:
            raise ValueError("Inference summary omits or duplicates cohort targets")
        for row in rows:
            item=summary_rows[key(row)]
            if (item["inst_count"]!=row["requested_instances"] or item["status"]!=row["status"]
                    or item["predictions"]!=row["saved_predictions"]):
                raise ValueError("Inference summary target status/count differs from saved image ledger")
        if (summary["requested_instances"]!=7 or summary["expected_target_rows"]!=7
                or summary["predictions"]!=len(predictions) or summary["GT_pose_access"]):
            raise ValueError("Inference summary coverage differs")
        csv_path=local_path(summary["CSV"]);audit.hash(csv_path,summary["CSV_sha256"]);assert_csv_matches(csv_path,predictions)
        onboards=audit_onboarding(audit,folder,{t["obj_id"] for t in targets},rows)
        evaluation=audit_evaluation(audit,folder/"evaluation/evaluation_receipt.json",cohort_path,cohort,targets,predictions,csv_path)
        results[arm]={"rows":rows,"saved_predictions":len(predictions),"additional_valid_poses_vs_baseline":len(predictions)-len(old_poses),
            "onboarding":onboards,"evaluation":evaluation,
            "runtime_s":{"saved_onboarding_stage_sum":sum(v.get("runtime_s",0) for v in onboards.values()),
                         "image_active_sum":sum(audit.read(p)["runtime_s"] for p,_ in items),
                         "candidate_active_sum":sum(c.get("runtime_s",0) for row in rows for c in row["candidates"])}}
    for left,right in zip(results["repair224"]["rows"],results["repair476"]["rows"]):
        a={c["candidate_id"]:c for c in left["candidates"]};b={c["candidate_id"]:c for c in right["candidates"]}
        if set(a)!=set(b):raise ValueError("Resolution arms used different predicted-mask candidates")
        for cid in a:
            if (a[cid]["mask_sha256"]!=b[cid]["mask_sha256"] or a[cid]["mask_source_sha256"]!=b[cid]["mask_source_sha256"]
                    or a[cid]["removed_sparse_indices"]!=b[cid]["removed_sparse_indices"]):
                raise ValueError("Observed masks or geometric sparse-row policy changed between resolution arms")
    for arm in ARMS:
        for row in results[arm]["rows"]:
            if any(c["mask_source_sha256"]!=original_results["datasets"]["tless"]["mask_sha256"] for c in row["candidates"]):
                raise ValueError("A candidate changed the frozen public predicted-mask source")
    reporter_sha256=audit.hash(Path(__file__))
    supporting={}
    for name in ("implementation_validation.json","pypng_runtime.json"):
        if (out/name).is_file():supporting[name]=audit.read(out/name)
    validation=supporting.get("implementation_validation.json",{})
    for name,field in (("final_results.json","baseline_final_results_sha256"),
                       ("FINAL_REPORT.md","baseline_final_report_sha256")):
        if field in validation:audit.hash(baseline/name,validation[field])
    values={"schema":"freezev2-finite-row-CAD-resolution-ablation-report-v1","dataset":"tless","target_rows":7,
        "requested_instances":7,"images":len({key(t)[:2] for t in targets}),"results":results,
        "AR_delta_percentage_points":{"repair224_vs_baseline":100*(results["repair224"]["evaluation"]["metrics"][METRICS[0]]-baseline_eval["metrics"][METRICS[0]]),
            "repair476_vs_repair224":100*(results["repair476"]["evaluation"]["metrics"][METRICS[0]]-results["repair224"]["evaluation"]["metrics"][METRICS[0]])},
        "claim_scope":config["claim_scope"],"task":"BOP19 model-based localization AR; supplied object IDs/counts; no AP or CAD-classification claim",
        "selection_limit":"Fixed numerical-failure-enriched seven-row cohort plus execution-valid controls; not representative full-dataset performance",
        "pose_count_limit":"Additional valid-format predictions are not necessarily GT-correct poses",
        "runtime_limit":"Saved active stage timings have cache effects; baseline full-scene timings are not a fair cohort runtime comparison",
        "GT_pose_reads_by_reporter":False,"source_identity_unchanged":True,"reporter_sha256":reporter_sha256,"supporting_validation":supporting,
        "verified_file_hashes":audit.files}
    lines=["# Controlled GeDi repair and CAD-resolution diagnostic","",values["claim_scope"],"",
           "Seven fixed T-LESS object-image targets / seven requested instances / six images. All failed or blocked targets remain in official evaluation. This cohort was enriched for recorded numerical failures; it is not the full T-LESS benchmark.","",
           "| Arm | Saved poses /7 | AR | AR VSD | AR MSSD | AR MSPD |","|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        item=results[arm];m=item["evaluation"]["metrics"]
        lines.append(f"| {arm} | {item['saved_predictions']}/7 | "+" | ".join(f"{100*m[k]:.4f}%" for k in METRICS)+" |")
    lines += ["","These are official BOP19 localization recalls. A saved pose is a valid-format output, not a statement that the pose is correct. Additional predictions may still have substantial pose error. No AP, CAD-identification accuracy, public-repository parity or paper-performance reproduction is claimed.","",
              "| Scene/image/object | Baseline poses | Repair224 poses | Repair476 poses |","|---|---:|---:|---:|"]
    for i,t in enumerate(targets):
        lines.append(f"| {t['scene_id']:06d}/{t['im_id']:06d}/{t['obj_id']:06d} | "+" | ".join(str(results[a]["rows"][i]["saved_predictions"]) for a in ARMS)+" |")
    lines += ["","All120 per-threshold official score files in each arm retain targets_count=7; each arm also has its verified aggregate score file. CSV poses match saved ledgers to absolute tolerance1e-12 in BOP millimetres. Candidate trace/artifact hashes, frozen baseline sources/results, and full original CAD/PCA/ICP geometry denominators were checked.","",
              "Per-target candidate ranking, selected/suppressed poses, coarse/refined/geometric scores, finite-row counts, failure reasons, stage runtimes and provenance hashes are retained in results_summary.json. Representative visual inspection is recorded separately by the supervising reviewer.","",
              "The repair preserves original full CAD points for ICP and geometric scoring. CAD476 changes only rendered-CAD visual preprocessing; observed DINO remains224. Parameters were fixed before cohort metrics. Both runs remain independent implementations with recorded assumptions.","",
              "Runtime sums include saved cache effects. Original baseline CSV/image timing includes full-scene work outside this cohort, so no baseline speedup ratio is reported.",""]
    fs(out/"results_summary.json").write_text(json.dumps(values,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    fs(out/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path(__file__).with_name("config.json"));args=parser.parse_args()
    result=report(args.root,args.config)
    print(json.dumps({"output":str(local_path(json.loads(args.config.read_text())["output"])),"target_rows":result["target_rows"],
                      "saved_predictions":{a:v["saved_predictions"] for a,v in result["results"].items()},
                      "official_AR":{a:v["evaluation"]["metrics"][METRICS[0]] for a,v in result["results"].items()}},indent=2))


if __name__=="__main__":main()
