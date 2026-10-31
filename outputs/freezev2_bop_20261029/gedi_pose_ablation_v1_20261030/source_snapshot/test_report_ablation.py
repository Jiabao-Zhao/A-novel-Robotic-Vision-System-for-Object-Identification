"""Focused CPU evidence checks; no evaluator, model or GT pose reads."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import report_ablation as report


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf-8")
    return path


def h(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction():
    return {"scene_id":1,"image_id":2,"object_id":27,"score":.024,
            "R_m2c":[[1,0,0],[0,1,0],[0,0,1]],"t_m":[.1,.2,.5],
            "image_time_s":1.25,"candidate_id":"candidate-1"}


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.audit=report.Auditor()

    def csv_file(self,translation="100 200 500"):
        path=self.root/"results.csv"
        with path.open("w",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow(["scene_id","im_id","obj_id","score","R","t","time"])
            writer.writerow([1,2,27,.024,"1 0 0 0 1 0 0 0 1",translation,1.25])
        return path

    def test_bop_millimetres_match_metre_ledger(self):
        self.assertEqual(report.assert_csv_matches(self.csv_file(),[prediction()]),1)

    def test_wrong_translation_units_rejected(self):
        with self.assertRaisesRegex(ValueError,"differs"):
            report.assert_csv_matches(self.csv_file(".1 .2 .5"),[prediction()])

    def test_missing_ledger_prediction_rejected(self):
        with self.assertRaisesRegex(ValueError,"counts differ"):
            report.assert_csv_matches(self.csv_file(),[])

    def candidate(self):
        folder=self.root/"images/000001_000002/obj_000027/candidate_1";folder.mkdir(parents=True)
        (folder/"mask.png").write_bytes(b"test fixture mask bytes")
        pose=prediction()
        trace={"candidate":{"candidate_id":"candidate-1","source_file_sha256":"mask-source"},
               "status":"complete","GT_pose_access":False,
               "coarse_feature_score":.2,"refined_feature_score":.3,"geometry_score":.4,"joint_score":.024,
               "pose":{"candidate_id":"candidate-1","obj_id":27,"score":.024,"R":pose["R_m2c"],"t":pose["t_m"]},
               "runtime_s":.5,"artifacts":{"mask.png":h(folder/"mask.png")},
               "original_CAD_rows":100,"retained_CAD_rows":99,
               "original_sparse_rows":10,"retained_sparse_rows":9,"removed_sparse_indices":[3],
               "score_detail":{"cad_count":100,"feature_cad_count":99,"geometry_inlier_points":40,"sparse_observed_count":9}}
        dump(folder/"diagnostic.json",trace)
        ref={"candidate_id":"candidate-1","status":"complete","directory":"obj_000027/candidate_1","sha256":h(folder/"diagnostic.json")}
        return folder,trace,ref

    def test_candidate_trace_artifacts_and_product_checked(self):
        folder,trace,ref=self.candidate()
        self.assertEqual(self.audit.candidate(folder,ref)["joint_score"],.024)
        (folder/"mask.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError,"Changed saved evidence"):
            self.audit.candidate(folder,ref)

    def test_feature_subset_cannot_replace_geometry_denominator(self):
        folder,trace,ref=self.candidate();trace["score_detail"]["cad_count"]=99
        dump(folder/"diagnostic.json",trace);ref["sha256"]=h(folder/"diagnostic.json")
        with self.assertRaisesRegex(ValueError,"denominators"):
            self.audit.candidate(folder,ref)

    def test_missing_targets_are_not_silently_dropped(self):
        folder,trace,ref=self.candidate();image=folder.parent.parent
        ledger={"identity":{"run":"fixed"},"GT_pose_access":False,"runtime_s":1.25,
                "targets":[{"scene_id":1,"im_id":2,"obj_id":27,"inst_count":1,"status":"completed","predictions":1,"candidates":[ref]}],
                "predictions":[prediction()]}
        path=dump(image/"complete.json",ledger)
        targets=[{"scene_id":1,"im_id":2,"obj_id":27,"inst_count":1},
                 {"scene_id":1,"im_id":3,"obj_id":30,"inst_count":1}]
        with self.assertRaisesRegex(ValueError,"Missing cohort target"):
            report.audit_image_ledgers(self.audit,[(path,None)],targets,run_digest="fixed")
        targets=targets[:1]
        rows,poses=report.audit_image_ledgers(self.audit,[(path,None)],targets,run_digest="fixed")
        self.assertEqual(rows[0]["saved_predictions"],1);self.assertEqual(len(poses),1)

    def score_fixture(self):
        folder=self.root/"eval/freezev2cohort_tless-test"
        aggregate={name:.5 for name in report.METRICS}
        final=dump(folder/"scores_bop19.json",aggregate)
        for family,taus in (("vsd",10),("mssd",1),("mspd",1)):
            for tau in range(taus):
                sub=folder/f"error={family}_ntop=-1_tau={tau}"
                for threshold in range(10):
                    dump(sub/f"scores_th={threshold}.json",{"targets_count":7,"gt_count":99,"recall":.5})
        return final,{"score_file":str(final),"score_sha256":h(final),"metrics":aggregate}

    def test_all_121_scores_checked_and_gt_count_is_separate(self):
        final,receipt=self.score_fixture()
        result=self.audit.scores(receipt,7)
        self.assertEqual(result["threshold_score_files"],120)
        self.assertEqual(result["all_score_files"],121)
        self.assertEqual(result["targets_count_each"],7)

    def test_reduced_denominator_rejected(self):
        final,receipt=self.score_fixture()
        path=next(final.parent.rglob("scores_th=0.json"))
        dump(path,{"targets_count":6,"recall":.5})
        with self.assertRaisesRegex(ValueError,"denominator"):
            self.audit.scores(receipt,7)

    def test_missing_threshold_score_rejected(self):
        final,receipt=self.score_fixture();next(final.parent.rglob("scores_th=0.json")).unlink()
        with self.assertRaisesRegex(ValueError,"120 threshold"):
            self.audit.scores(receipt,7)

    def test_changed_aggregate_metric_rejected(self):
        final,receipt=self.score_fixture();receipt["metrics"][report.METRICS[0]]=.7
        with self.assertRaisesRegex(ValueError,"Receipt official metric"):
            self.audit.scores(receipt,7)

    def test_reporter_refuses_gt_pose_input(self):
        with self.assertRaisesRegex(ValueError,"forbidden"):
            self.audit.hash(self.root/"scene_gt.json")

    def test_onboarding_keeps_full_original_geometry(self):
        folder=self.root/"arm/onboarding/obj_000027";folder.mkdir(parents=True)
        (folder/"arrays.npz").write_bytes(b"fixture cache")
        dump(folder/"complete.json",{"sha256":h(folder/"arrays.npz")})
        data={"original_points":5,"retained_feature_points":4,"removed_original_indices":[3],
              "PCA_fit_rows":5,"ICP_geometry_rows":5,"geometry_score_denominator":5,"runtime_s":1}
        dump(folder/"onboarding.json",data)
        rows=[{"obj_id":27,"status":"completed"}]
        result=report.audit_onboarding(self.audit,self.root/"arm",{27},rows)
        self.assertEqual(result["27"]["retained_feature_points"],4)
        data["geometry_score_denominator"]=4;dump(folder/"onboarding.json",data)
        with self.assertRaisesRegex(ValueError,"denominator"):
            report.audit_onboarding(self.audit,self.root/"arm",{27},rows)


if __name__=="__main__":unittest.main()
