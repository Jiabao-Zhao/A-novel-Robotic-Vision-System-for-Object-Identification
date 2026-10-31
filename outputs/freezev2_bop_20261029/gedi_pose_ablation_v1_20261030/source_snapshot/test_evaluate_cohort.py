"""CPU-only tests: denominator, immutable inputs and official command construction."""
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_cohort as ev

R = "1 0 0 0 1 0 0 0 1"


def pose(scene=1, image=2, obj=27, score="0.5", elapsed="1"):
    return [str(scene), str(image), str(obj), score, R, "0 0 500", elapsed]


def write_csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(ev.HEADER)
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


class CohortEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.targets = [dict(scene_id=1, im_id=2, obj_id=27, inst_count=2),
                        dict(scene_id=1, im_id=3, obj_id=30, inst_count=1)]
        self.cohort = self.root / "cohort.json"
        self.cohort.write_text(json.dumps(dict(dataset="tless", targets=self.targets)))
        self.source = self.root / "predictions.csv"
        write_csv(self.source, [pose(), pose(image=9, obj=1)])
        self.view = self.root / "existing_view"
        (self.view / "tless").mkdir(parents=True)
        (self.view / "tless/test_targets_bop19.json").write_text(json.dumps(self.targets))
        self.toolkit = self.root / "toolkit"
        (self.toolkit / "scripts").mkdir(parents=True)
        (self.toolkit / "scripts/eval_bop19_pose.py").write_text("# test fixture only\n")
        self.runtime = {"python": "/evaluation/venv/bin/python", "dataset_view": str(self.view),
                        "toolkit_root": str(self.toolkit), "toolkit_revision": ev.TOOLKIT_REVISION,
                        "renderer_type": "vispy", "protocol": {"visib_gt_min": -1},
                        "environment": {"BOP_PATH": "must-not-override-cohort", "PYOPENGL_PLATFORM": "egl"}}
        self.runtime_path = self.root / "runtime.json"
        self.runtime_path.write_text(json.dumps(self.runtime))
        self.output = self.root / "new_arm/evaluation"

    def prepare(self):
        with mock.patch.object(ev.subprocess, "check_output", return_value=ev.TOOLKIT_REVISION+"\n"):
            return ev.evaluate_saved(self.root, self.cohort, self.source, self.output,
                                     runtime_path=self.runtime_path, execute=False)

    def test_exact_official_inst_count_preserved(self):
        self.assertEqual(ev.validate_cohort({"dataset": "tless", "targets": self.targets}, self.targets), self.targets)

    def test_changed_instance_count_rejected(self):
        rows = [dict(self.targets[0], inst_count=1)]
        with self.assertRaisesRegex(ValueError, "exact official"):
            ev.validate_cohort(rows, self.targets)

    def test_unknown_target_rejected(self):
        with self.assertRaises(ValueError):
            ev.validate_cohort([dict(self.targets[0], obj_id=28)], self.targets)

    def test_duplicate_target_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ev.validate_cohort(self.targets+self.targets, self.targets)

    def test_wrong_dataset_and_boolean_ids_rejected(self):
        with self.assertRaises(ValueError):
            ev.validate_cohort({"dataset": "ycbv", "targets": self.targets}, self.targets)
        with self.assertRaises(ValueError):
            ev.validate_cohort([dict(self.targets[0], scene_id=True)], self.targets)

    def test_missing_predictions_stay_in_denominator(self):
        data, coverage = ev.filter_prediction_csv(self.source, self.targets)
        self.assertEqual(coverage["requested_instances"], 3)
        self.assertEqual(coverage["saved_predictions"], 1)
        self.assertEqual(coverage["missing_prediction_instances"], 2)
        self.assertEqual(coverage["target_rows_without_predictions"], 1)
        self.assertEqual(coverage["excluded_source_rows_outside_fixed_cohort"], 1)
        self.assertIn(b"1,2,27,0.5,", data)
        self.assertNotIn(b"1,9,1,", data)

    def test_header_only_csv_is_recorded_failure_not_empty_denominator(self):
        write_csv(self.source, [])
        data, coverage = ev.filter_prediction_csv(self.source, self.targets)
        self.assertEqual(coverage["requested_instances"], 3)
        self.assertEqual(coverage["missing_prediction_instances"], 3)
        self.assertEqual(data.decode().strip(), ",".join(ev.HEADER))

    def test_original_numeric_strings_are_preserved(self):
        write_csv(self.source, [pose(score="0.50000000000000001", elapsed="1.000000000")])
        data, _ = ev.filter_prediction_csv(self.source, self.targets)
        self.assertIn(b"0.50000000000000001", data)
        self.assertIn(b"1.000000000", data)

    def test_nonfinite_and_improper_rotation_rejected(self):
        write_csv(self.source, [pose(score="nan")])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            ev.filter_prediction_csv(self.source, self.targets)
        row = pose()
        row[4] = "-1 0 0 0 1 0 0 0 1"
        write_csv(self.source, [row])
        with self.assertRaisesRegex(ValueError, "SO\\(3\\)"):
            ev.filter_prediction_csv(self.source, self.targets)

    def test_excess_instance_predictions_rejected(self):
        write_csv(self.source, [pose(), pose(), pose()])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            ev.filter_prediction_csv(self.source, self.targets)

    def test_inconsistent_per_image_times_rejected(self):
        write_csv(self.source, [pose(elapsed="1"), pose(elapsed="2")])
        with self.assertRaisesRegex(ValueError, "timing"):
            ev.filter_prediction_csv(self.source, self.targets)

    def test_official_command_uses_cohort_and_cpu_evaluation(self):
        command, env = ev.build_evaluation_command(self.runtime, toolkit=self.toolkit,
            view=self.output/"bop_view", results=self.output/"results/freezev2cohort_tless-test.csv",
            evaluation=self.output/"official_evaluation")
        self.assertEqual(command[0], self.runtime["python"])
        self.assertEqual(command[command.index("--targets_filename")+1], ev.TARGETS_NAME)
        self.assertEqual(command[command.index("--num_workers")+1], "1")
        self.assertNotIn("--use_gpu", command)
        self.assertEqual(env["BOP_PATH"], str(self.output/"bop_view"))
        self.assertEqual(env["OMP_NUM_THREADS"], "4")
        self.assertEqual(env["PYOPENGL_PLATFORM"], "egl")

    def test_unpinned_toolkit_or_visibility_protocol_rejected(self):
        kwargs = dict(toolkit=self.toolkit, view=self.view, results=self.source, evaluation=self.output)
        with self.assertRaisesRegex(ValueError, "Unpinned"):
            ev.build_evaluation_command(dict(self.runtime, toolkit_revision="other"), **kwargs)
        with self.assertRaisesRegex(ValueError, "visib_gt_min"):
            ev.build_evaluation_command(dict(self.runtime, protocol={"visib_gt_min": 0.1}), **kwargs)
        with self.assertRaises(ValueError):
            ev.build_evaluation_command(self.runtime, targets_filename="../targets.json", **kwargs)

    def test_prepare_only_never_launches_evaluator_or_reads_gt(self):
        original_targets = (self.view/"tless/test_targets_bop19.json").read_bytes()
        with mock.patch.object(ev, "_pinned_helpers", side_effect=AssertionError("must not launch")):
            result = self.prepare()
        self.assertEqual(result["status"], "prepared_not_evaluated")
        self.assertIsNone(result["metrics"])
        self.assertEqual(result["requested_instances"], 3)
        self.assertEqual(json.loads(Path(result["targets_path"]).read_text()), self.targets)
        self.assertEqual((self.view/"tless/test_targets_bop19.json").read_bytes(), original_targets)
        self.assertFalse((self.output/"evaluation_receipt.json").exists())

    def test_changed_csv_cannot_reuse_output(self):
        self.prepare()
        write_csv(self.source, [pose(score="0.6")])
        with self.assertRaisesRegex(ValueError, "Incompatible existing"):
            self.prepare()

    def test_frozen_baseline_output_is_protected(self):
        self.output = self.root/"outputs/freezev2_bop_20261029/classic_core_base_v2/new"
        with self.assertRaisesRegex(ValueError, "frozen baseline"):
            self.prepare()

    def test_changed_official_target_pin_is_rejected(self):
        self.cohort.write_text(json.dumps(dict(dataset="tless", targets=self.targets,
            official_targets={"sha256": "0"*64})))
        with self.assertRaisesRegex(ValueError, "target source changed"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
