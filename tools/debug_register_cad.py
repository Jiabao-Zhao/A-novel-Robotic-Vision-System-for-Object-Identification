import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CADPointCloudRegistration import CADPointCloudRegistration


def load_plane_model(localization_path):
    if localization_path is None:
        return None
    payload = json.loads(Path(localization_path).read_text(encoding="utf-8"))
    return payload.get("plane_model")


def main():
    parser = argparse.ArgumentParser(description="Debug CAD-to-observed point-cloud registration.")
    parser.add_argument("--cad", required=True, help="CAD mesh or point-cloud path.")
    parser.add_argument("--observed", required=True, help="Observed object cluster PLY path.")
    parser.add_argument("--localization", help="Optional localization JSON containing plane_model.")
    parser.add_argument("--output", default="output/debug_alignment", help="Output directory.")
    args = parser.parse_args()

    result = CADPointCloudRegistration().run(
        cad_path=args.cad,
        observed_cloud_path=args.observed,
        plane_model=load_plane_model(args.localization),
        output_dir=args.output,
    )

    print(f"Selected candidate: {result['selected_candidate_id']} "
          f"{result['selected_candidate_name']} ({result['selected_candidate_source']})")
    print(f"Final score: {result['final_score']:.6f}")
    print(f"Score breakdown: {json.dumps(result['final_score_breakdown'], indent=2)}")
    print(f"ICP fitness: {result['icp_fitness']}")
    print(f"ICP RMSE: {result['icp_inlier_rmse']}")
    print(f"CAD extent: {result['cad_extent_m']}")
    print(f"Observed extent: {result['observed_extent_m']}")
    print(f"Warnings: {result['warnings']}")
    print(f"Aligned CAD cloud: {result['aligned_cad_cloud_path']}")
    print(f"Augmented cloud: {result['augmented_cloud_path']}")
    print(f"Result JSON: {result['result_path']}")
    print(f"Candidates JSON: {result['candidates_path']}")


if __name__ == "__main__":
    main()
