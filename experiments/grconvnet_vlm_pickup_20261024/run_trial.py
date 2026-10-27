"""Three matched pickup conditions; pretrained inference, no assembly views."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import torch

from experiments.runtime_antipodal_assembly_20261023.robot_trial import RobotTrial
from experiments.runtime_antipodal_assembly_20261023.grasp_planning import Candidate, sample_contacts, table_filter
from experiments.runtime_antipodal_assembly_20261023.run_experiment import PARTS, build_plan, check_plan
from experiments.runtime_antipodal_assembly_20261023.clarification_20261023.run_8mm_runtime_followup import restore_other_part_visibility, execute_once
from experiments.runtime_antipodal_assembly_20261023.smaller_search_20261023.search import frames_with_full_finger_depths, small_shortlist
from .grconvnet_adapter import GrConvNetAdapter
from .capture import capture_scene, rectangle_geometry, project_world
from .vlm_client import bridge, rank, digest, OPTIONS, MODEL, EXPECTED_DIGEST


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "outputs/grconvnet_vlm_pickup_20261024"
SOURCE = ROOT / "outputs/runtime_antipodal_assembly_20261023/input_scene/scene.xml"
MESHES = ROOT / "outputs/simulation/nist_task_board_1/assets/meshes"
ARMS = ("geometric", "grconvnet_score", "grconvnet_vlm")
DEPTHS = (.004, 0., -.004, -.008, .006, .008)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def learned_candidates(prediction, observation, trial):
    candidates, records, rejected = [], [], []
    for proposal in prediction["proposals"]:
        try:
            geometry = rectangle_geometry(proposal, observation["depth_m"], observation["target_mask"], observation["calibration"])
            width = geometry["predicted_width_m"]
            if not .002 < width < .075:
                raise ValueError(f"Predicted jaw span {width*1000:.3f} mm is outside the 2–75 mm Panda search range")
            axis = np.asarray(geometry["jaw_axis_world_unit"])
            axis[2] = 0.
            axis /= np.linalg.norm(axis)
            down = np.array([0., 0., -1.])
            center = np.asarray(geometry["center_world_m"])
            rotations = [np.column_stack((axis*s, np.cross(down, axis*s), down)) for s in (1., -1.)]
            rotation = min(rotations, key=lambda r: np.linalg.norm(Rotation.from_matrix(r @ trial.pose()[:3, :3].T).as_rotvec()))
            variants = []
            for depth in DEPTHS:
                pose = np.eye(4)
                pose[:3, :3] = rotation
                pose[:3, 3] = center-rotation[:, 2]*depth
                endpoints = np.stack([center-axis*width/2, center+axis*width/2])
                variants.append(Candidate(pose, endpoints, width, min(.08, width+.012),
                                          -proposal["quality"], 0., depth))
            clear = table_filter(variants, trial.m, trial.d)
            if not clear:
                raise ValueError("None of the six finger depths clears the table")
            chosen = clear[0]
            candidates.append(chosen)
            records.append({"network_candidate_id": proposal["network_candidate_id"],
                            "proposal": proposal, "depth_geometry": geometry,
                            "chosen_depth_policy": "First table-clear depth in fixed common six-depth order; no later depth fallback",
                            "predicted_endpoints_are_actual_surface_contacts": False,
                            **chosen.record()})
        except ValueError as error:
            rejected.append({"network_candidate_id": proposal["network_candidate_id"], "reason": str(error), "proposal": proposal})
    return candidates, records, rejected


def rank_sheet(observation, records, part_index, folder):
    """OWG-style native-image overlays; display IDs hide original score order."""
    rgb = Image.fromarray(observation["rgb"])
    mask = observation["target_mask"]
    rows, cols = np.nonzero(mask)
    padding = 50
    box = (max(0,int(cols.min())-padding), max(0,int(rows.min())-padding),
           min(rgb.width,int(cols.max())+padding+1), min(rgb.height,int(rows.max())+padding+1))
    edge = mask & (~np.roll(mask,1,0) | ~np.roll(mask,-1,0) | ~np.roll(mask,1,1) | ~np.roll(mask,-1,1))
    marked = np.asarray(rgb).copy()
    marked[edge] = [0,220,30]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",18)
    order = np.random.default_rng(20261024+part_index).permutation(len(records)).tolist()
    mapping, tiles = {}, []
    for display, index in enumerate(order,1):
        record = records[index]
        image = Image.fromarray(marked.copy())
        draw = ImageDraw.Draw(image)
        # Project this exact chosen gripper's commanded jaw opening at the
        # observed surface, retaining the same XY closing axis as its pose.
        center = np.asarray(record["depth_geometry"]["surface_center_world_m"])
        axis = np.asarray(record["world_T_grip"])[:3,0]
        across = np.cross([0.,0.,-1.],axis)
        endpoints = np.stack([center-axis*record["opening_m"]/2,center+axis*record["opening_m"]/2])
        rc,_ = project_world(endpoints,observation["calibration"])
        draw.line([(float(c),float(r)) for r,c in rc], fill="#ef3434",width=3)
        for endpoint in endpoints:
            finger_rc,_ = project_world(np.stack([endpoint-across*.007, endpoint+across*.007]), observation["calibration"])
            draw.line([(float(c),float(r)) for r,c in finger_rc], fill="#1672e5",width=5)
        crop = image.crop(box)
        crop.thumbnail((400,330))
        tile = Image.new("RGB",(420,410),"white")
        tile.paste(crop,((420-crop.width)//2,55+(330-crop.height)//2))
        text = ImageDraw.Draw(tile)
        text.text((12,10),f"Candidate {display}",font=font,fill="#172b40")
        text.text((12,381),f"Tool height {record['world_T_grip'][2][3]*1000:.1f} mm",font=font,fill="#526171")
        tile.save(folder / f"candidate_{display}.png")
        tiles.append(tile)
        mapping[display] = index
    sheet = Image.new("RGB",(840,60+410*((len(tiles)+1)//2)),"#eef2f6")
    ImageDraw.Draw(sheet).text((14,15),"Pickup candidates only — no prior outcomes",font=font,fill="#172b40")
    for index,tile in enumerate(tiles):
        sheet.paste(tile,((index%2)*420,60+(index//2)*410))
    sheet.save(folder / "rank_input.png")
    save(folder / "display_id_map.json", {str(k):{"pool_index":v,"network_candidate_id":records[v]["network_candidate_id"],
         "world_T_grip":records[v]["world_T_grip"],"pad_depth_m":records[v]["pad_depth_m"]} for k,v in mapping.items()})
    return folder / "rank_input.png", mapping


def run_part(part,arm,index,output,model):
    folder = output / arm / part
    started = time.perf_counter()
    trial = RobotTrial(SOURCE,folder,part)
    result = {"part":part,"arm":arm,"physical_attempts":0,"assembly_attempted":False,
              "stage_seconds":{},"candidates":[],"shortlisted_grasps":[],"training_performed":False,
              "source_scene_sha256":digest(SOURCE),"shared_pool_cost_reused":False}
    stages=result["stage_seconds"]
    decision_started=None
    try:
        result["other_parts_visible_but_collision_off"] = restore_other_part_visibility(trial)
        initial=trial.target_pose()  # Supplied geometric input / learned evaluation and target collision audit only.
        result["world_T_cad_input"] = initial.tolist()
        result["setup_seconds"] = time.perf_counter()-started
        decision_started = time.perf_counter()
        stage=time.perf_counter()
        observation=capture_scene(trial,folder / "capture")
        stages["capture"] = time.perf_counter()-stage
        if arm == "geometric":
            stage=time.perf_counter()
            _,_,_,pairs,stats=sample_contacts(MESHES/(part+"_m.stl"),initial,512)
            stages["contact_sampling"] = time.perf_counter()-stage
            result.update(stats)
            stage=time.perf_counter()
            generated=frames_with_full_finger_depths(pairs,trial.pose()[:3,:3],initial[:3,3],24)
            clear=table_filter(generated,trial.m,trial.d)
            pool=small_shortlist(clear,12)
            records=[{"candidate_id":j,**candidate.record()} for j,candidate in enumerate(pool)]
            result.update(generated_gripper_poses=len(generated),table_clear_candidates=len(clear))
            stages["frame_generation"] = time.perf_counter()-stage
        else:
            prediction=model.predict(observation["rgb"],observation["depth_m"],observation["target_mask"],4)
            stages.update(preprocessing=prediction["timings"]["preprocess_seconds"],
                          inference=prediction["timings"]["network_seconds"],postprocessing=prediction["timings"]["postprocess_seconds"])
            stage=time.perf_counter()
            pool,records,rejected=learned_candidates(prediction,observation,trial)
            stages["frame_generation"] = time.perf_counter()-stage
            result.update(raw_network_proposals=prediction["proposals"],proposal_mapping_rejections=rejected,
                          crop=prediction["crop"],model_sha256=prediction["model_sha256"],
                          object_pose_used_for_grasp_generation=False,
                          maximum_depth_combinations=len(prediction["proposals"])*len(DEPTHS),
                          table_clear_candidates=len(pool))
            stage=time.perf_counter()
            np.savez_compressed(folder/"network_evidence.npz",**prediction["maps"],input_tensor=prediction["input_tensor"],
                                preprocessed_depth_scaled=prediction["preprocessed_depth_scaled"])
            Image.fromarray(prediction["preprocessed_rgb"]).save(folder/"network_rgb.png")
            stages["evidence_writes"] = time.perf_counter()-stage
        result["candidates"]=records
        result["shortlisted_count"]=len(pool)
        ordering=list(range(len(pool)))
        if arm == "grconvnet_vlm":
            if pool:
                stage=time.perf_counter()
                sheet,mapping=rank_sheet(observation,records,index,folder)
                stages["rendering"] = time.perf_counter()-stage
                stage=time.perf_counter()
                ranking=rank(sheet,list(mapping),folder/"vlm")
                stages["vlm"] = time.perf_counter()-stage
                result["vlm_status"] = ranking["status"]
                if ranking["status"] != "ok":
                    result["status"]="vlm_output_unavailable"
                    ordering=[]
                else:
                    result["vlm_ranked_display_ids"]=ranking["ranked_ids"]
                    ordering=[mapping[x] for x in ranking["ranked_ids"]]
            else:
                result["vlm_status"]="skipped_no_candidates"
        result["candidate_check_order"]=ordering
        selected=None
        rejected_paths=[]
        stage=time.perf_counter()
        for pool_index in ordering:
            candidate=pool[pool_index]
            plan=build_plan(trial,candidate,initial,initial)
            passed,checks=check_plan(trial,candidate,plan,pickup_only=True)
            if passed:
                selected=candidate
                result.update(selected_pool_index=pool_index,selected_shortlist_index=pool_index,
                              selected_grasp=candidate.record(),path_checks=checks,
                              plan={key:value.tolist() for key,value in plan.items()})
                break
            rejected_paths.append({"pool_index":pool_index,"checks":checks})
        stages["path_checks"] = time.perf_counter()-stage
        result["planning_rejections"]=rejected_paths
        result["path_candidates_checked"]=len(rejected_paths)+int(selected is not None)
        result["decision_seconds"]=time.perf_counter()-decision_started
        result["planning_seconds_including_setup"]=time.perf_counter()-started
        result["shortlisted_grasps"]=records
        if selected is None:
            result.setdefault("status","no_pickup_path_found_in_candidates")
        else:
            result["physical_attempts"]=1
            stage=time.perf_counter()
            result.update(execute_once(trial,selected,plan,initial))
            result["execution_and_snapshots_seconds"]=time.perf_counter()-stage
    except Exception as error:
        result.update(status="motion_aborted" if len(trial.states)>1 else "experiment_error",
                      error=str(error),traceback=traceback.format_exc())
    finally:
        result.setdefault("setup_seconds",time.perf_counter()-started)
        if decision_started is not None:
            result.setdefault("decision_seconds",time.perf_counter()-decision_started)
            result.setdefault("planning_seconds_including_setup",time.perf_counter()-started)
        result["last_executed_phase"]=trial.states[-1]["phase"]
        result["maximum_observed_lift_m"]=float(max(s["target_pose_audit"][2][3] for s in trial.states)
            -result.get("world_T_cad_input",trial.target_pose())[2][3])
        result["controller_audit"]=trial.save()
        trial.close()
        result["total_seconds"]=time.perf_counter()-started
        save(folder/"result.json",result)
    return result


def main():
    torch.set_num_threads(4)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output=BASE/"runs"/stamp
    output.mkdir(parents=True,exist_ok=False)
    print(f"OUTPUT={output}",flush=True)
    model=GrConvNetAdapter(device="cpu")
    inventory=bridge({"action":"inventory"},output/"vlm_inventory")
    sources={str(p.relative_to(HERE)):digest(p) for p in HERE.rglob("*.py") if "vendor" not in p.parts}
    manifest={"parts":PARTS,"arms":ARMS,"created_utc":stamp,"source_scene_sha256":digest(SOURCE),
              "finger_depths_m":DEPTHS,"surface_rays_geometric":512,"contact_pair_budget_geometric":24,
              "geometric_shortlist_limit":12,"learned_network_candidate_limit":4,
              "model_setup_seconds_one_time":model.model_setup_seconds,"model_provenance":model.provenance,
              "vlm_inventory":inventory,"vlm_model":MODEL,"vlm_digest":EXPECTED_DIGEST,"vlm_options":OPTIONS,
              "prompt_sha256":digest(HERE/"grasp_rank_prompt.txt"),"source_hashes":sources,
              "input_conditions":{"same_old_14part_starting_scene":True,"perfect_target_mask_from_renderer":True,
                 "perfect_CAD_alignment_given_to_geometric_only":True,"learned_generation_uses_CAD_pose":False,
                 "overhead_camera_calibration_from_simulator":True,"final_assembly_views_or_goals":False},
              "trial":{"lift_command_m":.18,"hold_after_lift_s":1.,"min_lift_m":.12,
                 "bilateral_contact_required":True,"robot_overlap_limit_m":.001,"physical_attempt_limit_per_part_arm":1},
              "timing":"All arms capture freshRGBD; learned arms infer freshCPU proposals. Rotating armorder, one observation each. Decision time includes capture/evidence/render/localVLM/path checks. Setup and physicalexecution/save costs also recorded. One-time model setup reported separately. No tuning after outcomes."}
    save(output/"manifest.json",manifest)
    results=[]
    for index,part in enumerate(PARTS):
        order=ARMS[index%3:]+ARMS[:index%3]
        for arm in order:
            print(f"START {arm} {part}",flush=True)
            result=run_part(part,arm,index,output,model)
            results.append(result)
            save(output/"results.json",results)
            print(json.dumps({k:result.get(k) for k in ("arm","part","status","shortlisted_count","vlm_status","path_candidates_checked","decision_seconds","total_seconds","lift_m","error")}),flush=True)
    print(f"COMPLETE={output}",flush=True)


if __name__ == "__main__":
    main()
