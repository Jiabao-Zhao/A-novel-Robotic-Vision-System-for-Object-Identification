"""Five independent assembly jobs in the reduced twelve-object wrist scene."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from scripts.run_wrist_cluster import (MODEL, CAMERA_DESCRIPTION, THRESHOLD, prepare_observation,
                                      identity_audit, run_target, save_json)
from simulation.wrist_cluster import (prepare_catalog, random_placements, make_environment,
    CAMERA, DESCRIPTIONS, WORKSPACE_MIN, WORKSPACE_MAX, BOARD_EXCLUSION_XY)
from simulation.nist_assembly import (assembly_jobs, preserve_gear_bore, peg_seating_metrics,
                                      configure_assembly_contacts)
from simulation.libero_sensor import LiberoRGBDSensor
from simulation.libero_io import save_libero_observation
from simulation.nist_peg_task import localize_parts
from vlm_module import annotate_candidate_boxes


ROOT = Path('outputs/simulation/nist_assembly')
SEED = 20260920
EXCLUDED = ('Gear_Medium', 'M12_Hex_Nut')


def pin_wording_comparison(root, execute=False, *, square_pin_only=False):
    """Three repeats per noun on one frozen image; old system prompt is preserved."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import random
    from openai import OpenAI
    from simulation.libero_joint_association import request_arguments, joint_inferences
    from vlm_module import candidate_choice_map, load_localized_objects
    root = Path(root)
    output = root / ('square_pin_comparison' if square_pin_only else 'pin_wording_comparison')
    output.mkdir(exist_ok=True)
    visual = root / 'wrist_boxed.png'
    mapping = candidate_choice_map(load_localized_objects(root / 'point_cloud/point_cloud_localization.json'))
    truth = {row['object_id']: row['simulator_instance'] for row in json.loads((root / 'identity_audit.json').read_text())}
    jobs = []
    wordings = (('KET16_Square_16mm', ('square pin',)),) if square_pin_only else (
        ('RGOCG16-50_16mm', ('round peg', 'round pin')),
        ('KET16_Square_16mm', ('white square peg', 'white square pin')))
    for target, names in wordings:
        previous = json.loads((root / target / 'vlm_request.json').read_text())
        for name in names:
            for repeat in range(1, 4):
                instruction = f'Pick up the {name} and hold it above the workspace.'
                args = request_arguments(visual.read_bytes(), instruction, MODEL, camera_description=CAMERA_DESCRIPTION)
                args['messages'][0]['content'] = previous['system_prompt'].replace(previous['instruction'], instruction)
                case = output / (name.replace(' ', '_') + '_' + str(repeat))
                case.mkdir(exist_ok=True)
                record = {'target': target, 'name': name, 'repeat': repeat, 'instruction': instruction,
                          'system_prompt': args['messages'][0]['content'], 'image_path': str(visual),
                          'image_sha256': hashlib.sha256(visual.read_bytes()).hexdigest(),
                          'settings': {key: value for key, value in args.items() if key != 'messages'}}
                if (case / 'request.json').exists():
                    assert json.loads((case / 'request.json').read_text()) == record
                else:
                    save_json(case / 'request.json', record)
                jobs.append((case, record, args))
    save_json(output / 'manifest.json', {'requests': len(jobs), 'actual_scenes': 1, 'repeats_per_name': 3,
        'control': 'Only target noun changes. Identical image, boxes, saved system prompt, model, and sampling settings.',
        'purpose': 'Naming diagnostic; not independent robot trials or a calibrated accuracy estimate.'})
    if not execute:
        print(f'PREPARED {len(jobs)} naming requests; no API calls: {output}', flush=True)
        return output
    def query(job):
        case, record, args = job
        if (case / 'result.json').exists():
            return json.loads((case / 'result.json').read_text())
        if (case / 'provider_response.json').exists():
            raw = json.loads((case / 'provider_response.json').read_text())
        else:
            with OpenAI(timeout=60.) as client:
                assert str(client.base_url) == 'https://api.openai.com/v1/'
                raw = client.chat.completions.create(**args).model_dump(mode='json')
            save_json(case / 'provider_response.json', raw)
        inferences, parsed = joint_inferences(raw, record['instruction'], mapping, [record['name']])
        inference = inferences[0]
        valid = inference['diagnostics']['joint_format_valid'] and inference['diagnostics']['required_names_complete']
        selected = truth.get(inference['vlm_object_id']) if valid else None
        result = {key: record[key] for key in ('target', 'name', 'repeat')}
        result.update(valid=valid, selected=selected, correct=valid and selected == record['target'],
                      raw_score=inference['association_score'], output=parsed['generated_output_text'],
                      provider_model=raw.get('model'), system_fingerprint=raw.get('system_fingerprint'))
        save_json(case / 'result.json', result)
        return result
    random.Random(20260921).shuffle(jobs)
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(query, job) for job in jobs]):
            results.append(future.result())
            save_json(output / 'results.json', results)
            print(f'NAMING {len(results)}/{len(jobs)}: {results[-1]}', flush=True)
    return output


def prepare_requests(root, jobs):
    from simulation.libero_joint_association import request_arguments
    visual = root / 'wrist_boxed.png'
    requests = []
    for job in jobs:
        instruction = f"Pick up the {DESCRIPTIONS[job['target']]} and hold it above the workspace."
        args = request_arguments(visual.read_bytes(), instruction, MODEL, camera_description=CAMERA_DESCRIPTION)
        requests.append({'target': job['target'], 'assembly_instruction': job['instruction'],
            'association_instruction': instruction, 'system_prompt': args['messages'][0]['content'],
            'settings': {key: value for key, value in args.items() if key != 'messages'},
            'image_path': str(visual), 'image_sha256': hashlib.sha256(visual.read_bytes()).hexdigest(),
            'planner_payload': 'Assembly instruction, resolved source identity, estimated CAD/grasp poses, known fixture design, robot proprioception. No evaluation truth.'})
    save_json(root / 'prepared_requests.json', requests)


def run_jobs(environment, root, manifest, raw):
    observation = LiberoRGBDSensor(environment, CAMERA).capture(raw)
    paths = {'localization': root / 'point_cloud/point_cloud_localization.json'}
    localization = json.loads(paths['localization'].read_text())
    results = []
    for job in manifest['jobs']:
        case = root / job['target'] / 'evaluation.json'
        if case.exists():
            result = json.loads(case.read_text())
        else:
            result = run_target(environment, np.load(root / 'initial_state.npy'), job['target'],
                manifest['catalog'], root, localization, observation, root / 'wrist_boxed.png', paths,
                frozen_gripper_action=np.array(manifest['initial_gripper_action']),
                seed=manifest['seed'], assembly=job)
        results.append(result)
        save_json(root / 'results.json', results)
        print(f'JOB {len(results)}/5 {job["target"]}: success={result["success"]}; '
              f'{result.get("status", result.get("failure_attribution"))}; {result.get("error")}', flush=True)


def run_prepared(root):
    """Execute the reviewed scene without regenerating its poses or API inputs."""
    from scripts.diagnose_wrist_pickups import restore
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest['source_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    os.environ['OPENAI_VLM_MODEL'] = MODEL
    os.environ['OPENAI_LLM_MODEL'] = MODEL
    catalog = manifest['catalog']
    with make_environment(catalog, manifest['placements'],
                          additional_scene=lambda model: preserve_gear_bore(model, catalog)) as environment:
        if manifest.get('contact_settings'):
            configure_assembly_contacts(environment)
        raw = restore(environment, manifest, np.load(root / 'initial_state.npy'))
        run_jobs(environment, root, manifest, raw)
    print(f'COMPLETE {root}', flush=True)


def collision_preflight(environment, jobs):
    """Static geometry diagnostics only; restore state before any robot attempt."""
    sim = environment.sim
    frozen = sim.get_state().flatten().copy()
    records = []
    try:
        for job in jobs:
            record = {'target': job['target'], 'ready': job['ready'], 'reason': job['readiness_reason']}
            if job['ready']:
                body = sim.model.body_name2id(f"nist_part_{job['target']}")
                joint = int(sim.model.body_jntadr[body])
                start = int(sim.model.jnt_qposadr[joint])
                position = np.array(job['center_world_m'])
                position[2] += job['cad_extent_m'][2] / 2 - job['commanded_insertion_depth_m']
                probes = []
                for offset in (0., .0005, .02):
                    probe = position + [offset, 0, 0]
                    sim.data.qpos[start:start + 7] = [*probe, 1, 0, 0, 0]
                    sim.forward()
                    distances = [float(contact.dist) for contact in sim.data.contact[:sim.data.ncon]
                        if body in (int(sim.model.geom_bodyid[contact.geom1]), int(sim.model.geom_bodyid[contact.geom2]))]
                    metrics = peg_seating_metrics(probe, np.eye(3), job)
                    probes.append({'offset_x_m': offset, 'minimum_contact_distance_m': min(distances, default=0.),
                                   'geometry_metrics': metrics})
                record['probes'] = probes
                record['ready'] = (probes[0]['minimum_contact_distance_m'] >= -.00002
                    and probes[0]['geometry_metrics']['inserted']
                    and all(probe['minimum_contact_distance_m'] < -.00005
                            and not probe['geometry_metrics']['inserted'] for probe in probes[1:]))
                if not record['ready']:
                    record['reason'] = 'Nominal fit or offset-collision control did not pass.'
                    job.update(ready=False, readiness_reason=record['reason'])
                sim.set_state_from_flattened(frozen)
                sim.forward()
            if job['target'] == 'Gear_Large':
                import mujoco
                body = sim.model.body_name2id('nist_part_Gear_Large')
                position = sim.data.body_xpos[body].copy()
                hits = []
                # Rays otherwise include inactive mass-only geometry, unlike contacts.
                groups = sim.model.geom_group.copy()
                try:
                    inactive = (sim.model.geom_contype == 0) & (sim.model.geom_conaffinity == 0)
                    sim.model.geom_group[inactive] = 5
                    for offset in (0., .02):
                        hit = np.array([-1], dtype=np.int32)
                        distance = mujoco.mj_ray(sim.model._model, sim.data._data,
                            position + [offset, 0, .08], np.array([0., 0., -1.]),
                            np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1, hit)
                        hits.append({'offset_x_m': offset, 'distance_m': float(distance),
                                     'hit_geom': sim.model.geom_id2name(int(hit[0])) if hit[0] >= 0 else None})
                finally:
                    sim.model.geom_group[:] = groups
                record['bore_collision_ray_checks'] = hits
                assert not (hits[0]['hit_geom'] or '').startswith('assembly_gear_')
                assert (hits[1]['hit_geom'] or '').startswith('assembly_gear_')
            records.append(record)
    finally:
        sim.set_state_from_flattened(frozen)
        sim.forward()
    np.testing.assert_array_equal(sim.get_state().flatten(), frozen)
    return records


def main(execute=True):
    os.environ['OPENAI_VLM_MODEL'] = MODEL
    os.environ['OPENAI_LLM_MODEL'] = MODEL
    root = ROOT / ('pilot_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    root.mkdir(parents=True, exist_ok=False)
    catalog = {name: record for name, record in prepare_catalog().items() if name not in EXCLUDED}
    assert len(catalog) == 12
    placements = random_placements(catalog, SEED)
    jobs = assembly_jobs(catalog)
    snapshot = root / 'source_snapshot'
    sources = ['scripts/run_nist_assembly.py', 'scripts/run_wrist_cluster.py', 'simulation/nist_assembly.py',
               'simulation/nist_task_board_1.py', 'simulation/wrist_cluster.py', 'simulation/libero_control.py',
               'simulation/libero_planning.py', 'simulation/libero_joint_association.py', 'vlm_module.py',
               'CADPointCloudRegistration.py', 'point_cloud_localization.py']
    hashes = {}
    for name in sources:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        hashes[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    with make_environment(catalog, placements,
                          additional_scene=lambda model: preserve_gear_bore(model, catalog)) as environment:
        contact_settings = configure_assembly_contacts(environment)
        raw = prepare_observation(environment, SEED)
        preflight = collision_preflight(environment, jobs)
        save_json(root / 'preflight.json', preflight)
        for record in preflight:
            print('PREFLIGHT', record['target'], record['ready'], record['reason'], flush=True)
        frozen = np.ascontiguousarray(environment.sim.get_state().flatten())
        frozen_gripper = environment.robots[0].gripper.current_action.copy()
        np.save(root / 'initial_state.npy', frozen)
        observation = LiberoRGBDSensor(environment, CAMERA).capture(raw)
        capture = save_libero_observation(observation, environment, root / 'capture')
        save_json(root / 'manifest.json', {'seed': SEED, 'objects': list(catalog), 'object_count': len(catalog),
            'excluded': EXCLUDED, 'placements': placements, 'catalog': catalog, 'jobs': jobs,
            'source_hashes': hashes, 'model': MODEL, 'threshold': THRESHOLD,
            'contact_settings': contact_settings, 'scoring_revision': 2,
            'camera_inputs': [CAMERA], 'camera_description': CAMERA_DESCRIPTION,
            'fixture_pose_source': 'known fixed workcell design, not board perception',
            'initial_gripper_action': frozen_gripper.tolist(),
            'initial_state_sha256': hashlib.sha256(frozen.tobytes()).hexdigest(),
            'protocol': 'Five independent jobs, same frozen 12-object state. Human correction only after confidence deferral. Readiness blocks excluded from framework failure counts.',
            'success': 'Requested peg inserted >=5mm inside CAD opening, released and hand withdrawn, stable for final 20 steps. No wrong object lifted.',
            'gear_collision': 'Conservative 64-sector annular rim and hub; CAD minimum bore radius, original mass, no invented slip-fit clearance.'})
        localization, paths = localize_parts(observation, capture['rgb'], root,
            workspace_min=WORKSPACE_MIN, workspace_max=WORKSPACE_MAX,
            cluster_in_table_plane=True, exclude_xy_bounds=BOARD_EXCLUSION_XY)
        visual = annotate_candidate_boxes(capture['rgb'], paths['localization'], root / 'wrist_boxed.png')
        save_json(root / 'identity_audit.json', identity_audit(environment, localization, observation))
        prepare_requests(root, jobs)
        print(f'SCENE {root}: {len(localization["objects"])} localized / 12 objects', flush=True)
        if execute:
            run_jobs(environment, root, json.loads((root / 'manifest.json').read_text()), raw)
    print(f'COMPLETE {root}', flush=True)
    return root


if __name__ == '__main__':
    main()
