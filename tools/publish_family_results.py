"""Export the local family ablation into a Git-trackable evidence directory.

Does not commit or push. Run from any directory; only Python stdlib is required.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--name',default='family_association_20260926')
    args=parser.parse_args()
    assert re.fullmatch(r'[a-z][a-z0-9_]+',args.name)
    workspace=args.workspace.resolve();repo=args.repo.resolve()
    source=workspace/'family-association-ablation'
    baseline=repo/'experiments/vlm_four_scene_20260925'
    dest=repo/'experiments'/args.name
    assert baseline.is_dir(),'Publish the matched baseline first.'
    assert not dest.exists(),'Use a new name; never overwrite published evidence.'
    assert (repo/'tools/verify_family_results.py').is_file()
    manifest=read(source/'manifest.json')
    validation=read(source/'validation.json')
    assert validation['calls']==1452 and validation['baseline_unchanged']
    for path,expected in manifest['immutable_inputs'].items():
        assert sha(workspace/path)==expected,path
    for path,expected in validation['raw_response_hashes'].items():
        assert sha(workspace/path)==expected,path
    mapping={r['source']:'../vlm_four_scene_20260925/'+r['file'] for r in read(baseline/'file_index.json')}
    def relocate(value):
        if isinstance(value,dict):
            return {k:relocate(v) for k,v in value.items()}
        if isinstance(value,list):
            return [relocate(v) for v in value]
        if isinstance(value,str):
            normalized=value.replace('\\','/')
            if normalized.startswith('family-association-ablation/'):
                return normalized.split('/',1)[1]
            return mapping.get(normalized,value)
        return value
    dest.mkdir(parents=True)
    index=[]
    for p in sorted(source.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix not in ('.json','.csv','.png','.pdf','.md','.py','.txt'):
            continue
        relative=p.relative_to(source)
        provenance=p.name in ('manifest.json','validation.json') or p.suffix=='.py'
        out=dest/('provenance/'+relative.as_posix() if provenance else relative)
        out.parent.mkdir(parents=True,exist_ok=True)
        raw='responses' in relative.parts
        if raw or provenance or p.suffix not in ('.json','.csv'):
            shutil.copyfile(p,out)
        elif p.suffix=='.json':
            save(out,relocate(read(p)))
        else:
            with p.open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
            with out.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(relocate(rows))
        index.append(dict(file=out.relative_to(dest).as_posix(),source=relative.as_posix(),
                          source_sha256=sha(p),published_sha256=sha(out),raw_bytes_preserved=raw))
    published={k:relocate(v) for k,v in manifest.items() if k not in ('immutable_inputs',)}
    published['source_manifest']='provenance/manifest.json'
    published['path_policy']='Canonical table paths are relative to this folder; baseline crops and records are in ../vlm_four_scene_20260925. Raw records/provenance preserve original local paths.'
    save(dest/'manifest.json',published)
    save(dest/'file_index.json',index)
    (dest/'.gitattributes').write_text('# Preserve evidence hashes across operating systems.\n* -text\n',encoding='utf-8',newline='\n')
    report=dest/'README.md'
    text=report.read_text(encoding='utf-8')
    header='''## GitHub / GPT Cloud Entry Point

This is the **new family-association prompt/description ablation**, not the baseline.
The [unchanged four-model baseline](../vlm_four_scene_20260925/README.md) remains separately versioned.

- [Paired raw scores](paired_scores.csv), [all baseline and revised scores](all_scores_comparison.json).
- [Exact revised prompts](selected_prompts.json), [family descriptions](descriptions.json).
- [Threshold comparison](threshold_comparison.png), [family retention at matched false-match rates](family_retention_vs_false_matches.png).
- [Per-family metrics](per_family_metrics.csv), [threshold sweeps](threshold_sweep.csv), [known failures](known_failure_pairs.csv).
- [Qwen4B heatmaps](qwen3.5-4b/four_scene_heatmaps.pdf), [Qwen9B heatmaps](qwen3.5-9b/four_scene_heatmaps.pdf), [Gemma heatmaps](gemma4-12b/four_scene_heatmaps.pdf).

All 1,452 new raw responses are in the three model folders. Canonical table paths
resolve from this directory. Identical input crops and the raw baseline records
are referenced in the sibling baseline folder, not duplicated. Raw request
metadata retains its original local paths; use the canonical score row for the
portable crop/response link. `provenance/` holds original local scripts and manifests,
not portable inference entry points. No cloud/API rerun was performed.

From the repository root, run:
`python tools/verify_family_results.py experiments/family_association_20260926`

This offline check needs no models, credentials, GPU, or third-party packages.
Read the limitations below before selecting a threshold; no setting was deployed.

'''
    title,rest=text.split('\n',1)
    report.write_text(title+'\n\n'+header.replace('family_association_20260926',args.name)+rest,encoding='utf-8',newline='\n')
    forbidden=re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}|\bgh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN .*PRIVATE KEY-----')
    hashes={}
    for p in dest.rglob('*'):
        if p.is_file():
            assert p.stat().st_size<45_000_000
            if p.suffix in ('.json','.csv','.md','.py','.txt'):
                assert not forbidden.search(p.read_text(encoding='utf-8-sig')),f'Potential credential: {p}'
            hashes[p.relative_to(dest).as_posix()]=sha(p)
    save(dest/'SHA256SUMS.json',hashes)
    print(json.dumps(dict(published=str(dest),files=len(hashes)+1,
                         raw_responses=len(list(dest.glob('*/responses/**/*.json'))),
                         bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())),indent=2))


if __name__=='__main__':
    main()
