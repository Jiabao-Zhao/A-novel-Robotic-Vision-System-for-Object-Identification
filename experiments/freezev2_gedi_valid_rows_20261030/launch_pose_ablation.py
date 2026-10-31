"""Reuse only cached pure-Python pypng; keep inference numerical dependencies unchanged."""
from pathlib import Path
import hashlib,json,os,runpy,sys
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/freezev2_bop_20261029'
CONFIG=Path(__file__).with_name('config.json')
config=json.loads(CONFIG.read_text())
text=config['output'].replace('\\','/')
if os.name!='nt' and text[1:2]==':':text='/mnt/'+text[0].lower()+text[2:]
OUT=Path(text)
source=ROOT/'outputs/cache/bop_eval_deps/png.py'
if not source.is_file():raise FileNotFoundError('Existing cached pypng module is required; no installation attempted')
runtime=OUT/'runtime_deps';runtime.mkdir(parents=True,exist_ok=True)
link=runtime/'png.py'
if link.is_symlink():
    if link.resolve()!=source.resolve():raise ValueError('Wrong cached pypng link')
elif link.exists():raise ValueError('Refusing an unverified runtime module')
else:link.symlink_to(source)
record={'purpose':'Official BOP format validation only; no numeric environment replacement','cached_png_source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'link':str(link),'first_attempt':'Saved all repaired224 poses; final CSV check failed only because png import was absent. Resume uses unchanged inference source/config/checkpoints.'}
(OUT/'pypng_runtime.json').write_text(json.dumps(record,indent=2)+'\n')
os.environ['PYTHONPATH']=str(runtime)+os.pathsep+os.environ.get('PYTHONPATH','')
sys.path.insert(0,str(runtime))
runner=Path(__file__).with_name('run_pose_ablation.py')
sys.argv=[str(runner),*sys.argv[1:]]
runpy.run_path(str(runner),run_name='__main__')