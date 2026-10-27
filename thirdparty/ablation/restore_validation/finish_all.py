"""Queue disjoint remaining preparation, then run all-ten-video evaluation."""
import json,subprocess,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
PYTHON=ROOT/'.env/hawor/bin/python'
def state(name):
 p=HERE/'runs'/name/'status.json'
 return json.loads(p.read_text()) if p.exists() else {'status':'waiting'}
def wait(name):
 while True:
  d=state(name)
  if d['status'] in ('complete','complete_diagnostic_no_formal_row','prepared'): return
  if d['status']=='failed': raise RuntimeError(f'{name}: {d.get("error")}')
  time.sleep(20)
def run(name,videos=None):
 command=[str(PYTHON),str(HERE/'after_export.py'),'--physical_gpu','1','--run_dir',str(HERE/'runs'/name)]
 if videos:
  command+=['--prepare_only']
  for seq in videos: command+=['--video',seq]
 else:
  command+=['--inference_gpus','1,2,3']
 with (HERE/f'{name}.log').open('x') as log:
  result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(f'{name} exited {result.returncode}')
def write(status,**kwargs):
 (HERE/'workflow_status.json').write_text(json.dumps({'status':status,'pid':os.getpid(),**kwargs},indent=2)+'\n')
try:
 write('waiting_for_pilot')
 wait('pilot_first_video_retry1')
 write('preparing_remaining_gpu1')
 if (HERE/'runs/prepare_remaining_gpu1/status.json').exists():
  wait('prepare_remaining_gpu1')
 else:
  run('prepare_remaining_gpu1',['P0012_130a66e1','P0014_24cb3bf0','P0010_1c9fe708'])
 write('waiting_for_other_gpu_groups')
 wait('prepare_group_gpu3');wait('prepare_group_gpu2')
 write('evaluating_all_ten')
 run('full_evaluation_20261020')
 write('complete',result=str(HERE/'runs/full_evaluation_20261020/status.json'))
except BaseException as error:
 write('failed',error=f'{type(error).__name__}: {error}')
 raise
