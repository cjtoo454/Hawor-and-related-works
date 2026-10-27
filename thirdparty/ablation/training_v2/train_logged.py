"""Run the existing trainer unchanged, adding progress and lifecycle records."""
import sys,json,time,traceback,os
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'thirdparty/try/scripts'),str(ROOT/'thirdparty/try'),str(ROOT)]
import torch
import train_handloop
from dual_loop.handloop_training import HandLoopTrainer
original=HandLoopTrainer.optimize
started=time.monotonic()
def optimize(self,loss):
 stats=original(self,loss)
 if self.global_step==1 or self.global_step%10==0:
  row={'time_utc':datetime.now(timezone.utc).isoformat(),'step':self.global_step,
       'total_steps':self.total_steps,'elapsed_seconds':time.monotonic()-started,**stats,
       'peak_gpu_memory_gb':torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None}
  with (HERE/'progress.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  print(json.dumps(row,allow_nan=False),flush=True)
 return stats
HandLoopTrainer.optimize=optimize
status={'pid':os.getpid(),'start_utc':datetime.now(timezone.utc).isoformat(),'argv':sys.argv,'status':'running'}
def save(): (HERE/'training_status.json').write_text(json.dumps(status,indent=2)+'\n')
save()
try:
 train_handloop.main()
 status.update(status='completed',end_utc=datetime.now(timezone.utc).isoformat())
except BaseException as error:
 status.update(status='failed',error=f'{type(error).__name__}: {error}',end_utc=datetime.now(timezone.utc).isoformat())
 raise
finally:save()
