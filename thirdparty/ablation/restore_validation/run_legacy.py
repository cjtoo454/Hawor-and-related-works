"""Compatibility wrapper for local legacy MANO assets and repository scripts."""
import sys,inspect,runpy,json,os,tempfile,fcntl
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if not hasattr(inspect,'getargspec'): inspect.getargspec=inspect.getfullargspec
for name,value in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'unicode':str,'str':str}.items():
 if name not in np.__dict__: setattr(np,name,value)
script=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(script.parent),str(ROOT),str(ROOT/'hot3d')]
sys.argv=sys.argv[1:]
locks=[]
if script.name == 'test_mdslam_hot3d.py' and '--help' not in sys.argv:
 def argument(name,default):
  return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default
 video_root=Path(argument('--video_root',str(ROOT/'datasets/hot3d_valset_export')))
 set_file=video_root/argument('--set_file','val.json')
 lock_root=Path(__file__).resolve().parent/'slam_locks'
 lock_root.mkdir(exist_ok=True)
 for sequence in sorted(json.loads(set_file.read_text())):
  handle=(lock_root/f'{sequence}.lock').open('a')
  fcntl.flock(handle,fcntl.LOCK_EX)
  locks.append(handle)
 original_savez=np.savez
 def atomic_savez(path,*args,**kwargs):
  if not isinstance(path,(str,os.PathLike)) or Path(path).parent.name != 'SLAM':
   return original_savez(path,*args,**kwargs)
  target=Path(path)
  fd,temporary=tempfile.mkstemp(prefix='.'+target.name+'.',suffix='.tmp',dir=target.parent)
  try:
   with os.fdopen(fd,'wb') as stream: original_savez(stream,*args,**kwargs)
   os.replace(temporary,target)
  finally:
   if os.path.exists(temporary):os.unlink(temporary)
 np.savez=atomic_savez
try:
 runpy.run_path(str(script),run_name='__main__')
finally:
 for handle in reversed(locks):handle.close()
