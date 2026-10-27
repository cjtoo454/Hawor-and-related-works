"""Check restored GT joints against archived references; no predictions are scored."""
import sys,json
from pathlib import Path
import numpy as np,torch,joblib
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'thirdparty/try'),str(ROOT)]
from dual_loop.geometry import ManoGeometry
from dual_loop.state import pack_state,axis_angle_to_matrix
torch.set_num_threads(2)
sequences=json.loads((ROOT/'diagnostics/pinch_hawor_vs_ema_10videos_20261018_final/protocol_frozen.json').read_text())['test_sequences']
geometry=ManoGeometry(ROOT).eval(); rows=[]
for seq in sequences:
 a=joblib.load(ROOT/'datasets/hot3d_valset_export'/seq/'anno.pth')
 with np.load(ROOT/'eval_log'/seq/'evaluation_details.npz',allow_pickle=False) as z: old=z['gt_world_joints_m'].copy()
 if len(a['trans_r'])!=len(old):raise ValueError(f'{seq}: frame mismatch')
 ids=np.flatnonzero(np.isfinite(old).all((1,2))); max_error=0.
 with torch.inference_mode():
  for start in range(0,len(ids),256):
   ix=ids[start:start+256]
   state=pack_state(a['trans_r'][ix].float(),axis_angle_to_matrix(a['rot_r'][ix].float())[:,None],axis_angle_to_matrix(a['pose_r'][ix].float().reshape(-1,15,3)),a['betas_r'][ix].float())
   joints=geometry.joints(state).numpy();max_error=max(max_error,float(np.max(np.abs(joints-old[ix]))))
 rows.append({'sequence':seq,'frames':len(old),'compared_frames':len(ids),'max_absolute_coordinate_difference_m':max_error,'reference_matches':max_error<1e-5})
 print(json.dumps(rows[-1]),flush=True)
p=Path(__file__).resolve().with_name('restored_reference_check.json');p.write_text(json.dumps({'videos':rows,'all_reference_matches':all(r['reference_matches'] for r in rows)},indent=2)+'\n')
