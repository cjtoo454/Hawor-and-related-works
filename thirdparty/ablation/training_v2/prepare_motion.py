"""Convert existing motion data, checking held-out participants from frozen protocol."""
import sys,json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'thirdparty/try/scripts'),str(ROOT/'thirdparty/try'),str(ROOT)]
from prepare_handloop_data import convert_motion,sha256
from dual_loop.geometry import ManoGeometry
from dual_loop.handloop_training import validate_manifest

def main():
 torch.set_num_threads(2)
 out=Path(sys.argv[1]).resolve(); out.mkdir(parents=True,exist_ok=True)
 split_path=ROOT/'datasets/hot3d_train100_export/training_split.json'
 protocol_path=ROOT/'diagnostics/pinch_hawor_vs_ema_10videos_20261018_final/protocol_frozen.json'
 split=json.loads(split_path.read_text()); heldout=json.loads(protocol_path.read_text())['test_sequences']
 people=lambda seqs:{s.split('_')[0] for s in seqs}
 assert not people(split['train'])&people(split['calibration'])
 assert not people(split['train']+split['calibration'])&people(heldout)
 manifest={'schema':'handloop_v2','train':[],'calibration':[],'test':[],'skipped':[],
 'provenance':{'split_file':str(split_path),'split_sha256':sha256(split_path),'heldout_protocol':str(protocol_path),'heldout_protocol_sha256':sha256(protocol_path)}}
 geometry=ManoGeometry(ROOT).eval()
 directory=out/'motion'; directory.mkdir(exist_ok=True)
 for group in ('train','calibration'):
  for seq in split[group]:
   source=ROOT/'thirdparty/try/data/raw_trajectories'/f'{seq}.npz'
   if not source.exists():
    manifest['skipped'].append({'sequence':seq,'reason':'source archive missing'}); continue
   target=directory/source.name
   if target.exists(): raise FileExistsError(target)
   convert_motion(source,target,geometry)
   manifest[group].append({'path':str(target),'participant':seq.split('_')[0],'source':'HOT3D','kind':'motion','sha256':sha256(target)})
   print(f'[{group}] {seq}',flush=True)
 target=out/'motion_manifest.json'; target.write_text(json.dumps(manifest,indent=2)+'\n')
 validate_manifest(target,required_kind='motion')
 print(json.dumps({'manifest':str(target),'train':len(manifest['train']),'calibration':len(manifest['calibration']),'skipped':len(manifest['skipped'])}),flush=True)
if __name__=='__main__': main()
