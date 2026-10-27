import concurrent.futures,subprocess,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
sequences=json.loads((ROOT/'diagnostics/pinch_hawor_vs_ema_10videos_20261018_final/protocol_frozen.json').read_text())['test_sequences']
def run(sequence):
 env=os.environ.copy(); env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
 command=[str(ROOT/'.env/hawor/bin/python'),str(HERE/'run_legacy.py'),str(HERE/'export_one.py'),sequence]
 with (HERE/f'export_{sequence}.log').open('w') as log:
  p=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'{sequence} failed: {p.returncode}')
 print(f'COMPLETE {sequence}',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for result in pool.map(run,sequences):pass
(ROOT/'datasets/hot3d_valset_export/val.json').write_text(json.dumps(sequences,indent=2)+'\n')
print('ALL_EXPORTS_COMPLETED',flush=True)
