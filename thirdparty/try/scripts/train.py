#!/usr/bin/env python3
"""Train the full observer on independently generated visual observations and GT labels."""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from torch.nn import functional as F
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(REPOSITORY_ROOT)]
from dual_loop.geometry import ManoGeometry
from dual_loop.model import DualLoopConfig, DualLoopObserver
from dual_loop.networks import WORLD_MODEL_FORMAT_VERSION, WORLD_MODEL_VARIANTS
from dual_loop.state import STATE_DIM


def world_model_variant_from_checkpoint(checkpoint):
    if checkpoint.get('stage') != 'world_model_pretrain':
        raise ValueError('Expected prior pretraining checkpoint')
    model_config = checkpoint.get('model_config')
    if not isinstance(model_config, dict):
        raise ValueError('Prior checkpoint is missing model architecture metadata')
    variant = checkpoint.get('variant', model_config.get('variant', 'legacy'))
    version = checkpoint.get(
        'world_model_format_version',
        model_config.get('format_version', WORLD_MODEL_FORMAT_VERSION.get(variant, -1)))
    if variant not in WORLD_MODEL_VARIANTS:
        raise ValueError(f'Prior checkpoint has unsupported world-model variant {variant!r}')
    if model_config.get('variant', variant) != variant:
        raise ValueError('Prior checkpoint world-model variant metadata disagrees')
    if version != WORLD_MODEL_FORMAT_VERSION[variant]:
        raise ValueError('Prior checkpoint has an incompatible world-model format version')
    if model_config.get('state_dim', STATE_DIM) != STATE_DIM:
        raise ValueError('Prior checkpoint has an incompatible state dimension')
    if 'hidden_dim' not in model_config:
        raise ValueError('Prior checkpoint is missing its hidden dimension')
    return variant, int(model_config['hidden_dim'])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', type=Path, default=PROJECT_ROOT/'data/prepared')
    p.add_argument('--split_file', type=Path, required=True, help='JSON train/calibration sequence lists')
    p.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/default.yaml')
    p.add_argument('--output', type=Path, default=PROJECT_ROOT/'checkpoints/dual_loop.pt')
    p.add_argument('--world_checkpoint', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--steps_per_epoch', type=int, default=300)
    p.add_argument('--validation_steps', type=int, default=30)
    p.add_argument('--window', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=2e-4)
    p.add_argument('--quality_error_m', type=float, default=0.03)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=7)
    return p.parse_args()


def image_input(path, size, device):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hint = float(np.clip(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())/7.,0,1))
    image = cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(size,size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
    return tensor, torch.tensor([[hint]],device=device)


def state_loss(pred, target):
    return (10*F.smooth_l1_loss(pred[...,:3],target[...,:3])
            + F.smooth_l1_loss(pred[...,3:99],target[...,3:99])
            + .1*F.smooth_l1_loss(pred[...,99:],target[...,99:]))


def load_split(args):
    split = json.loads(args.split_file.read_text())
    train, calibration = split['train'], split['calibration']
    if not train or not calibration or set(train)&set(calibration):
        raise ValueError('Nonempty disjoint train/calibration splits required')
    participants = lambda seqs: {x.split('_')[0] for x in seqs}
    if participants(train)&participants(calibration):
        raise ValueError('Participant leakage between train and calibration')
    heldout = REPOSITORY_ROOT/'datasets/hot3d_valset_export/val.json'
    if heldout.exists() and participants(train+calibration)&participants(json.loads(heldout.read_text())):
        raise ValueError('Training/calibration participant overlaps held-out test')
    groups = []
    for sequences in (train,calibration):
        archives = []
        for seq in sequences:
            with np.load(args.data_root/f'{seq}.npz',allow_pickle=False) as z:
                d = {key:z[key].copy() for key in z.files}
            required = {'segment_id','timestamp_sec','coordinate_frame','oracle_assisted',
                        'input_provenance_json','oracle_assisted_status'}
            if not required.issubset(d):
                raise ValueError(f'{seq}: regenerate aligned data with current preparation script')
            if bool(d['oracle_assisted']):
                raise ValueError(f'{seq}: GT boxes forbidden for full observer training')
            provenance=json.loads(str(d['input_provenance_json'].item()))
            if provenance.get('raw_export_diagnostic'):
                raise ValueError(f'{seq}: diagnostic raw export forbidden for full observer training')
            if (provenance.get('manifest_status')!='verified'
                    or str(d['oracle_assisted_status'].item())!='verified_no_oracle'
                    or provenance.get('oracle_assisted') is not False):
                raise ValueError(f'{seq}: frontend origin is unverified; regenerate prepared observations')
            t = d['timestamp_sec']
            if np.any(np.diff(t)<=0):
                raise ValueError(f'{seq}: nonmonotonic timestamps')
            d['sequence'] = seq
            archives.append(d)
        groups.append(archives)
    return split,groups


def window_index(archives, window):
    items=[]
    for sequence_id,d in enumerate(archives):
        n=len(d['visual_valid'])
        # Preserve time; no stepping across segment boundaries or dropped timestamps.
        good = (np.diff(d['segment_id'])==0)&(np.abs(np.diff(d['timestamp_sec'])-1/30)<.005)
        for start in range(max(0,n-window+1)):
            if (d['visual_valid'][start] and d['gt_valid'][start:start+window].any()
                    and good[start:start+window-1].all()):
                items.append((sequence_id,start))
    if not items:
        raise ValueError('No continuous windows with an observed initialization')
    return items


def run_window(model, geometry, sequence, start, args, device):
    history=[]; losses=[]; config=model.config
    for frame in range(start,start+args.window):
        valid=bool(sequence['visual_valid'][frame])
        if valid:
            visual=torch.as_tensor(sequence['visual_state'][frame:frame+1],device=device,dtype=torch.float32)
        else:
            visual=history[-1]
        past=torch.stack(history,dim=1) if history else visual.new_empty((1,0,config.state_dim))
        rotation=torch.as_tensor(sequence['rotation_w2c'][frame:frame+1],device=device,dtype=torch.float32)
        translation=torch.as_tensor(sequence['translation_w2c'][frame:frame+1],device=device,dtype=torch.float32)
        focals=sequence['focal_length']; focal=focals[frame]
        if not np.isfinite(focal):
            finite=focals[np.isfinite(focals)]
            if not len(finite):raise ValueError('No finite focal length')
            focal=finite[0]
        focal=torch.tensor([float(focal)],device=device)
        size=torch.as_tensor(sequence['image_size'],device=device,dtype=torch.float32).unsqueeze(0)
        def projector(state):
            return geometry.project_world_joints(state,rotation,translation,focal,size)
        image,hint=image_input(sequence['image_paths'][frame],config.image_size,device)
        if not valid:
            hint = torch.zeros_like(hint)
        output=model.forward_step(image,visual,past,projector,torch.tensor([valid],device=device),hint)
        corrected=output['state']; prior=output['prior']
        if bool(sequence['gt_valid'][frame]):
            target=torch.as_tensor(sequence['gt_state'][frame:frame+1],device=device,dtype=torch.float32)
            pred_joints=geometry.joints(corrected); target_joints=geometry.joints(target)
            error=(pred_joints-target_joints).norm(dim=-1).mean(-1,keepdim=True)
            # A calibrated reliability target; no GT enters inference.
            reliability=torch.exp(-error.detach()/args.quality_error_m)
            quality_loss=F.binary_cross_entropy_with_logits(output['visual_quality_logit'],reliability)
            logvar=output['prior_logvar']
            nll=.5*((prior-target).square()*torch.exp(-logvar)+logvar).mean()
            losses.append(state_loss(corrected,target)+.5*state_loss(prior,target)
                          +20*F.smooth_l1_loss(pred_joints,target_joints)+.2*quality_loss+.01*nll)
        # Missing supervision masks the loss only, never deletes a physical timestep.
        history.append(corrected if config.corrected_state_feedback else visual)
        history=history[-config.history_length:]
    return torch.stack(losses).mean() if losses else None


def main():
    args=parse_args()
    if args.window<2 or args.quality_error_m<=0:raise ValueError('Invalid window/quality tolerance')
    if min(args.epochs,args.steps_per_epoch,args.validation_steps)<1:
        raise ValueError('Epoch and step counts must be positive')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Output checkpoint already exists; choose a new output or use --resume')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    raw=yaml.safe_load(args.config.read_text())
    model_values=dict(raw['model'])
    device=torch.device(args.device)
    prior_checkpoint=None
    if args.world_checkpoint:
        prior_checkpoint=torch.load(args.world_checkpoint,map_location=device,weights_only=False)
        prior_variant, prior_hidden_dim=world_model_variant_from_checkpoint(prior_checkpoint)
        configured_variant=model_values.get('world_model_variant')
        if configured_variant is not None and configured_variant!=prior_variant:
            raise ValueError('Full-model config world_model_variant disagrees with prior checkpoint')
        configured_hidden_dim=int(model_values.get('hidden_dim',DualLoopConfig().hidden_dim))
        if configured_hidden_dim!=prior_hidden_dim:
            raise ValueError('Full-model hidden_dim disagrees with prior checkpoint')
        model_values['world_model_variant']=prior_variant
    raw['model']=model_values
    config=DualLoopConfig(**model_values)
    # Save the resolved default as architecture metadata for future inference
    # and resume, including runs trained without a prior checkpoint.
    raw['model']['world_model_variant']=config.world_model_variant
    model=DualLoopObserver(config).to(device)
    geometry=ManoGeometry(REPOSITORY_ROOT).to(device).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    split,groups=load_split(args); indices=[window_index(g,args.window) for g in groups]
    begin=0
    if args.world_checkpoint:
        ckpt=prior_checkpoint
        prior_run=ckpt.get('run', {})
        prior_train=set(prior_run.get('train_participants', []))
        prior_calibration=set(prior_run.get('calibration_participants', []))
        full_train={s.split('_')[0] for s in split['train']}
        full_calibration={s.split('_')[0] for s in split['calibration']}
        if not prior_train or not prior_calibration:
            raise ValueError('Prior checkpoint is missing participant provenance')
        if prior_train & full_calibration or prior_calibration & full_train:
            raise ValueError('Prior and full observer train/calibration participant splits disagree')
        heldout=REPOSITORY_ROOT/'datasets/hot3d_valset_export/val.json'
        if heldout.exists():
            test_participants={s.split('_')[0] for s in json.loads(heldout.read_text())}
            if (prior_train|prior_calibration)&test_participants:
                raise ValueError('Prior checkpoint overlaps held-out test participants')
        if ckpt.get('coordinate_normalization')!='first_history_wrist':
            raise ValueError('Prior uses an incompatible state coordinate convention')
        try:
            model.world_model.load_state_dict(ckpt['world_model'],strict=True)
        except RuntimeError as error:
            raise ValueError(
                'Prior checkpoint weights do not match the resolved world-model architecture') from error
    if args.resume:
        ckpt=torch.load(args.resume,map_location=device,weights_only=False)
        if ckpt.get('stage')!='dual_loop':raise ValueError('Expected full observer checkpoint')
        saved_config=ckpt.get('config')
        if isinstance(saved_config,dict) and isinstance(saved_config.get('model'),dict):
            saved_config=dict(saved_config)
            saved_config['model']=dict(saved_config['model'])
            saved_config['model'].setdefault('world_model_variant','legacy')
        if saved_config!=raw or ckpt['split']!=split:raise ValueError('Resume split/config mismatch')
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer']); begin=ckpt['epoch']
        random.setstate(ckpt['python_rng']); np.random.set_state(ckpt['numpy_rng']); torch.set_rng_state(ckpt['torch_rng'].cpu())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    best=float('inf')
    if args.resume:best=ckpt.get('best_validation_loss',best)
    run={'stage':'dual_loop','args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'split':split,'config':raw,
         'train_windows':len(indices[0]),'calibration_windows':len(indices[1]),'quality_target':'exp(-mean_joint_error_m/quality_error_m)'}
    args.output.with_suffix('.run.json').write_text(json.dumps(run,indent=2))
    for epoch in range(begin,args.epochs):
        model.train(); total=0.; count=0
        for step in range(args.steps_per_epoch):
            sid,start=random.choice(indices[0]); optimizer.zero_grad(set_to_none=True)
            loss=run_window(model,geometry,groups[0][sid],start,args,device)
            if loss is None:continue
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            total+=loss.item();count+=1
            if (step+1)%10==0:print(json.dumps({'epoch':epoch+1,'step':step+1,'train_loss':total/count}),flush=True)
        model.eval(); validation=[]; rng=random.Random(args.seed+1)
        with torch.no_grad():
            for _ in range(args.validation_steps):
                sid,start=rng.choice(indices[1]); loss=run_window(model,geometry,groups[1][sid],start,args,device)
                if loss is not None:validation.append(loss.item())
        if not validation:raise ValueError('No calibration supervision')
        score=float(np.mean(validation)); improved=score<best;best=min(best,score)
        checkpoint={'stage':'dual_loop','model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,
                    'config':raw,'split':split,'best_validation_loss':best,'python_rng':random.getstate(),
                    'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state()}
        temporary=args.output.with_suffix('.tmp');torch.save(checkpoint,temporary);temporary.replace(args.output)
        if improved:torch.save(checkpoint,args.output.with_name(args.output.stem+'.best.pt'))
        record={'epoch':epoch+1,'train_loss':total/max(count,1),'calibration_loss':score}
        with args.output.with_suffix('.metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)

if __name__=='__main__':main()
