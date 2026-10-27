"""Live visual adaptation and sequential HandLoop inference share this path."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist

from .handloop_frontend import HaWoRFrontend, load_hawor, project_mano16
from .handloop_state import to_world, state_to_legacy, to_local
from .handloop_probability import log_prob


class HandLoopSystem(nn.Module):
    architecture = "handloop_v2"

    def __init__(self, hawor, observer, geometry):
        super().__init__()
        self.hawor = HaWoRFrontend(hawor, geometry)
        self.observer = observer
        self.geometry = geometry
        self.geometry.requires_grad_(False)

    @property
    def twm(self):
        return self.observer.twm

    @property
    def visual(self):
        return self.observer.visual


def build_system(hawor_checkpoint, config, device="cpu"):
    from .geometry import ManoGeometry
    from .handloop_model import HandLoopConfig, HandLoopObserver
    geometry = ManoGeometry(Path(__file__).resolve().parents[3]).to(device)
    observer = HandLoopObserver(HandLoopConfig.from_dict(config)).to(device)
    return HandLoopSystem(load_hawor(hawor_checkpoint, device), observer, geometry).to(device)


def move_batch(batch, device):
    return {k: move_batch(v, device) if isinstance(v, dict) else v.to(device)
            if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class FeedbackHistory:
    """One detached commit per real frame; usable across inference contexts."""
    def __init__(self, length=30):
        self.length = length
        self.states, self.available, self.times, self.dispersions, self.masks = [], [], [], [], []
        self.last_time = self.last_segment = self.last_hand = None

    def before(self, time, segment, hand):
        reset = torch.zeros_like(time, dtype=torch.bool)
        if self.last_time is not None:
            gap = time - self.last_time
            reset = (gap <= 0) | (gap > .25) | (segment != self.last_segment) | (hand != self.last_hand)
            self.masks = [mask & ~reset for mask in self.masks]
        dt = (time - self.last_time).float() if self.last_time is not None else time.new_full(time.shape, 1 / 30).float()
        dt = torch.where(reset, torch.full_like(dt, 1 / 30), dt)
        self.last_time, self.last_segment, self.last_hand = time.detach(), segment.detach(), hand.detach()
        return dt

    def tensors(self, raw, time):
        b = raw.shape[0]
        if not self.states:
            return dict(history=raw.new_empty(b, 0, 99), history_available=torch.empty(b, 0, 17, dtype=torch.bool, device=raw.device),
                        history_mask=torch.empty(b, 0, dtype=torch.bool, device=raw.device),
                        history_times=raw.new_empty(b, 0), history_dispersion=raw.new_empty(b, 0, 17),
                        dispersion_known=torch.empty(b, 0, dtype=torch.bool, device=raw.device))
        masks = torch.stack(self.masks, 1)
        return dict(history=torch.stack(self.states, 1), history_available=torch.stack(self.available, 1) & masks[..., None],
                    history_mask=masks, history_times=(torch.stack(self.times, 1) - time[:, None]).float(),
                    history_dispersion=torch.stack(self.dispersions, 1), dispersion_known=masks)

    def commit(self, state, available, time, dispersion):
        for store, value in ((self.states, state), (self.available, available), (self.times, time),
                             (self.dispersions, dispersion), (self.masks, torch.ones_like(time, dtype=torch.bool))):
            store.append(value.detach())
            del store[:-self.length]


def encode_contexts(system, hawor):
    frames = hawor["img"].shape[1]
    if frames % 16:
        raise ValueError("Visual context length must be divisible by16; pad only at inference boundary")
    outputs = [system.hawor({key: value[:, start:start + 16] for key, value in hawor.items()})
               for start in range(0, frames, 16)]
    return {key: torch.cat([out[key] for out in outputs], 1) for key in outputs[0]}


def rollout(system, batch, feedback=None, real_frames=None, future_times=None):
    """Only observation/camera keys are consumed. Targets cannot enter history."""
    frontend = encode_contexts(system, batch["hawor"])
    frames = frontend["state_camera"].shape[1] if real_frames is None else real_frames
    feedback = feedback or FeedbackHistory(system.observer.config.twm.history_length)
    outputs = []
    for frame in range(frames):
        time = batch["timestamp_sec"][:, frame]
        hand = batch["handedness"][:, frame]
        dt = feedback.before(time, batch["segment_id"][:, frame], hand)
        rwc, twc = batch["rotation_w2c"][:, frame], batch["translation_w2c"][:, frame]
        rcw = rwc.transpose(-1, -2)
        raw = to_world(frontend["state_camera"][:, frame], rcw, -(rcw @ twc[..., None]).squeeze(-1))
        history = feedback.tensors(raw, time)
        beta = frontend["betas"][:, frame]

        def projector(state):
            return project_mano16(state, beta, system.geometry, rwc, twc,
                                  batch["hawor"]["img_focal"][:, frame],
                                  batch["hawor"]["img_center"][:, frame], batch["pixel_to_patch"][:, frame])

        output = system.observer.forward_step(
            patches=frontend["patches"][:, frame], raw_state=raw,
            raw_available=batch["raw_available"][:, frame], dt=dt, handedness=hand,
            rotation_w2c=rwc, translation_w2c=twc, projector=projector,
            image_available=batch["image_available"][:, frame], future_times=future_times,
            **history)
        available = batch["raw_available"][:, frame].clone()
        if history["history_available"].shape[1]:
            available |= history["history_available"].any(1)
        output["available"] = available
        output["betas"] = beta
        feedback.commit(output["state"], available, time, output["dispersion"])
        outputs.append(output)
    return outputs, frontend, feedback


def global_mean(total, count):
    count = torch.as_tensor(count, device=total.device, dtype=total.dtype)
    if dist.is_available() and dist.is_initialized():
        count = count.clone()
        dist.all_reduce(count)
        return total * dist.get_world_size() / count.clamp_min(1)
    return total / count.clamp_min(1)


def visual_clip_loss(system, batch, config=None):
    config = config or {}
    outputs, frontend, _ = rollout(system, batch)
    zero = frontend["state_camera"].sum() * 0
    sums = {key: zero for key in ("raw_wrist_m", "joint_m", "state_nll")}
    counts = {key: 0 for key in sums}
    for frame, output in enumerate(outputs):
        supervised = batch["supervise"][:, frame]
        target = batch["target_state"][:, frame]
        available = batch["target_available"][:, frame]
        wrist = supervised & available[:, 0] & batch["raw_available"][:, frame, 0]
        if wrist.any():
            camera = frontend["state_camera"][:, frame]
            rwc, twc = batch["rotation_w2c"][:, frame], batch["translation_w2c"][:, frame]
            rcw = rwc.transpose(-1, -2)
            raw = to_world(camera, rcw, -(rcw @ twc[..., None]).squeeze(-1))
            sums["raw_wrist_m"] = sums["raw_wrist_m"] + (raw[wrist, :3] - target[wrist, :3]).norm(dim=-1).sum()
            counts["raw_wrist_m"] += int(wrist.sum())
        complete = supervised & available.all(-1)
        if complete.any():
            local_target = to_local(target[complete], output["anchor_rotation"][complete], output["anchor_position"][complete])
            distribution = {key: value[complete] for key, value in output["current"].items()}
            sums["state_nll"] = sums["state_nll"] - log_prob(distribution, local_target).sum()
            counts["state_nll"] += int(complete.sum())
        valid_joints = batch["joint_valid"][:, frame] & supervised[:, None]
        valid_rows = valid_joints.any(-1)
        if valid_rows.any():
            legacy = state_to_legacy(output["state"][valid_rows], frontend["betas"][valid_rows, frame], system.geometry)
            predicted = system.geometry.joints(legacy)
            labels = batch["target_joints"][valid_rows, frame]
            mask = valid_joints[valid_rows]
            sums["joint_m"] = sums["joint_m"] + (predicted[mask] - labels[mask]).norm(dim=-1).sum()
            counts["joint_m"] += int(mask.sum())
    losses = {key: global_mean(value, counts[key]) for key, value in sums.items()}
    if not any(counts.values()) and not (dist.is_available() and dist.is_initialized()):
        raise ValueError("Visual clip has no enabled supervision")
    losses["loss"] = (config.get("state_weight", 1e-4) * losses["state_nll"]
                      + config.get("wrist_weight", 1.) * losses["raw_wrist_m"] / config.get("wrist_scale", .05)
                      + config.get("joint_weight", 1.) * losses["joint_m"] / config.get("joint_scale", .05))
    return losses
