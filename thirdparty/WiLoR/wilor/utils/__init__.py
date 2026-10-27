import torch
from typing import Any

from .pose_utils import eval_pose, Evaluator


def __getattr__(name: str):
    """Load optional OpenGL renderers only when explicitly requested."""
    if name == "Renderer":
        from .renderer import Renderer
        return Renderer
    if name == "MeshRenderer":
        from .mesh_renderer import MeshRenderer
        return MeshRenderer
    if name == "SkeletonRenderer":
        from .skeleton_renderer import SkeletonRenderer
        return SkeletonRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def recursive_to(x: Any, target: torch.device):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x (Any): Batch of data.
        target (torch.device): Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x
