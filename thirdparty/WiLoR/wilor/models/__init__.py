from .mano_wrapper import MANO
from .wilor import WiLoR

from .discriminator import Discriminator

def load_wilor(checkpoint_path, cfg_path, init_renderer=True):
    from pathlib import Path
    from wilor.configs import get_config
    package_root = Path(__file__).resolve().parents[2]
    print('Loading ', checkpoint_path)
    model_cfg = get_config(cfg_path, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if ('vit' in model_cfg.MODEL.BACKBONE.TYPE) and ('BBOX_SHAPE' not in model_cfg.MODEL):
        
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()
        
        # Update config to be compatible with demo

    if ('DATA_DIR' in model_cfg.MANO):
        local_mano_dir = package_root / 'mano_data'
        mano_candidates = [
            local_mano_dir,
            package_root.parent.parent / '_DATA' / 'data' / 'mano',
            package_root.parent.parent / 'hot3d' / 'mano_v1_2' / 'models',
        ]
        mano_model_dir = next(
            (path for path in mano_candidates if (path / 'MANO_RIGHT.pkl').is_file()),
            local_mano_dir,
        )
        model_cfg.defrost()
        model_cfg.MANO.DATA_DIR = str(local_mano_dir)
        model_cfg.MANO.MODEL_PATH = str(mano_model_dir)
        model_cfg.MANO.MEAN_PARAMS = str(local_mano_dir / 'mano_mean_params.npz')
        model_cfg.freeze()

    model = WiLoR.load_from_checkpoint(
        checkpoint_path,
        strict=False,
        cfg=model_cfg,
        init_renderer=init_renderer,
    )
    return model, model_cfg
