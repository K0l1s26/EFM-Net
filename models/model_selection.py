import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import torch
import logging
logging.getLogger().setLevel(logging.ERROR)
import torch.nn as nn
from monai.networks.nets import UNet,UNETR,SwinUNETR,MedNeXtS,VNet
from models.SegMamba import SegMamba
from model.Ablation import MNet, FMNet,EFMNet
from model.baseline import Baseline
from models.ibm_hno_nets.hnosegxs import HNOSegXS

def build_model(model_name:str,
                in_channels:int = 4,
                out_channels: int = 3):
    name = model_name.lower()
    print(f'Instantiating the model {model_name}')
    if name == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 320),
            strides=(2, 2, 2, 2),
            num_res_units=0,
            norm="INSTANCE",
            dropout=0.0,
        )

    if name == "unetr":

        return UNETR(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=(128, 128, 128),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="conv",
            norm_name="instance",
            conv_block=True,
            res_block=True,
            dropout_rate=0.0,
        )

    if name == "swinunetr":

        return SwinUNETR(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        )
    if name == "mednext_s":
        return MedNeXtS(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            deep_supervision=False,
        )
    if name == "vnet":
        return VNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            act="prelu",
            dropout_prob_down=0.5,
            dropout_prob_up=(0.5, 0.5),
            dropout_dim=3,
            bias=False,
        )
    if name in {"waveformer", "wave_former"}:
        project_root = Path(__file__).resolve().parents[1]
        waveformer_root = project_root / "third_party" / "WaveFormer"

        if not waveformer_root.exists():
            raise FileNotFoundError(
                f"WaveFormer not found：{waveformer_root}\n"
                "Please put the official repository in third_party/WaveFormer"
            )

        waveformer_path = str(waveformer_root)
        if waveformer_path not in sys.path:
            sys.path.append(waveformer_path)

        from network_models.network_backbone import Waveformer

        return Waveformer(
            img_size=(128, 128, 128),
            patch_size=2,
            in_chans=in_channels,
            out_chans=out_channels,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            num_heads=[3, 6, 12, 24],
            drop_path_rate=0.1,
            layer_scale_init_value=1e-6,
            hidden_size=768,
            norm_name="instance",
            conv_block=True,
            res_block=True,
            spatial_dims=3,
            use_checkpoint=False,
        )
    if name in {"segmamba", "seg_mamba"}:
        return SegMamba(
            in_chans=in_channels,
            out_chans=out_channels,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0.0,
            hidden_size=768,
            norm_name="instance",
            conv_block=True,
            res_block=True,
            spatial_dims=3,
        )
    if name in {"hnoseg_xs", "hnoseg-xs", "hno"}:
        return HNOSegXS(
            in_channels=in_channels,
            out_channels=out_channels,
            filters=24,
            num_transform_blocks=[3, 3, 3, 3, 3, 3, 3, 3],
            num_modes=(10, 14, 14),
            weights_type="shared",
            use_resize=True,
            use_deep_supervision=False,
            use_unet_skip=True,
            use_block_concat=True,
            activation="selu",
            output_activation=nn.Identity(),
            ndim=5,
            device=None,
        )

    if name =='baseline':
        return Baseline(in_channels=4, num_classes=3)
    if name =='mnet':
        return MNet(in_channels=4, num_classes=3)
    if name =='fmnet':
        return FMNet(in_channels=4, num_classes=3)
    if name =='efmnet':
        return EFMNet(in_channels=4, num_classes=3)
    raise ValueError(f"Unsupported model: {model_name}")


if __name__ == '__main__':
    in_channels = 4
    num_classes = 3
    model_names = [
        'unet',
        'unetr',
        'swinunetr',
        'mednext_s',
        'segmamba',
        'waveformer',
        'vnet',
        'hnoseg_xs',
        'baseline', 'mnet', 'fmnet', 'efmnet'
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    x = torch.randn(1, 4, 128, 128, 128).to(device)

    model = build_model('vnet', in_channels, num_classes).to(device)
    model.eval()
    with torch.inference_mode():
        output = model(x)

    print(f"Output shape: {tuple(output.shape)}")
