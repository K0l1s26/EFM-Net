import warnings
import argparse
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"mamba_ssm(\.|$)",
)
import math
import os
import tempfile
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode
from mamba_ssm import Mamba
from models.model_selection import build_model
from model.EFMNet import EFMNet

def parse_benchmark_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",default=None)
    return parser.parse_args()


INPUT_SHAPE = (1, 4, 128, 128, 128)
WARMUP = 10
ITERS = 30

def parameter_statistics(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)

    parameter_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    buffer_bytes = sum(
        b.numel() * b.element_size() for b in model.buffers()
    )

    return {
        "total": total,
        "trainable": trainable,
        "parameter_mib": parameter_bytes / 2**20,
        "state_mib": (parameter_bytes + buffer_bytes) / 2**20,
    }


def serialized_state_dict_size(model):
    fd, filename = tempfile.mkstemp(suffix=".pth")
    os.close(fd)

    try:
        torch.save(model.state_dict(), filename)
        return Path(filename).stat().st_size / 2**20
    finally:
        Path(filename).unlink(missing_ok=True)


def count_trilinear_flops(
    input_shape,
    output_size,
    align_corners,
    scales_d=None,
    scales_h=None,
    scales_w=None,
    out_shape=None,
    **kwargs,
):

    return 31 * math.prod(out_shape)


def mamba_flops(module, batch, length):

    d_model = module.d_model
    d_inner = module.d_inner
    d_state = module.d_state
    d_conv = module.d_conv
    dt_rank = module.dt_rank

    in_proj = 2 * batch * length * d_model * (2 * d_inner)
    conv1d = 2 * batch * length * d_inner * d_conv
    x_proj = (
        2 * batch * length * d_inner
        * (dt_rank + 2 * d_state)
    )
    dt_proj = 2 * batch * length * dt_rank * d_inner
    out_proj = 2 * batch * length * d_inner * d_model


    selective_scan = (
        9 * batch * length * d_inner * d_state
        + 2 * batch * length * d_inner
    )

    return (
        in_proj
        + conv1d
        + x_proj
        + dt_proj
        + selective_scan
        + out_proj
    )


def theoretical_flops(model, x):
    mamba_calls = []
    original_forwards = []

    for module in model.modules():
        if isinstance(module, Mamba):
            original_forwards.append((module, module.forward))

            def fake_forward(inp, *args, _module=module, **kwargs):
                batch, length, _ = inp.shape
                mamba_calls.append((_module, batch, length))
                return inp

            module.forward = fake_forward

    mapping = {
        torch.ops.aten.upsample_trilinear3d:
            count_trilinear_flops
    }

    try:
        with torch.no_grad():
            with FlopCounterMode(
                display=False,
                custom_mapping=mapping,
            ) as counter:
                model(x)
        op_counts = counter.get_flop_counts()["Global"]

        print("\nFLOP breakdown:")
        for op, value in sorted(
                op_counts.items(),
                key=lambda item: item[1],
                reverse=True,
        ):
            print(f"{str(op):55s} {value / 1e9:10.3f} GFLOPs")


        non_mamba_flops = counter.get_total_flops()
    finally:
        for module, original_forward in original_forwards:
            module.forward = original_forward

    total_mamba_flops = sum(
        mamba_flops(module, batch, length)
        for module, batch, length in mamba_calls
    )

    return non_mamba_flops, total_mamba_flops


@torch.inference_mode()
def benchmark(model, x, amp_dtype=None):
    use_amp = amp_dtype is not None

    def forward():
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            return model(x)

    for _ in range(WARMUP):
        output = forward()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        output = forward()
    end.record()

    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)

    batch_size = x.shape[0]
    latency_batch_ms = total_ms / ITERS
    latency_sample_ms = latency_batch_ms / batch_size
    throughput = batch_size * ITERS / (total_ms / 1000)
    peak_memory_mib = (
        torch.cuda.max_memory_allocated() / 2**20
    )

    return {
        "latency_batch_ms": latency_batch_ms,
        "latency_sample_ms": latency_sample_ms,
        "throughput": throughput,
        "peak_memory_mib": peak_memory_mib,
    }


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Model testing requires a CUDA GPU.")

    device = torch.device("cuda:0")

    model = build_model(
        model_name=args.model,
        in_channels=4,
        out_channels=3,
    ).eval()

    stats = parameter_statistics(model)
    checkpoint_mib = serialized_state_dict_size(model)

    print("=" * 60)
    print(f"Total Params:       {stats['total']:,}")
    print(f"Params (M):         {stats['total'] / 1e6:.6f} M")
    print(f"Trainable Params:   {stats['trainable'] / 1e6:.6f} M")
    print(f"Parameter memory:   {stats['parameter_mib']:.3f} MiB")
    print(f"Params + buffers:   {stats['state_mib']:.3f} MiB")
    print(f"state_dict file:    {checkpoint_mib:.3f} MiB")

    model.requires_grad_(False)
    model = model.to(device)

    x = torch.randn(INPUT_SHAPE, device=device)

    non_mamba, mamba = theoretical_flops(model, x)
    total_flops = non_mamba + mamba

    print("=" * 60)
    print(f"Input shape:        {INPUT_SHAPE}")
    print(f"Non-Mamba GFLOPs:  {non_mamba / 1e9:.3f}")
    print(f"Mamba GFLOPs:      {mamba / 1e9:.3f}")
    print(f"Estimated GFLOPs:  {total_flops / 1e9:.3f}")
    print(f"Estimated FLOPs:   {total_flops:,.0f}")

    cases = [("FP32", None)]

    if torch.cuda.is_bf16_supported():
        cases.append(("AMP-BF16", torch.bfloat16))

    print("=" * 60)
    print(f"GPU:               {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:           {torch.__version__}")

    for name, dtype in cases:
        result = benchmark(model, x, amp_dtype=dtype)

        print("-" * 60)
        print(f"Precision:          {name}")
        print(
            f"Latency/batch:      "
            f"{result['latency_batch_ms']:.3f} ms"
        )
        print(
            f"Latency/sample:     "
            f"{result['latency_sample_ms']:.3f} ms"
        )
        print(
            f"Throughput:         "
            f"{result['throughput']:.3f} patches/s"
        )
        print(
            f"Peak GPU memory:    "
            f"{result['peak_memory_mib']:.1f} MiB"
        )

