"""Run from v4: python -m kernels.test_chunkwise_kda [--benchmark]."""

import argparse

import torch
import triton.testing

from config import dtypes
from .chunkwise_kda import chunk_kda, pt_chunk_kda
from .recurrent_kda import pt_recurrent_kda_fwd, recurrent_kda_fwd


def make_inputs(B, T, H, K, V, dtype=torch.float32, gates="normal"):
    q = torch.randn(B, T, H, K, device="cuda").to(dtype)
    k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, device="cuda").to(dtype)
    g = -torch.rand(B, T, H, K, device="cuda")

    if gates == "zero":
        g.zero_()
    elif gates == "strong":
        g.mul_(20)
    beta = torch.rand(B, T, H, device="cuda")
    state = torch.randn(B, H, K, V, device="cuda")
    return q, k, v, g, beta, state, K ** -0.5


def check_case(
        shape, chunk_size, dtype=torch.float32, gates="normal", precision="tf32x3",
        zero_beta=False, repeated_keys=False,
    ):
    args = make_inputs(*shape, dtype=dtype, gates=gates)

    if repeated_keys:
        args[1].fill_(shape[-2] ** -0.5)
        args[4].fill_(1)

    if zero_beta:
        args[4].zero_()
        args[5].zero_()
    original_state = args[5].clone()
    reference, reference_state = pt_recurrent_kda_fwd(*args)
    pt_output, pt_state = pt_chunk_kda(*args, chunk_size=chunk_size, output_dtype=torch.float32)
    output, state = chunk_kda(
        *args,
        chunk_size=chunk_size,
        input_precision=precision,
        output_dtype=torch.float32,
    )

    for actual, expected in ((pt_output, reference), (output, reference),
                             (pt_state, reference_state), (state, reference_state)):
        assert torch.isfinite(actual).all(), "nonfinite chunk result"
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(args[5], original_state, rtol=0, atol=0)
    print(
        f"shape={shape} C={chunk_size} {dtype} gates={gates} {precision}: "
        f"output diff={(output-reference).abs().max().item():.3g}, "
        f"state diff={(state-reference_state).abs().max().item():.3g}",
        flush=True,
    )
    return args, output, state


@torch.no_grad()
def correctness():
    torch.manual_seed(42)
    # Make both recurrent references produce FP32 outputs for comparison.
    original_output = dtypes.output
    dtypes.output = "fp32"

    try:
        args, output, state = check_case((1, 256, 8, 32, 64), 32)
        recurrent_output, recurrent_state = recurrent_kda_fwd(*args)
        torch.testing.assert_close(output, recurrent_output, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(state, recurrent_state, rtol=2e-4, atol=2e-5)
        check_case((2, 77, 3, 17, 45), 32)
        check_case((1, 1, 2, 7, 9), 16, precision="ieee")
        check_case((1, 65, 2, 64, 128), 64, dtype=torch.bfloat16)
        check_case((1, 97, 2, 32, 64), 32, gates="zero")
        check_case((1, 129, 2, 32, 64), 64, gates="zero", repeated_keys=True)
        check_case((1, 65, 2, 32, 64), 64, gates="strong")
        check_case((1, 33, 2, 32, 64), 32, zero_beta=True)

        # Cached-state continuation at a boundary unrelated to chunk_size.
        split = 45
        first = [x[:, :split] for x in args[:5]] + [args[5], args[6]]
        o1, s1 = chunk_kda(*first, output_dtype=torch.float32)
        second = [x[:, split:] for x in args[:5]] + [s1, args[6]]
        o2, s2 = chunk_kda(*second, output_dtype=torch.float32)
        torch.testing.assert_close(torch.cat((o1, o2), dim=1), output, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(s2, state, rtol=2e-4, atol=2e-5)

        # Configured BF16 storage and empty sequences.
        stored, stored_state = chunk_kda(*args, output_dtype=torch.bfloat16)
        torch.testing.assert_close(stored.float(), output, rtol=4e-3, atol=4e-3)
        torch.testing.assert_close(stored_state, state, rtol=0, atol=0)
        empty = [x[:, :0] for x in args[:5]] + [args[5], args[6]]
        empty_output, empty_state = chunk_kda(*empty)
        assert empty_output.shape == (1, 0, 8, 64)
        torch.testing.assert_close(empty_state, args[5], rtol=0, atol=0)
        print("Continuation, output dtype, and empty-sequence checks passed.", flush=True)
    finally:
        dtypes.output = original_output


@torch.no_grad()
def benchmark():
    print("\nGPU forward timings, including scratch allocations and all launches:", flush=True)

    for T in (256, 2048):
        args = make_inputs(1, T, 8, 64, 128, dtype=torch.bfloat16)
        # Warm compilation separately from timing.
        recurrent_kda_fwd(*args)

        for C in (32, 64):
            chunk_kda(*args, chunk_size=C)
        torch.cuda.synchronize()
        recurrent_ms = triton.testing.do_bench(lambda: recurrent_kda_fwd(*args), warmup=50, rep=150)
        print(f"T={T} recurrent: {recurrent_ms:.3f} ms", flush=True)

        for C in (32, 64):
            chunk_ms = triton.testing.do_bench(
                lambda: chunk_kda(*args, chunk_size=C),
                warmup=50,
                rep=150,
            )
            print(
                f"T={T} chunk C={C}: {chunk_ms:.3f} ms ({recurrent_ms/chunk_ms:.2f}x)",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    options = parser.parse_args()
    correctness()

    if options.benchmark:
        benchmark()
    print("All chunkwise forward checks passed.", flush=True)
