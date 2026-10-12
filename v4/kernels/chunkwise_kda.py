import torch
import triton
import triton.language as tl

from config import dtypes, torch_map


def pt_chunk_kda(q, k, v, g, beta, state, scale, chunk_size=32, output_dtype=None):

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    output_dtype = output_dtype or torch_map[dtypes.output]
    q, k, v, g = [x.float().permute(0, 2, 1, 3) for x in (q, k, v, g)]
    beta = beta.float().permute(0, 2, 1)
    current = state.float()
    outputs = []

    for start in range(0, q.shape[2], chunk_size):
        qc, kc, vc, gc = [x[:, :, start:start + chunk_size] for x in (q, k, v, g)]
        bc = beta[:, :, start:start + chunk_size]
        C = qc.shape[2]
        G = gc.cumsum(dim=2)
        rt = torch.arange(C, device=q.device)

        causal = rt[:, None] >= rt[None, :]

        difference = G.unsqueeze(3) - G.unsqueeze(2)
        pair_decay = difference.masked_fill(~causal[..., None], 0).exp()
        R = torch.tril(
            bc[..., None] * torch.einsum("bhijk,bhik,bhjk->bhij", pair_decay, kc, kc),
            diagonal=-1,
        )
        L = torch.eye(C, device=q.device, dtype=torch.float32) + R
        P = bc[..., None] * kc * G.exp()
        U = torch.linalg.solve_triangular(
            L,
            bc[..., None] * vc,
            upper=False,
            unitriangular=True,
        )
        W = torch.linalg.solve_triangular(L, P, upper=False, unitriangular=True)
        E = U - W @ current
        A = scale * torch.tril(torch.einsum("bhijk,bhik,bhjk->bhij", pair_decay, qc, kc))
        outputs.append((scale * qc * G.exp()) @ current + A @ E)
        F = kc * (G[:, :, -1:, :] - G).exp()
        current = G[:, :, -1].exp().unsqueeze(-1) * current + F.transpose(-1, -2) @ E

    if not outputs:
        return v.permute(0, 2, 1, 3).to(output_dtype).contiguous(), current.to(state.dtype).clone()

    return torch.cat(outputs, dim=2).permute(0, 2, 1, 3).to(output_dtype).contiguous(), current.to(state.dtype)


@triton.jit
def _prepare_gates(
        q, k, g, beta, G, P, Qbar, F, D, scale,
        T: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        NC: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
    ):
    tile = tl.program_id(0)
    bh, chunk = tile // NC, tile % NC
    b, h = bh // H, bh % H
    rt, rk = tl.arange(0, BT), tl.arange(0, BK)

    tt = chunk * BT + rt

    token_head = (b * T + tt) * H + h

    offsets = (
        token_head[:, None] * K
        + rk[None, :]
    )
    mask = (tt[:, None] < T) & (rk[None, :] < K)

    gc = tl.load(g + offsets, mask=mask, other=0).to(tl.float32)
    kc = tl.load(k + offsets, mask=mask, other=0).to(tl.float32)
    qc = tl.load(q + offsets, mask=mask, other=0).to(tl.float32)
    bc = tl.load(beta + token_head, mask=tt < T, other=0).to(tl.float32)

    prefix = tl.cumsum(gc, axis=0)
    last = tl.minimum(BT, T - chunk * BT) - 1
    end = tl.sum(tl.where(rt[:, None] == last, prefix, 0), axis=0)
    scratch = (
        tile * BT * K
        + rt[:, None] * K
        + rk[None, :]
    )

    scratch_mask = rk[None, :] < K
    decay = tl.exp(prefix)
    tl.store(G + scratch, prefix, mask=scratch_mask)
    tl.store(P + scratch, bc[:, None] * kc * decay, mask=scratch_mask)
    tl.store(Qbar + scratch, scale * qc * decay, mask=scratch_mask)
    tl.store(F + scratch, kc * tl.exp(end[None, :] - prefix), mask=scratch_mask)
    tl.store(D + tile * K + rk, tl.exp(end), mask=rk < K)


@triton.jit
def _prepare_interactions(
        q, k, beta, G, R, A, scale,
        T: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        NC: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
    ):

    tile, i = tl.program_id(0), tl.program_id(1)
    bh, chunk = tile // NC, tile % NC
    b, h = bh // H, bh % H
    rj, rk = tl.arange(0, BT), tl.arange(0, BK)
    ti, tj = chunk * BT + i, chunk * BT + rj
    row_head = (b * T + ti) * H + h
    col_head = (b * T + tj) * H + h

    qi = tl.load(q + row_head * K + rk, mask=(ti < T) & (rk < K), other=0).to(tl.float32)
    ki = tl.load(k + row_head * K + rk, mask=(ti < T) & (rk < K), other=0).to(tl.float32)
    kj = tl.load(
        k + col_head[:, None] * K + rk[None, :],
        mask=(tj[:, None] < T) & (rk[None, :] < K),
        other=0,
    ).to(tl.float32)
    gi = tl.load(G + tile * BT * K + i * K + rk, mask=rk < K, other=0)
    gj = tl.load(
        G + tile * BT * K + rj[:, None] * K + rk[None, :],
        mask=rk[None, :] < K,
        other=0,
    )

    causal = (rj <= i) & (ti < T) & (tj < T)
    decay = tl.exp(tl.where(causal[:, None], gi[None, :] - gj, 0))
    weighted_keys = kj * decay
    bi = tl.load(beta + row_head, mask=ti < T, other=0).to(tl.float32)
    r = bi * tl.sum(weighted_keys * ki[None, :], axis=1)
    a = scale * tl.sum(weighted_keys * qi[None, :], axis=1)

    offsets = tile * BT * BT + i * BT + rj
    tl.store(R + offsets, tl.where(causal & (rj < i), r, 0))
    tl.store(A + offsets, tl.where(causal, a, 0))


@triton.jit
def _prepare_solve(
        v, beta, R, P, U, W,
        T: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        NC: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        LOG_BT: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
    tile = tl.program_id(0)
    bh, chunk = tile // NC, tile % NC
    b, h = bh // H, bh % H
    rt, rk, rv = tl.arange(0, BT), tl.arange(0, BK), tl.arange(0, BV)

    matrix_offsets = (
        tile * BT * BT
        + rt[:, None] * BT
        + rt[None, :]
    )
    r = tl.load(R + matrix_offsets)
    row, col = rt[:, None], rt[None, :]


    inverse = (row == col).to(tl.float32) - tl.where(row // 2 == col // 2, r, 0)


    for stage in tl.static_range(1, LOG_BT):
        half = 1 << stage
        width = 2 * half
        cross_mask = ((row // width == col // width)
                      & (row % width >= half) & (col % width < half))
        cross = tl.where(cross_mask, -r, 0)
        cross = tl.dot(inverse, cross, input_precision=PRECISION)

        inverse = tl.dot(cross, inverse, inverse, input_precision=PRECISION)

    tt = chunk * BT + rt

    token_head = (b * T + tt) * H + h
    vc = tl.load(
        v + token_head[:, None] * V + rv[None, :],
        mask=(tt[:, None] < T) & (rv[None, :] < V),
        other=0,
    ).to(tl.float32)
    bc = tl.load(beta + token_head, mask=tt < T, other=0).to(tl.float32)

    key_offsets = (
        tile * BT * K
        + rt[:, None] * K
        + rk[None, :]
    )
    pc = tl.load(P + key_offsets, mask=rk[None, :] < K, other=0)

    u = tl.dot(inverse, bc[:, None] * vc, input_precision=PRECISION)
    w = tl.dot(inverse, pc, input_precision=PRECISION)
    tl.store(U + tile * BT * V + rt[:, None] * V + rv[None, :], u, mask=rv[None, :] < V)
    tl.store(W + key_offsets, w, mask=rk[None, :] < K)


@triton.jit
def _carry_states(
        U, W, F, D, initial_state, incoming_states, E, final_state,
        K: tl.constexpr,
        V: tl.constexpr,
        NC: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
    bh, val_tile = tl.program_id(0), tl.program_id(1)
    rt, rk = tl.arange(0, BT), tl.arange(0, BK)
    rv = val_tile * BV + tl.arange(0, BV)

    state_offsets = (
        bh * K * V
        + rk[:, None] * V
        + rv[None, :]
    )
    state_mask = (rk[:, None] < K) & (rv[None, :] < V)
    state = tl.load(initial_state + state_offsets, mask=state_mask, other=0).to(tl.float32)


    for chunk in tl.range(0, NC):
        tile = bh * NC + chunk
        history_offsets = (
            tile * K * V
            + rk[:, None] * V
            + rv[None, :]
        )
        tl.store(incoming_states + history_offsets, state, mask=state_mask)

        key_offsets = (
            tile * BT * K
            + rt[:, None] * K
            + rk[None, :]
        )
        value_offsets = (
            tile * BT * V
            + rt[:, None] * V
            + rv[None, :]
        )
        w = tl.load(W + key_offsets, mask=rk[None, :] < K, other=0)
        f = tl.load(F + key_offsets, mask=rk[None, :] < K, other=0)

        u = tl.load(U + value_offsets, mask=rv[None, :] < V, other=0)
        e = u - tl.dot(w, state, input_precision=PRECISION)
        tl.store(E + value_offsets, e, mask=rv[None, :] < V)
        decay = tl.load(D + tile * K + rk, mask=rk < K, other=0)
        state = tl.dot(tl.trans(f), e, state * decay[:, None], input_precision=PRECISION)
    tl.store(final_state + state_offsets, state, mask=state_mask)


@triton.jit
def _produce_output(
        Qbar, A, incoming_states, E, output,
        T: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        NC: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
    tile, val_tile = tl.program_id(0), tl.program_id(1)
    bh, chunk = tile // NC, tile % NC
    b, h = bh // H, bh % H
    rt, rk = tl.arange(0, BT), tl.arange(0, BK)
    rv = val_tile * BV + tl.arange(0, BV)

    qbar = tl.load(
        Qbar + tile * BT * K + rt[:, None] * K + rk[None, :],
        mask=rk[None, :] < K,
        other=0,
    )
    a = tl.load(A + tile * BT * BT + rt[:, None] * BT + rt[None, :])
    state = tl.load(
        incoming_states + tile * K * V + rk[:, None] * V + rv[None, :],
        mask=(rk[:, None] < K) & (rv[None, :] < V),
        other=0,
    )
    e = tl.load(
        E + tile * BT * V + rt[:, None] * V + rv[None, :],
        mask=rv[None, :] < V,
        other=0,
    )

    out = tl.dot(qbar, state, input_precision=PRECISION)

    out = tl.dot(a, e, out, input_precision=PRECISION)

    tt = chunk * BT + rt

    token_head = (b * T + tt) * H + h
    tl.store(
        output + token_head[:, None] * V + rv[None, :],
        out,
        mask=(tt[:, None] < T) & (rv[None, :] < V),
    )


def chunk_kda(
        q, k, v, g, beta, state, scale, chunk_size=32, input_precision="tf32x3",
        output_dtype=None,
    ):

    if chunk_size not in (16, 32, 64):
        raise ValueError("chunk_size must be 16, 32, or 64")

    if input_precision not in ("ieee", "tf32x3"):
        raise ValueError("input_precision must be 'ieee' or 'tf32x3'")

    if dtypes.compute != "fp32":
        raise ValueError("chunk_kda currently requires FP32 computation")

    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k and v must have shape (B,T,H,K/V)")
    B, T, H, K = k.shape
    V = v.shape[-1]

    if (q.shape != k.shape or g.shape != k.shape or v.shape[:3] != k.shape[:3]
            or beta.shape != (B, T, H) or state.shape != (B, H, K, V)):
        raise ValueError("incompatible q/k/v/g/beta/state shapes")
    tensors = (q, k, v, g, beta, state)

    if not all(x.is_cuda and x.device == q.device for x in tensors):
        raise ValueError("all inputs must be on the same CUDA device")

    if not all(x.dtype in (torch.float16, torch.bfloat16, torch.float32) for x in tensors):
        raise ValueError("inputs must be FP16, BF16, or FP32")

    if torch.is_grad_enabled() and any(x.requires_grad for x in tensors):
        raise RuntimeError("chunk_kda has no backward yet; use torch.no_grad() for forward testing")

    if min(B, H, K, V) <= 0:
        raise ValueError("batch, heads, and key/value dimensions must be positive")

    output_dtype = output_dtype or torch_map[dtypes.output]
    q, k, v, g, beta, state = [x.contiguous() for x in tensors]
    output = torch.empty((B, T, H, V), device=v.device, dtype=output_dtype)
    final_state = torch.empty_like(state)

    if T == 0:
        return output, state.clone()

    NC = triton.cdiv(T, chunk_size)
    BK = max(16, triton.next_power_of_2(K))
    solve_BV = max(16, triton.next_power_of_2(V))
    BV = 32
    tiles = B * H * NC

    def scratch(*shape):
        return torch.empty(shape, device=v.device, dtype=torch.float32)

    G, P, Qbar, F, W = [scratch(B * H, NC, chunk_size, K) for _ in range(5)]
    R, A = [scratch(B * H, NC, chunk_size, chunk_size) for _ in range(2)]
    U, E = [scratch(B * H, NC, chunk_size, V) for _ in range(2)]
    D = scratch(B * H, NC, K)
    incoming_states = scratch(B * H, NC, K, V)


    _prepare_gates[(tiles,)](
        q, k, g, beta, G, P, Qbar, F, D, scale,
        T=T, H=H, K=K, NC=NC, BT=chunk_size, BK=BK,
    )
    _prepare_interactions[(tiles, chunk_size)](
        q, k, beta, G, R, A, scale,
        T=T, H=H, K=K, NC=NC, BT=chunk_size, BK=BK,
    )
    _prepare_solve[(tiles,)](
        v, beta, R, P, U, W,
        T=T, H=H, K=K, V=V, NC=NC, BT=chunk_size, BK=BK, BV=solve_BV,
        LOG_BT=chunk_size.bit_length() - 1, PRECISION=input_precision,
    )


    _carry_states[(B * H, triton.cdiv(V, BV))](
        U, W, F, D, state, incoming_states, E, final_state,
        K=K, V=V, NC=NC, BT=chunk_size, BK=BK, BV=BV, PRECISION=input_precision,
    )


    _produce_output[(tiles, triton.cdiv(V, BV))](
        Qbar, A, incoming_states, E, output,
        T=T, H=H, K=K, V=V, NC=NC, BT=chunk_size, BK=BK, BV=BV, PRECISION=input_precision,
    )

    return output, final_state
