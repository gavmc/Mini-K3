import triton
import triton.language as tl
import torch

from config import dtypes, tl_map, torch_map


def pt_recurrent_kda_fwd(q, k, v, g, beta, state, scale):
    B, T, H, V = v.shape
    compute = torch_map[dtypes.compute]

    q_c = q.to(compute)
    k_c = k.to(compute)
    v_c = v.to(compute)
    g_c = g.to(compute)
    beta_c = beta.to(compute)
    state_c = state.to(compute).clone()

    output = torch.empty(
        (B, T, H, V),
        device=v.device,
        dtype=torch_map[dtypes.output],
    )

    for t in range(T):
        decayed_state = state_c * g_c[:, t].exp().unsqueeze(-1)

        prediction = torch.einsum("bhkv,bhk->bhv", decayed_state, k_c[:, t])

        residual = beta_c[:, t].unsqueeze(-1) * (v_c[:, t] - prediction)

        state_c = decayed_state + torch.einsum("bhk,bhv->bhkv", k_c[:, t], residual)

        output[:, t] = torch.einsum("bhk,bhkv->bhv", q_c[:, t] * scale, state_c)

    return output, state_c.to(state.dtype)

def pt_recurrent_kd_bkw(q, k, v, g, beta, states, scale, doutput, dfinal_state=None):
    compute = torch_map[dtypes.compute]

    q_c = q.to(compute)
    k_c = k.to(compute)
    v_c = v.to(compute)
    g_c = g.to(compute)
    beta_c = beta.to(compute)
    states_c = states.to(compute)
    doutput_c = doutput.to(compute)

    T = v.shape[1]

    if dfinal_state is None:
        dstate = torch.zeros_like(states_c[:, 0])
    else:
        dstate = dfinal_state.to(compute).clone()

    dq = torch.empty_like(q_c)
    dk = torch.empty_like(k_c)
    dv = torch.empty_like(v_c)
    dg = torch.empty_like(g_c)
    dbeta = torch.empty_like(beta_c)


    for t in reversed(range(T)):
        decay = g_c[:, t].exp().unsqueeze(-1)
        decayed_state = states_c[:, t] * decay
        prediction = torch.einsum("bhkv,bhk->bhv", decayed_state, k_c[:, t])
        residual = beta_c[:, t].unsqueeze(-1) * (v_c[:, t] - prediction)
        updated_state = states_c[:, t + 1]

        dq[:, t] = scale * torch.einsum("bhkv,bhv->bhk", updated_state, doutput_c[:, t])
        d_updated_state = dstate + (scale * q_c[:, t]).unsqueeze(-1) * doutput_c[:, t].unsqueeze(-2)

        dk_write = torch.einsum("bhkv,bhv->bhk", d_updated_state, residual)
        dresidual = torch.einsum("bhkv,bhk->bhv", d_updated_state, k_c[:, t])

        dbeta[:, t] = (dresidual * (v_c[:, t] - prediction)).sum(dim=-1)
        dv[:, t] = beta_c[:, t].unsqueeze(-1) * dresidual
        dprediction = -beta_c[:, t].unsqueeze(-1) * dresidual

        dk[:, t] = dk_write + torch.einsum("bhkv,bhv->bhk", decayed_state, dprediction)
        d_decayed_state = d_updated_state + k_c[:, t].unsqueeze(-1) * dprediction.unsqueeze(-2)

        dg[:, t] = (d_decayed_state * decayed_state).sum(dim=-1)
        dstate = decay * d_decayed_state

    return dq, dk, dv, dg, dbeta, dstate


@triton.jit
def recurrent_kda_kernel_bkw(
        q, k, v, g, beta, states, scale, doutput, dfinal_state,
        dq, dk, dv, dg, dbeta, dinitial_state,
        BK: tl.constexpr,
        BV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        H: tl.constexpr,
        T: tl.constexpr,
        COMPUTE: tl.constexpr,
    ):


    bh = tl.program_id(0)
    batch = bh // H
    head = bh % H

    rk = tl.arange(0, BK)
    rv = tl.arange(0, BV)

    tile_offsets = rk[:, None] * V + rv[None, :]
    state_offsets = bh * K * V + tile_offsets
    state_mask = (rk[:, None] < K) & (rv[None, :] < V)

    dstate_t = tl.load(dfinal_state + state_offsets, mask=state_mask, other=0.0).to(COMPUTE)

    for step in tl.range(0, T):

        t = T - 1 - step
        token_head = (batch * T + t) * H + head
        key_offsets = token_head * K + rk
        value_offsets = token_head * V + rv
        beta_offset = token_head
        history_offsets = ((batch * (T + 1) + t) * H + head) * K * V + tile_offsets

        state_t = tl.load(states + history_offsets, mask=state_mask, other=0.0).to(COMPUTE)
        updated_state = tl.load(states + history_offsets + H * K * V, mask=state_mask, other=0.0).to(COMPUTE)

        k_t = tl.load(k + key_offsets, mask=rk < K, other=0.0).to(COMPUTE)
        q_t = tl.load(q + key_offsets, mask=rk < K, other=0.0).to(COMPUTE)
        g_t = tl.load(g + key_offsets, mask=rk < K, other=0.0).to(COMPUTE)
        v_t = tl.load(v + value_offsets, mask=rv < V, other=0.0).to(COMPUTE)
        beta_t = tl.load(beta + beta_offset).to(COMPUTE)
        doutput_t = tl.load(doutput + value_offsets, mask = rv < V, other=0.0).to(COMPUTE)

        decay = tl.exp(g_t[:, None])
        decayed_state = state_t * decay

        prediction = tl.sum(decayed_state * k_t[:, None], axis=0)
        residual = beta_t * (v_t - prediction)

        dq_t = scale * tl.sum(updated_state * doutput_t[None, :], axis=1)
        d_updated_state = dstate_t + (scale * q_t)[:, None] * doutput_t[None, :]

        dk_write = tl.sum(d_updated_state * residual[None, :], axis=1)
        dresidual = tl.sum(d_updated_state * k_t[:, None], axis=0)

        dbeta_t = tl.sum(dresidual * (v_t - prediction), axis=0)
        dv_t = beta_t * dresidual
        dprediction = -beta_t * dresidual

        dk_t = dk_write + tl.sum(decayed_state * dprediction[None, :], axis=1)
        d_decayed_state = d_updated_state + k_t[:, None] * dprediction[None, :]

        dstate_t = decay * d_decayed_state
        dg_t = tl.sum(d_decayed_state * decayed_state, axis=1)

        tl.store(dk + key_offsets, dk_t, mask=rk < K)
        tl.store(dq + key_offsets, dq_t, mask=rk < K)
        tl.store(dg + key_offsets, dg_t, mask=rk < K)
        tl.store(dv + value_offsets, dv_t, mask=rv < V)
        tl.store(dbeta + beta_offset, dbeta_t)

        tl.store(dinitial_state + state_offsets, dstate_t, mask=state_mask)

@triton.jit
def recurrent_kda_kernel_fwd(
        q, k, v, g, beta, state, new_state, scale, output,
        BK: tl.constexpr,
        BV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        H: tl.constexpr,
        T: tl.constexpr,
        COMPUTE: tl.constexpr,
    ):

    pid = tl.program_id(0)

    nv = tl.cdiv(V, BV)
    val_tile = pid % nv
    bh = pid // nv

    batch = bh // H
    head = bh % H

    rk = tl.arange(0, BK)
    rv = val_tile * BV + tl.arange(0, BV)

    token_head = batch * T * H + head

    state_offsets = (
        bh * K * V
        + rk[:, None] * V
        + rv[None, :]
    )

    key_offsets = token_head * K + rk
    value_offsets = token_head * V + rv
    beta_offsets = token_head

    state_mask = (rk[:, None] < K) & (rv[None, :] < V)

    state_t = tl.load(
        state + state_offsets,
        mask=state_mask,
        other=0.0,
    ).to(COMPUTE)

    for _ in tl.range(0, T):
        k_t = tl.load(k + key_offsets, mask=rk < K, other=0.0).to(COMPUTE)
        q_t = tl.load(q + key_offsets, mask=rk < K, other=0.0).to(COMPUTE)
        g_t = tl.load(g + key_offsets, mask=rk < K, other=0.0).to(COMPUTE)
        v_t = tl.load(v + value_offsets, mask=rv < V, other=0.0).to(COMPUTE)
        beta_t = tl.load(beta + beta_offsets).to(COMPUTE)

        state_t *= tl.exp(g_t[:, None])
        prediction = tl.sum(state_t * k_t[:, None], axis=0)
        residual = beta_t * (v_t - prediction)
        state_t += k_t[:, None] * residual[None, :]
        out_t = tl.sum(state_t * (q_t * scale)[:, None], axis=0)

        tl.store(output + value_offsets, out_t, mask=rv < V)

        key_offsets += H * K
        value_offsets += H * V
        beta_offsets += H

    tl.store(new_state + state_offsets, state_t, mask=state_mask)


def recurrent_kda_bkw(q, k, v, g, beta, states, scale, doutput, dfinal_state=None):
    compute = torch_map[dtypes.compute]

    B, T, H, K = k.shape
    V = v.shape[-1]

    if dfinal_state is None:
        dstate = torch.zeros_like(states[:, 0])
    else:
        dstate = dfinal_state.to(compute).clone()

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    dbeta = torch.empty_like(beta)
    dinitial_state = torch.empty_like(states[:, 0])

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)

    grid = (B * H,)

    recurrent_kda_kernel_bkw[grid](
        q, k, v, g, beta, states, scale, doutput, dstate,
        dq, dk, dv, dg, dbeta, dinitial_state,
        BK=BK, BV=BV, K=K, V=V, H=H, T=T,
        COMPUTE=tl_map[dtypes.compute]
    )

    return dq, dk, dv, dg, dbeta, dinitial_state

def recurrent_kda_fwd(q, k, v, g, beta, state, scale):

    B, T, H, K = k.shape
    _, _, _, V = v.shape

    BK = triton.next_power_of_2(K)
    BV = 32 #  <----------------  needs to be tuned

    grid = (B * H * triton.cdiv(V, BV),)

    output = torch.empty((B, T, H, V), device=v.device, dtype=torch_map[dtypes.output])
    new_state = torch.empty_like(state)

    recurrent_kda_kernel_fwd[grid](
        q, k, v, g, beta, state, new_state, scale, output,
        BK=BK, BV=BV, K=K, V=V, H=H, T=T,
        COMPUTE=tl_map[dtypes.compute]
    )

    return output, new_state
