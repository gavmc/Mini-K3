import triton
import triton.language as tl
import torch

from config import dtypes, tl_map, torch_map


def pt_recurrent_kda(q, k, v, g, beta, state, scale):
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


def pt_recurrent_kd_bkw(
        q, k, v, g, beta, state, scale,
        doutput, dfinal_state=None,
    ):

    B, T, H, V = v.shape
    compute = torch_map[dtypes.compute]

    q_c = q.to(compute)
    k_c = k.to(compute)
    v_c = v.to(compute)
    g_c = g.to(compute)
    beta_c = beta.to(compute)
    state_c = state.to(compute).clone()

    if dfinal_state is not None:
        dfinal_state_c = dfinal_state.to(compute).clone()
    else:
        dfinal_state_c = torch.zeros_like(state_c)

    doutput_c = doutput.to(compute)

    dq = torch.empty_like(q_c)
    dk = torch.empty_like(k_c)
    dv = torch.empty_like(v_c)
    dg = torch.empty_like(g_c)
    dbeta = torch.empty_like(beta_c)
    dinitial_state = torch.empty_like(state_c)

    for t in range(T):


        decay = g_c[:, t].exp().unsqueeze(-1)
        decayed_state = state_c * decay

        
        
        prediction = torch.einsum("bhkv,bhk->bhv", decayed_state, k_c[:, t])
        residual = beta[:, t].unsqueeze(-1) * (v_c[:, t] - prediction)
        updated_state = decayed_state + (k_c[:, t].unsqueeze(-1) * residual.unsqueeze(-2))
    
        dq[:, t] = scale * torch.einsum("bhkv,bhv->bhk", updated_state, doutput_c[:, t])

        d_updated_state = dfinal_state_c + ((scale * q).unsqueeze(-1) * doutput_c[:, t].unsqueeze(-2))
    
        dk_write = torch.einsum("bhkv,bhv->bhk", d_updated_state, residual)
        dresidual = torch.einsum("bhkv,bhk->bhv", d_updated_state, k_c[:, t])
    
        dbeta[:, t] = (dresidual * (v_c[:, t] - prediction)).sum(dim=-1)
        dv[:, t] = beta_c[:, t].unsqueeze(-1) * dresidual
        dprediction = -beta[:, t].unsqueeze(-1) * dresidual
    
        dk[:, t] = dk_write + torch.einsum("bhkv,bhv->bhk", decayed_state, dprediction)
        d_decayed_state = d_updated_state + (k.unsqueeze(-1) * dprediction.unsqueeze(-2))
    
        dinitial_state[:, t] = decay * d_decayed_state
        dg[:, t] = (d_decayed_state * decayed_state).sum(dim=-1)
    
    return dq, dk, dv, dg, dbeta, dinitial_state



@triton.jit
def recurrent_kda_kernel_bkw(
        q, k, v, g, beta, state, scale, doutput, dfinal_state,
        dq, dk, dv, dg, dbeta, dinitial_state,
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
        doutput_t = tl.load(doutput + value_offsets, mask = rv < V, other=0.0).to(COMPUTE)

        ''''
        dk_t = tl.load(dk + key_offsets, mask=rk < K, other=0.0).to(COMPUTE)
        dq_t = tl.load(dq + key_offsets, mask=rk < K, other=0.0).to(COMPUTE)
        dg_t = tl.load(dg + key_offsets, mask=rk < K, other=0.0).to(COMPUTE)
        dv_t = tl.load(dv + value_offsets, mask=rv < V, other=0.0).to(COMPUTE)
        dbeta_t = tl.load(dbeta + beta_offsets).to(COMPUTE)
        '''

        decay = tl.exp(g_t[:, None])
        decayed_state = state_t * decay

        prediction = tl.sum(decayed_state * k_t[:, None], axis=0)
        residual = beta_t * (v_t - prediction)
        updated_state = decayed_state + k_t[:, None] * residual[None, :]
        
        dq_t = scale * tl.sum(updated_state, doutput_t[None, :])
        d_updated_state = dfinal_state + ((scale * q_t)[:, None] * doutput[None, :])

        dk_write = tl.sum(d_updated_state * residual[None, :])
        dresidual = tl.sum(d_updated_state * k_t[:, None])

        dbeta_t = tl.sum(dresidual * (v_t - prediction))
        dv_t = beta_t[:, None] * dresidual
        dprediction = -1 * beta_t[:, None] * dresidual

        dk_t = dk_write + tl.sum(decayed_state * dprediction[None, :])
        d_decayed_state = d_updated_state + k_t[:, None] * dprediction[None, :]

        dinitial_state_t = decay * d_decayed_state
        dg_t = tl.sum(d_decayed_state * decayed_state)

        #tl.store(output + value_offsets, out_t, mask=rv < V)

        tl.store(dk + key_offsets, dk_t, mask=rk < K)
        tl.store(dq + key_offsets, dq_t, mask=rk < K)
        tl.store(dg + key_offsets, dg_t, mask=rk < K)
        tl.store(dv + value_offsets, dv_t, mask=rv < V)
        tl.store(dbeta + beta_offsets, dbeta_t)

        tl.store(dinitial_state + state_offsets, dinitial_state_t, mask=state_mask)

        key_offsets += H * K
        value_offsets += H * V
        beta_offsets += H


@triton.jit
def recurrent_kda_kernel(
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


def recurrent_kda(q, k, v, g, beta, state, scale):

    B, T, H, K = k.shape
    _, _, _, V = v.shape

    BK = triton.next_power_of_2(K)
    BV = 32 # needs to be tuned

    grid = (B * H * triton.cdiv(V, BV),)

    output = torch.empty((B, T, H, V), device=v.device, dtype=torch_map[dtypes.output])  
    new_state = torch.empty_like(state)

    recurrent_kda_kernel[grid](
        q, k, v, g, beta, state, new_state, scale, output, 
        BK=BK, BV=BV, K=K, V=V, H=H, T=T, 
        COMPUTE=tl_map[dtypes.compute]
    )     

    return output, new_state




