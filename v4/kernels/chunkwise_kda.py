import triton
import triton.language as tl
import torch


def pt_chunk_kda(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, state: torch.Tensor, scale):
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    g = g.permute(0, 2, 1, 3)
    beta = beta.permute(0, 2, 1)

    C = q.shape[2]

    G = g.cumsum(dim=2)
    boundary_decay = G.exp()

    boundary_keys = boundary_decay * k
    prediction = boundary_keys @ state

    E_bar = beta[..., None] * (v-prediction)

    token_ids = torch.arange(C, device=q.device)
    causal = token_ids[:, None] >= token_ids[None, :]

    log_pair_decay = G.unsqueeze(3) - G.unsqueeze(2)
    log_pair_decay = log_pair_decay.masked_fill(~causal[..., None], 0.0)
    pair_decay = log_pair_decay.exp()

    key_interactions = torch.einsum("bhijk,bhik,bhjk->bhij", pair_decay, k, k)

    R = torch.tril(beta[..., None] * key_interactions, diagonal=1)
    identity = torch.eye(C, device=q.device, dtype=q.dtype)

    E = torch.linalg.solve_triangular(
        identity + R,
        E_bar,
        upper=False,
        unitriangular=True,
    )

    end_log_decay = G[:, :, -1, :]

    keys_to_end = k * torch.exp(end_log_decay.unsqueeze(2) - G)
    next_state = state * end_log_decay.exp().unsqueeze(-1) + keys_to_end.transpose(-1, -2) @ E

    boundary_queries = boundary_decay * q
    boundary_output = scale * (boundary_queries @ state)

    query_key_interactions = torch.einsum("bhijk,bhik,bhjk->bhij", pair_decay, q, k)
    A_qk = scale * torch.tril(query_key_interactions, diagonal=0)

    output = boundary_output + A_qk @ E

    return output.permute(0, 2, 1, 3).contiguous(), next_state



@triton.jit
def chunk_kda_kernel(
        q, k, v, g, beta, state, scale, output, 
        BK: tl.constexpr,
        BV: tl.constexpr,
        BT: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
    ):

    pid = tl.program_id(0)

    nv = tl.cdiv(V, BV)
    val_tile = pid % nv

    bh = pid // nv
    b = bh // H
    h = bh % H

    rt = tl.arange(0, BT)
    rk = tl.arange(0, BK)
    rv = val_tile * BV + tl.arange(0, BV)

    state_offsets = (
        bh * K * V 
        + rk[:, None] * V
        + rv[None, :]
    )

    key_offsets = (
        b * T * H * K
        + rt[:, None] * H * K
        + h * K
        + rk[None, :]
    )   
    value_offsets = (
        b * T * H * V
        + rt[:, None] * H * V
        + h * V
        + rv[None, :]
    )   
    output_offsets = value_offsets
    beta_offsets = (
        b * T * H
        + rt * H
        + h
    )     

    state_mask = (rk[:, None] < K) & (rv[None, :] < V)

    state_t = tl.load(
        state + state_offsets,
        mask=state_mask,
        other=0.0,
    )  # [BK, BV]

    key_mask = (rt[:, None] < T) & (rk[None, :] < K)
    value_mask = (rt[:, None] < T) & (rv[None, :] < V)

    k_chunk = tl.load(k + key_offsets, mask=key_mask, other=0.0) # [BT, BK]
    q_chunk = tl.load(q + key_offsets, mask=key_mask, other=0.0) # [BT, BK]
    g_chunk = tl.load(g + key_offsets, mask=key_mask, other=0.0) # [BT, BK]

    v_chunk = tl.load(v + value_offsets, mask=value_mask, other=0.0) # [BT, BV]
    beta_chunk = tl.load(beta + beta_offsets, mask=rt < T, other=0.0) # [BT]

    G = tl.cumsum(g_chunk, axis=0)

    boundary_decay = tl.exp(G)
    boundary_keys = boundary_decay * k_chunk
    prediction = tl.dot(boundary_keys, state_t, input_precision="ieee")

    E_bar = beta_chunk[:, None] * (v_chunk-prediction)
    causal = (
        (rt[:, None] >= rt[None, :])
        & (rt[:, None] < T)
        & (rt[None, :] < T)
    )



def chunk_kda(q, k, v, g, beta, state, scale):

    B, T, H, K = k.shape
    _, _, _, V = v.shape

    BK = triton.next_power_of_2(K)
    BT = triton.next_power_of_2(T)
    BV = 32 # needs to be tuned

    grid = (B * H * triton.cdiv(V, BV),)

    output = torch.empty_like(v)    

    chunk_kda_kernel[grid](q, k, v, g, beta, state, scale, output, BK=BK, BV=BV, K=K, V=V)     

    return output, state 





