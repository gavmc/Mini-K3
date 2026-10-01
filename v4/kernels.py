import triton
import triton.language as tl
import torch


def pt_recurrent_kda(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, state: torch.Tensor, scale):
    state *= g.exp().unsqueeze(-1)
    prediction = torch.einsum("bhkv,bhk->bhv", state, k)
    residual = beta.unsqueeze(-1) * (v - prediction)
    state += torch.einsum("bhk,bhv->bhkv", k, residual)
    output = torch.einsum("bhk,bhkv->bhv", q * scale, state)
    return output, state


@triton.jit
def recurrent_kda_kernel(
        q, k, v, g, beta, state, scale, output, 
        BK: tl.constexpr,
        BV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
    ):

    pid = tl.program_id(0)

    nv = tl.cdiv(V, BV)
    val_tile = pid % nv
    bh = pid // nv

    rk = tl.arange(0, BK)
    rv = val_tile * BV + tl.arange(0, BV)

    state_offsets = (
        bh * K * V 
        + rk[:, None] * V
        + rv[None, :]
    )

    key_offsets = bh * K + rk    
    value_offsets = bh * V + rv  
    output_offsets = bh * V + rv 
    beta_offset = bh             

    state_mask = (rk[:, None] < K) & (rv[None, :] < V)

    state_t = tl.load(
        state + state_offsets,
        mask=state_mask,
        other=0.0,
    )

    k_t = tl.load(k + key_offsets, mask=rk < K, other=0.0)
    q_t = tl.load(q + key_offsets, mask=rk < K, other=0.0)
    g_t = tl.load(g + key_offsets, mask=rk < K, other=0.0)
    v_t = tl.load(v + value_offsets, mask=rv < V, other=0.0)
    beta_t = tl.load(beta + beta_offset)


    state_t *= tl.exp(g_t[:, None])
    prediction = tl.sum(state_t * k_t[:, None], axis=0)
    residual = beta_t * (v_t - prediction)
    state_t += k_t[:, None] * residual[None, :]
    out_t = tl.sum(state_t * (q_t * scale)[:, None], axis=0)

    tl.store(output + output_offsets, out_t, mask=rv < V)
    tl.store(state + state_offsets, state_t, mask=state_mask)



def recurrent_kda(q, k, v, g, beta, state, scale):

    B, H, K = k.shape
    _, _, V = v.shape

    BK = triton.next_power_of_2(K)
    BV = 32 # needs to be tuned

    grid = (B * H * triton.cdiv(V, BV),)

    output = torch.empty_like(v)    

    recurrent_kda_kernel[grid](q, k, v, g, beta, state, scale, output, BK=BK, BV=BV, K=K, V=V)     

    return output, state     




B, H, dk, dv = 1, 8, 64, 128


q = torch.rand(B, H, dk, device="cuda")
k = torch.rand(B, H, dk, device="cuda")
k = torch.nn.functional.normalize(k, dim=-1)
v = torch.rand(B, H, dv, device="cuda")
g = -torch.rand(B, H, dk, device="cuda")
beta = torch.rand(B, H, device="cuda")
state_1 = torch.rand(B, H, dk, dv, device="cuda")
state_2 = state_1.clone()
scale = dk ** -0.5



bench_state_1 = state_1.clone()
bench_state_2 = state_2.clone()

recurrent_kda(q, k, v, g, beta, bench_state_1, scale)
pt_recurrent_kda(q, k, v, g, beta, bench_state_2, scale)
torch.cuda.synchronize()

triton_ms = triton.testing.do_bench(
    lambda: recurrent_kda(q, k, v, g, beta, bench_state_1, scale),
    return_mode="median",
)

pytorch_ms = triton.testing.do_bench(
    lambda: pt_recurrent_kda(q, k, v, g, beta, bench_state_2, scale),
    return_mode="median",
)

print(f"Triton:  {triton_ms:.6f} ms")
print(f"PyTorch: {pytorch_ms:.6f} ms")
print(f"Speedup: {pytorch_ms / triton_ms:.2f}x")