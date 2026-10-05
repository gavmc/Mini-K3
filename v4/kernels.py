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
def recurrent_kda_kernel(q, k, v, g, beta, state, scale, BK, BV):
    pid = tl.program_id(0)   

    rk = tl.arange(0, BK)
    rv = pid * BV + tl.arange(0, BV)



def recurrent_kda(q, k, v, g, beta, state, scale):

    B, H, K = k.shape
    _, _, V = v.shape

    BK = triton.next_power_of_2(K)
    BV = 4 # <- needs to be tested (default test value)




B, H, dk, dv = 1, 8, 32, 64


q = torch.rand(B, H, dk)
k = torch.nn.functional.normalize(k, dim=-1)
v = torch.rand(B, H, dv)
g = -torch.rand(B, H, dk)
beta = torch.rand(B, H)
state = torch.rand(B, H, dk, dv)
scale = 1 / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

out, state = pt_recurrent_kda(q, k, v, g, beta, state, scale)

print(out)

print(out.shape)
print(state.shape)