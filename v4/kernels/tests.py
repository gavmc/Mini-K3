import torch

from  kernels.recurrent_kda import pt_recurrent_kd_bkw
from config import dtypes, torch_map

B, T, H, dk, dv = 1, 256, 8, 32, 64


q = torch.rand(B, T, H, dk, device="cuda", dtype=torch_map[dtypes.q])
k = torch.rand(B, T, H, dk, device="cuda", dtype=torch_map[dtypes.k])
k = torch.nn.functional.normalize(k, dim=-1)
v = torch.rand(B, T, H, dv, device="cuda" , dtype=torch_map[dtypes.v])
g = -torch.rand(B, T, H, dk, device="cuda", dtype=torch_map[dtypes.g])
beta = torch.rand(B, T, H, device="cuda", dtype=torch_map[dtypes.beta])
state = torch.rand(B, H, dk, dv, device="cuda", dtype=torch_map[dtypes.state])
scale = dk ** -0.5

doutput = torch.ones(v.shape, device=v.device, dtype=torch_map[dtypes.output])  


dq_k = torch.empty_like(q)
dk_k = torch.empty_like(k)
dv_k = torch.empty_like(v)
dg_k = torch.empty_like(g)
dbeta_k = torch.empty_like(beta)
dinitial_state = torch.empty_like(state)


dq, dk, dv, dg, dbeta, dinitial_state = pt_recurrent_kd_bkw(
    q, k, v, g, beta, state, scale, doutput, None
)

print(dq.shape)
print(dk.shape)
print(dv.shape)
print(dg.shape)
print(dbeta.shape)
print(dinitial_state.shape)



