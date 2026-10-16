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


