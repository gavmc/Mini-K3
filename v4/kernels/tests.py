import torch

from kernels.recurrent_kda import pt_recurrent_kd_bkw, recurrent_kda_bkw

B, T, H, dk, dv = 1, 256, 8, 32, 64

torch.manual_seed(0)
# Use FP32 inputs to compare gradients without BF16 storage rounding.
q = torch.rand(B, T, H, dk, device="cuda", dtype=torch.float32).requires_grad_()
k = torch.rand(B, T, H, dk, device="cuda", dtype=torch.float32)
k = torch.nn.functional.normalize(k, dim=-1)
k.requires_grad_()
v = torch.rand(B, T, H, dv, device="cuda", dtype=torch.float32).requires_grad_()
g = (-torch.rand(B, T, H, dk, device="cuda", dtype=torch.float32)).requires_grad_()
beta = torch.rand(B, T, H, device="cuda", dtype=torch.float32).requires_grad_()
initial_state = torch.rand(B, H, dk, dv, device="cuda", dtype=torch.float32).requires_grad_()
scale = dk ** -0.5
doutput = torch.randn_like(v)

# Build real forward history and an autograd graph for the same recurrence.
state = initial_state
history = [state]
outputs = []

for t in range(T):
    decayed_state = state * g[:, t].exp().unsqueeze(-1)
    prediction = torch.einsum("bhkv,bhk->bhv", decayed_state, k[:, t])
    residual = beta[:, t].unsqueeze(-1) * (v[:, t] - prediction)
    state = decayed_state + torch.einsum("bhk,bhv->bhkv", k[:, t], residual)
    history.append(state)
    outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, state))

output = torch.stack(outputs, dim=1)
states = torch.stack(history, dim=1).detach().contiguous()
inputs = (q, k, v, g, beta, initial_state)
names = ("dq", "dk", "dv", "dg", "dbeta", "dinitial_state")

for use_final_gradient in (False, True):
    dfinal_state = torch.randn_like(initial_state) if use_final_gradient else None
    reference = torch.autograd.grad(
        (output, state),
        inputs,
        grad_outputs=(doutput, torch.zeros_like(initial_state) if dfinal_state is None else dfinal_state),
        retain_graph=not use_final_gradient,
    )
    with torch.no_grad():
        manual = pt_recurrent_kd_bkw(q, k, v, g, beta, states, scale, doutput, dfinal_state)
        actual = recurrent_kda_bkw(q, k, v, g, beta, states, scale, doutput, dfinal_state)

    print("\nFinal-state gradient:", "nonzero" if use_final_gradient else "zero")

    for name, pt_grad, triton_grad, ref_grad in zip(names, manual, actual, reference):
        print(f"{name} max diff: {(triton_grad - ref_grad).abs().max().item()}")
        torch.testing.assert_close(pt_grad, ref_grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(triton_grad, ref_grad, rtol=1e-4, atol=1e-4)

print("\nAll backward gradients match autograd.")
