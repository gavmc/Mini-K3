import torch
from torch import nn
import torch.nn.functional as F
from fla.ops.kda import fused_recurrent_kda, chunk_kda
from fla.modules.conv import ShortConvolution
import math


class KDA(nn.Module): 
    def __init__(self, d_model, dk, dv, n_head, g_min=-5):
        super().__init__()

        self.dk = dk
        self.dv = dv
        self.n_head = n_head
        self.g_min = g_min

        self.q_proj = nn.Linear(d_model, dk*n_head, bias=False)
        self.k_proj = nn.Linear(d_model, dk*n_head, bias=False)
        self.v_proj = nn.Linear(d_model, dv*n_head, bias=False)

        self.q_conv = ShortConvolution(dk*n_head, kernel_size=4)
        self.k_conv = ShortConvolution(dk*n_head, kernel_size=4)
        self.v_conv = ShortConvolution(dv*n_head, kernel_size=4)

        self.b_proj = nn.Linear(d_model, n_head, bias=False)
        self.a_proj = nn.Sequential(
            nn.Linear(d_model, dk, bias=False),
            nn.Linear(dk, dk*n_head, bias=False)
        )

        self.a_log = nn.Parameter(torch.zeros(n_head)) 

        dt = torch.exp(torch.rand(n_head * dk, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))

        inv_dt = dt + torch.log(-torch.expm1(-dt))

        self.dt_bias = nn.Parameter(inv_dt)

        self.a_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True

        self.o_norm = nn.RMSNorm(dv)
        self.g_proj = nn.Linear(d_model, dv*n_head, bias=False)
        self.o_proj = nn.Linear(dv*n_head, d_model, bias=False)


    def forward(
        self, x,
        initial_state=None,
        conv_state=None,
        use_cache=False,
    ):
        
        B, T, _ = x.shape

        if conv_state is None:
            conv_state = (None, None, None)

        q, q_state = self.q_conv(self.q_proj(x), cache=conv_state[0], output_final_state=use_cache)
        k, k_state = self.k_conv(self.k_proj(x), cache=conv_state[1], output_final_state=use_cache)
        v, v_state = self.v_conv(self.v_proj(x), cache=conv_state[2], output_final_state=use_cache)

        q = q.reshape(B, T, self.n_head, self.dk)
        k = k.reshape(B, T, self.n_head, self.dk)
        v = v.reshape(B, T, self.n_head, self.dv)

        beta = self.b_proj(x).sigmoid()

        z = self.a_proj(x).reshape(B, T, self.n_head, self.dk)

        gate_input = z.float() + self.dt_bias.float().reshape(1, 1, self.n_head, self.dk)
        g = self.g_min * torch.sigmoid(self.a_log.float().exp()[None, None, :, None] * gate_input)

        decode = not self.training and not torch.is_grad_enabled() and T == 1

        fn = fused_recurrent_kda if decode else chunk_kda

        output, final_state = fn(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
        )

        output_gate = self.g_proj(x).reshape(B, T, self.n_head, self.dv)
        output = self.o_norm(output) * output_gate.sigmoid()

        output = output.reshape(B, T, self.n_head * self.dv)
        output = self.o_proj(output)

        final_conv_state = (q_state, k_state, v_state) if use_cache else None

        return output, final_state, final_conv_state


class MLA(nn.Module):
    def __init__(self, d_model, d_c, n_head, d_head):
        super().__init__()

        self.d_model = d_model
        self.d_c = d_c
        self.d_head = d_head
        self.n_head = n_head
        self.width = n_head * d_head

        self.q_proj = nn.Linear(d_model, self.width, bias=False)

        self.c_proj = nn.Linear(d_model, d_c, bias=False)
        self.c_norm = nn.RMSNorm(d_c)

        self.k_proj = nn.Linear(d_c, self.width, bias=False)
        self.v_proj = nn.Linear(d_c, self.width, bias=False)

        self.g_proj = nn.Linear(d_model, self.width, bias=False)
        self.o_proj = nn.Linear(self.width, d_model, bias=False)

    def forward(self, x, latent_cache=None, use_cache=False):
        B, T, _ = x.shape
        H, D = self.n_head, self.d_head

        c_new = self.c_norm(self.c_proj(x))

        past_len = 0

        if latent_cache is not None:
            past_len = latent_cache.shape[1]
            c_all = torch.cat((latent_cache, c_new), dim=1)
        else:
            c_all = c_new

        total_len = c_all.shape[1]

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)

        k = self.k_proj(c_all).reshape(B, total_len, H, D).transpose(1, 2)

        v = self.v_proj(c_all).reshape(B, total_len, H, D).transpose(1, 2)

        if past_len == 0:
            output = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=0.0,
            )

        elif T == 1:
            output = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=False,
                dropout_p=0.0,
            )

        else:
            query_positions = (past_len + torch.arange(T, device=x.device))
            key_positions = torch.arange(total_len, device=x.device)

            causal_mask = (key_positions[None, :] <= query_positions[:, None])

            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=causal_mask,
                is_causal=False,
                dropout_p=0.0,
            )

        output = output.transpose(1, 2).reshape(B, T, self.width)

        output_gate = self.g_proj(x).sigmoid()
        output = self.o_proj(output * output_gate)

        final_cache = c_all if use_cache else None

        return output, final_cache


class MoE(nn.Module):
    def __init__(self):
        super().__init__()



