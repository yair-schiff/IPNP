import torch
import torch.nn as nn

from regression.models.attention import MultiHeadAttn


class XABA(nn.Module):
    def __init__(self, data_dim, latent_dim, latent_dim_mult, num_heads):
        super().__init__()
        self.attr_xattn = MultiHeadAttn(dim_q=latent_dim, dim_k=data_dim, dim_v=data_dim,
                                        dim_out=latent_dim*latent_dim_mult,
                                        num_heads=num_heads)

    def forward(self, x, H_A, mask=None):
        return self.attr_xattn(q=H_A, k=x, v=x, mask=mask, permute_dims=True)


class ABLA(nn.Module):
    def __init__(self, latent_dim, latent_dim_mult, num_heads=8):
        super().__init__()
        self.attr_attn = MultiHeadAttn(dim_q=latent_dim, dim_k=latent_dim, dim_v=latent_dim,
                                       dim_out=latent_dim*latent_dim_mult,
                                       num_heads=num_heads)

    def forward(self, H):
        return self.attr_attn(q=H, k=H, v=H, mask=None, permute_dims=True)


class XABD(nn.Module):
    def __init__(self, latent_dim, latent_dim_mult, num_heads=8):
        super().__init__()
        self.data_xattn = MultiHeadAttn(dim_q=latent_dim, dim_k=latent_dim, dim_v=latent_dim,
                                        dim_out=latent_dim*latent_dim_mult,
                                        num_heads=num_heads)

    def forward(self, H_A, H_D, mask=None):
        return self.data_xattn(q=H_D, k=H_A, v=H_A, mask=mask, permute_dims=False)  # bsz x num_induce x latent_dim


class SpinBlock(nn.Module):
    def __init__(self,
                 use_H_A,
                 use_ABLA_induce,
                 data_dim, latent_dim, latent_dim_mult,
                 num_heads
                 ):
        super().__init__()

        self.use_H_A = use_H_A
        if use_H_A:
            # Cross Attn Between Attributes
            self.xaba = XABA(data_dim=data_dim, latent_dim=latent_dim, latent_dim_mult=latent_dim_mult,
                             num_heads=num_heads)

            # (Self) Attn Between Latent Attributes - attribute latents
            self.abla = ABLA(latent_dim=latent_dim, latent_dim_mult=latent_dim_mult, num_heads=num_heads)

        # Cross Attn Between Datapoints
        self.xabd = XABD(latent_dim=latent_dim, latent_dim_mult=latent_dim_mult, num_heads=num_heads)

        # (Self) Attn Between Latent Attributes - induced latents
        self.use_ABLA_induce = use_ABLA_induce
        if use_ABLA_induce:
            self.abla_induce = ABLA(latent_dim=latent_dim, latent_dim_mult=latent_dim_mult, num_heads=num_heads)

    def forward(self, x, H_A, H_D, mask=None):
        if self.use_H_A:
            H_A_prime = self.xaba(x=x, H_A=H_A, mask=mask)
            H_A = self.abla(H=H_A_prime)
        H_D = self.xabd(H_A=H_A if self.use_H_A else x, H_D=H_D, mask=mask)
        if self.use_ABLA_induce:
            H_D = self.abla_induce(H=H_D)
        return H_A, H_D


class Spin(nn.Module):
    def __init__(self,
                 data_dim=256,
                 latent_dim_mult=1,
                 use_H_A=False,
                 use_ABLA_induce=True,
                 H_A_dim=1,
                 H_D_init='xavier',
                 num_induce=16,
                 num_heads=8,
                 num_spin_blocks=1):
        super().__init__()
        self.use_H_A = use_H_A
        if use_H_A:
            self.H_A_proj = nn.Linear(data_dim, H_A_dim)
        self.H_D = self.init_h_d(num_inds=num_induce, latent_dim=H_A_dim if use_H_A else data_dim, init=H_D_init)
        self.spin_blocks = nn.ModuleList([
            SpinBlock(
                use_H_A=use_H_A,
                use_ABLA_induce=use_ABLA_induce,
                data_dim=data_dim, latent_dim=H_A_dim if use_H_A else data_dim, latent_dim_mult=latent_dim_mult,
                num_heads=num_heads,
            )
            for _ in range(num_spin_blocks)
        ])

    @staticmethod
    def init_h_d(num_inds, latent_dim, init='xavier'):
        induced = nn.Parameter(torch.Tensor(1, num_inds, latent_dim))
        if init == 'xavier':
            nn.init.xavier_uniform_(induced)
        elif init == 'orthogonal':
            nn.init.orthogonal_(induced)
        else:
            raise ValueError('Invalid H_D init method passed. Use either \'xavier\' (uniform) or \'orthogonal\'.')
        return induced

    def forward(self, context, mask=None):
        H_A = self.H_A_proj(context) if self.use_H_A else None
        H_D = self.H_D.repeat(context.shape[0], 1, 1)
        for sb in self.spin_blocks:
            H_A, H_D = sb(x=context, H_A=H_A, H_D=H_D, mask=mask)
        return H_D
