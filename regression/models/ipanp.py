import torch
import torch.nn as nn
from attrdict import AttrDict
from torch.distributions import kl_divergence

from regression.models.anp import ANP
from regression.models.modules import CrossAttnSpinEncoder, PoolingSpinEncoder, Decoder
from regression.utils.misc import stack, logmeanexp


class IPANP(ANP):
    def __init__(self,
                 use_latent_path=True,
                 dim_x=1,
                 dim_y=1,
                 dim_hid=128,
                 dim_lat=128,
                 enc_v_depth=4,
                 enc_qk_depth=2,
                 enc_pre_depth=4,
                 enc_post_depth=2,
                 dec_depth=3,
                 use_H_A=False,
                 use_ABLA_induce=True,
                 H_A_dim=64,
                 latent_dim_mult=1,
                 num_induce=16,
                 H_D_init='xavier',
                 num_heads=8,
                 num_spin_blocks=8,
                 ):

        super().__init__()
        self.use_latent_path = use_latent_path

        self.denc = CrossAttnSpinEncoder(
            dim_x=dim_x,
            dim_y=dim_y,
            dim_hid=dim_hid * (1 if use_latent_path else 2),
            v_depth=enc_v_depth,
            qk_depth=enc_qk_depth,
            use_H_A=use_H_A,
            use_ABLA_induce=use_ABLA_induce,
            H_A_dim=H_A_dim,
            latent_dim_mult=latent_dim_mult,
            num_induce=num_induce,
            H_D_init=H_D_init,
            num_heads=num_heads,
            num_spin_blocks=num_spin_blocks,
        )

        if use_latent_path:
            self.lenc = PoolingSpinEncoder(
                dim_x=dim_x, dim_y=dim_y,
                dim_hid=dim_hid,
                dim_lat=dim_lat,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth,
                use_H_A=use_H_A,
                use_ABLA_induce=use_ABLA_induce,
                H_A_dim=H_A_dim,
                latent_dim_mult=latent_dim_mult,
                num_induce=num_induce,
                H_D_init='xavier',
                num_heads=num_heads,
                num_spin_blocks=num_spin_blocks
            )

        self.dec = Decoder(
            dim_x=dim_x,
            dim_y=dim_y,
            dim_enc=dim_hid + (dim_lat if self.use_latent_path else dim_hid),
            dim_hid=dim_hid,
            depth=dec_depth)

    # def predict(self, xc, yc, xt, z=None, num_samples=None):
    #     theta = stack(self.denc(xc, yc, xt), num_samples)
    #     if self.use_latent_path:
    #         if z is None:
    #             pz = self.lenc(xc, yc)
    #             z = pz.rsample() if num_samples is None \
    #                 else pz.rsample([num_samples])
    #         z = stack(z, xt.shape[-2], -2)
    #         encoded = torch.cat([theta, z], -1)
    #     else:
    #         encoded = theta
    #     return self.dec(encoded, stack(xt, num_samples))
    #
    # def sample(self, xc, yc, xt, z=None, num_samples=None):
    #     pred_dist = self.predict(xc, yc, xt, z, num_samples)
    #     return pred_dist.loc
    #
    # def forward(self, batch, num_samples=None):
    #     outs = AttrDict()
    #     if self.training:
    #         if self.use_latent_path:
    #             pz = self.lenc(batch.xc, batch.yc)
    #             qz = self.lenc(batch.x, batch.y)
    #             z = qz.rsample() if num_samples is None else qz.rsample([num_samples])
    #         else:
    #             z = None
    #         py = self.predict(batch.xc, batch.yc, batch.x,
    #                           z=z, num_samples=num_samples)
    #
    #         if num_samples is not None and num_samples > 1:
    #             if self.use_latent_path:
    #                 # K * B * N
    #                 recon = py.log_prob(stack(batch.y, num_samples)).sum(-1)
    #                 # K * B
    #                 log_qz = qz.log_prob(z).sum(-1)
    #                 log_pz = pz.log_prob(z).sum(-1)
    #
    #                 # K * B
    #                 log_w = recon.sum(-1) + log_pz - log_qz
    #                 outs.loss = -logmeanexp(log_w).mean() / batch.x.shape[-2]
    #             else:
    #                 outs.recon = py.log_prob(stack(batch.y, num_samples)).sum(-1).mean()
    #                 outs.loss = -outs.recon
    #         else:
    #             outs.recon = py.log_prob(batch.y).sum(-1).mean()
    #             if self.use_latent_path:
    #                 outs.kld = kl_divergence(qz, pz).sum(-1).mean()
    #                 outs.loss = -outs.recon + outs.kld / batch.x.shape[-2]
    #             else:
    #                 outs.loss = -outs.recon
    #
    #     else:
    #         py = self.predict(batch.xc, batch.yc, batch.x, num_samples=num_samples)
    #         if num_samples is None:
    #             ll = py.log_prob(batch.y).sum(-1)
    #         else:
    #             y = torch.stack([batch.y] * num_samples)
    #             ll = logmeanexp(py.log_prob(y).sum(-1))
    #         num_ctx = batch.xc.shape[-2]
    #
    #         outs.ctx_ll = ll[..., :num_ctx].mean()
    #         outs.tar_ll = ll[..., num_ctx:].mean()
    #
    #     return outs
