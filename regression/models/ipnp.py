import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

from regression.models.attention import MultiHeadAttn
from regression.models.modules import build_mlp, PoolingEncoder
from regression.models.spin import Spin
from regression.utils.misc import forward_plot_func


class IPNP(nn.Module):
    def __init__(self,
                 dim_x=1,
                 dim_y=1,
                 use_latent_path=False,
                 use_spin_latent=True,
                 data_emb_dim=256,
                 data_emb_depth=2,
                 use_H_A=False,
                 H_A_dim=64,
                 latent_dim_mult=1,
                 num_induce=16,
                 H_D_init='xavier',
                 num_heads=8,
                 num_spin_blocks=8,
                 dec_dim=128,
                 bound_std=False):

        super().__init__()
        # Dataset embedding
        self.data_emb = build_mlp(dim_in=(dim_x + dim_y),
                                  dim_hid=data_emb_dim,
                                  dim_out=data_emb_dim,
                                  depth=data_emb_depth)

        # Deterministic path
        self.denc = Spin(data_dim=data_emb_dim,
                         use_H_A=use_H_A,
                         H_A_dim=H_A_dim,
                         latent_dim_mult=latent_dim_mult,
                         num_induce=num_induce,
                         H_D_init=H_D_init,
                         num_heads=num_heads,
                         num_spin_blocks=num_spin_blocks)
        self.use_latent_path = use_latent_path
        self.use_spin_latent = use_spin_latent
        if use_latent_path:
            if use_spin_latent:
                self.data_emb_latent = build_mlp(dim_in=(dim_x + dim_y),
                                                 dim_hid=data_emb_dim,
                                                 dim_out=data_emb_dim,
                                                 depth=data_emb_depth)
                self.lenc = Spin(data_dim=data_emb_dim,
                                 use_H_A=use_H_A,
                                 H_A_dim=H_A_dim,
                                 latent_dim_mult=latent_dim_mult//num_induce,
                                 num_induce=num_induce,
                                 H_D_init=H_D_init,
                                 num_heads=num_heads,
                                 num_spin_blocks=num_spin_blocks)
                self.post_z = build_mlp(dim_in=data_emb_dim*num_induce,
                                        dim_hid=data_emb_dim,
                                        dim_out=data_emb_dim*2,
                                        depth=data_emb_depth)
            else:
                self.lenc = PoolingEncoder(dim_x=dim_x,
                                           dim_y=dim_y,
                                           dim_hid=data_emb_dim,
                                           dim_lat=data_emb_dim,
                                           self_attn=True,
                                           pre_depth=data_emb_depth,
                                           post_depth=data_emb_depth)

        # Target-context Cross attention
        self.target_xattn = MultiHeadAttn(dim_q=data_emb_dim,
                                          dim_k=H_A_dim if use_H_A else data_emb_dim,
                                          dim_v=H_A_dim if use_H_A else data_emb_dim,
                                          dim_out=H_A_dim if use_H_A else data_emb_dim,
                                          num_heads=num_heads)
        # Target embedding
        self.target_embed = build_mlp(dim_in=dim_x,
                                      dim_hid=data_emb_dim,
                                      dim_out=data_emb_dim,
                                      depth=data_emb_depth)

        # Decoder
        self.bound_std = bound_std
        pred_in_dim = (H_A_dim if use_H_A else data_emb_dim) + (data_emb_dim if use_latent_path else 0)
        self.predictor = nn.Sequential(
            nn.Linear(pred_in_dim, dec_dim),
            nn.ReLU(),
            nn.Linear(dec_dim, dim_y * 2)
        )

    def spin_latent_path(self, dataset, expand_size):
        latent_embed = self.data_emb_latent(dataset)
        latent_encoded = self.lenc(context=latent_embed)
        latent_encoded = latent_encoded.reshape(latent_encoded.shape[0], 1, -1)
        latent_encoded = latent_encoded.expand(latent_encoded.shape[0],
                                               expand_size,
                                               latent_encoded.shape[2])
        mu, sigma = self.post_z(latent_encoded).chunk(2, -1)
        sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
        return Normal(mu, sigma)

    def forward(self, batch, plot_func=False):
        num_ctx = batch.xc.shape[1]
        dataset = torch.cat([batch.x, batch.y], -1)
        ctx = torch.cat([batch.xc, batch.yc], -1)

        # Deterministic path
        ctx_embed = self.data_emb(ctx)
        ctx_encoded = self.denc(context=ctx_embed)
        tgt_embed = self.target_embed(batch.x)
        encoder_out = self.target_xattn(q=tgt_embed, k=ctx_encoded, v=ctx_encoded,
                                        mask=None, permute_dims=False)

        if self.use_latent_path:
            pz = self.spin_latent_path(dataset=ctx, expand_size=batch.x.shape[1]) if self.use_spin_latent \
                else self.lenc(batch.xc, batch.yc)
            if self.training:
                qz = self.spin_latent_path(dataset=dataset, expand_size=batch.x.shape[1]) if self.use_spin_latent \
                    else self.lenc(batch.x, batch.y)
                z = qz.rsample()
            else:
                z = pz.rsample()
            if not self.use_spin_latent:
                z = z.unsqueeze(1).expand(z.shape[0], batch.x.shape[1], z.shape[1])
            encoder_out = torch.cat([encoder_out, z], -1)

        decoded = self.predictor(encoder_out)
        mean, std = torch.chunk(decoded, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)
        py = Normal(mean, std)
        if plot_func:
            forward_plot_func(nt=10, batch=batch, mean=mean, std=std, ll=py.log_prob(batch.yt).sum(-1))
        outs = AttrDict()
        ll = py.log_prob(batch.y).sum(-1).mean()

        if self.training:
            if self.use_latent_path:
                outs.recon = ll
                outs.kld = kl_divergence(qz, pz).sum(-1).mean()
                outs.loss = -outs.recon + outs.kld / batch.x.shape[-2]
            else:
                outs.loss = -ll
        else:
            ll = py.log_prob(batch.y).sum(-1)
            outs.ctx_ll = ll[..., :num_ctx].mean()
            outs.tar_ll = ll[..., num_ctx:].mean()
        return outs

    def predict(self, xc, yc, xt):
        # TODO: Need to account for latent path here
        batch = AttrDict()
        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = torch.zeros((xt.shape[0], xt.shape[1], yc.shape[2])).to(batch.yt.device)

        encoded = self.denc(batch)
        query_attn = self.target_xattn(q=batch.x, k=encoded, v=encoded, mask=None, permute_dims=False)
        num_tar = batch.yt.shape[1]
        decoded = self.predictor(query_attn)[:, -num_tar:]
        return decoded
