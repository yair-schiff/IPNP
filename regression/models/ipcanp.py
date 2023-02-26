from regression.models.canp import CANP
from regression.models.modules import CrossAttnSpinEncoder, PoolingSpinEncoder, Decoder


class IPCANP(CANP):
    def __init__(self,
                 dim_x=1,
                 dim_y=1,
                 dim_hid=128,
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

        self.enc1 = CrossAttnSpinEncoder(
            dim_x=dim_x,
            dim_y=dim_y,
            dim_hid=dim_hid,
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

        self.enc2 = PoolingSpinEncoder(
            dim_x=dim_x, dim_y=dim_y,
            dim_hid=dim_hid,
            dim_lat=None,
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
            dim_enc=2*(H_A_dim if use_H_A else dim_hid),
            dim_hid=dim_hid,
            depth=dec_depth)
