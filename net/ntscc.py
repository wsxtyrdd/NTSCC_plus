import math
import time

import numpy as np
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from timm.models.layers import trunc_normal_

from channel.channel import Channel
from layer.layers import (ResViTBlock, MultistageMaskedConv2d, RateAdaptionEncoder, RateAdaptionDecoder,
                          SwinBasicLayer, CrossAttentionLayer)
from net.utils import conv, deconv, update_registered_buffers, quantize_ste, DEMUX, MUX


class NAT(nn.Module):
    r"""Nonlinear Transform based on the Neighborhood Attention Transformer.

    Introduced in `"Neighborhood Attention Transformer & Dilated Neighborhood Attention Transformer"
    <https://arxiv.org/abs/2204.07143>`_,
    by Ali Hassani, Steven Walton, Jiachen Li, Shen Li and Humphrey Shi, (2022).
    """

    def __init__(self, N, M, using_hyperprior=True,
                 depths=[2, 2, 6, 2, 2, 2], num_heads=[8, 12, 16, 20, 12, 12],
                 kernel_size=7, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a1 = ResViTBlock(dim=N,
                                depth=depths[0],
                                num_heads=num_heads[0],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:0]):sum(depths[:1])],
                                norm_layer=norm_layer,
                                )
        self.g_a2 = conv(N, N * 3 // 2, kernel_size=3, stride=2)
        self.g_a3 = ResViTBlock(dim=N * 3 // 2,
                                depth=depths[1],
                                num_heads=num_heads[1],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:1]):sum(depths[:2])],
                                norm_layer=norm_layer,
                                )
        self.g_a4 = conv(N * 3 // 2, N * 2, kernel_size=3, stride=2)
        self.g_a5 = ResViTBlock(dim=N * 2,
                                depth=depths[2],
                                num_heads=num_heads[2],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:2]):sum(depths[:3])],
                                norm_layer=norm_layer,
                                )
        self.g_a6 = conv(N * 2, M, kernel_size=3, stride=2)
        self.g_a7 = ResViTBlock(dim=M,
                                depth=depths[3],
                                num_heads=num_heads[3],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:3]):sum(depths[:4])],
                                norm_layer=norm_layer,
                                )

        if using_hyperprior:
            self.h_a0 = conv(M, N * 3 // 2, kernel_size=3, stride=2)
            self.h_a1 = ResViTBlock(dim=N * 3 // 2,
                                    depth=depths[4],
                                    num_heads=num_heads[4],
                                    kernel_size=kernel_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                    drop_path_rate=dpr[sum(depths[:4]):sum(depths[:5])],
                                    norm_layer=norm_layer,
                                    )
            self.h_a2 = conv(N * 3 // 2, N * 3 // 2, kernel_size=3, stride=2)
            self.h_a3 = ResViTBlock(dim=N * 3 // 2,
                                    depth=depths[5],
                                    num_heads=num_heads[5],
                                    kernel_size=kernel_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                    drop_path_rate=dpr[sum(depths[:5]):sum(depths[:6])],
                                    norm_layer=norm_layer,
                                    )

        depths = depths[::-1]
        num_heads = num_heads[::-1]

        if using_hyperprior:
            self.h_s0 = ResViTBlock(dim=N * 3 // 2,
                                    depth=depths[0],
                                    num_heads=num_heads[0],
                                    kernel_size=kernel_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                    drop_path_rate=dpr[sum(depths[:0]):sum(depths[:1])],
                                    norm_layer=norm_layer,
                                    )
            self.h_s1 = deconv(N * 3 // 2, N * 3 // 2, kernel_size=3, stride=2)
            self.h_s2 = ResViTBlock(dim=N * 3 // 2,
                                    depth=depths[1],
                                    num_heads=num_heads[1],
                                    kernel_size=kernel_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                    drop_path_rate=dpr[sum(depths[:1]):sum(depths[:2])],
                                    norm_layer=norm_layer,
                                    )
            self.h_s3 = deconv(N * 3 // 2, M * 2, kernel_size=3, stride=2)

        self.g_s0 = ResViTBlock(dim=M,
                                depth=depths[2],
                                num_heads=num_heads[2],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:2]):sum(depths[:3])],
                                norm_layer=norm_layer,
                                )
        self.g_s1 = deconv(M, N * 2, kernel_size=3, stride=2)
        self.g_s2 = ResViTBlock(dim=N * 2,
                                depth=depths[3],
                                num_heads=num_heads[3],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:3]):sum(depths[:4])],
                                norm_layer=norm_layer,
                                )
        self.g_s3 = deconv(N * 2, N * 3 // 2, kernel_size=3, stride=2)
        self.g_s4 = ResViTBlock(dim=N * 3 // 2,
                                depth=depths[4],
                                num_heads=num_heads[4],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:4]):sum(depths[:5])],
                                norm_layer=norm_layer,
                                )
        self.g_s5 = deconv(N * 3 // 2, N, kernel_size=3, stride=2)
        self.g_s6 = ResViTBlock(dim=N,
                                depth=depths[5],
                                num_heads=num_heads[5],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:5]):sum(depths[:6])],
                                norm_layer=norm_layer,
                                )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)
        # self.ga = nn.Sequential(self.g_a0, self.g_a1, self.g_a2, self.g_a3, self.g_a4, self.g_a5,
        #                          self.g_a6, self.g_a7)
        self.apply(self._init_weights)

    def g_a(self, x):
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x)
        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x)
        x = self.g_a6(x)
        x = self.g_a7(x)
        return x

    def g_s(self, x):
        x = self.g_s0(x)
        x = self.g_s1(x)
        x = self.g_s2(x)
        x = self.g_s3(x)
        x = self.g_s4(x)
        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s7(x)
        return x

    def h_a(self, x):
        x = self.h_a0(x)
        x = self.h_a1(x)
        x = self.h_a2(x)
        x = self.h_a3(x)
        return x

    def h_s(self, x):
        x = self.h_s0(x)
        x = self.h_s1(x)
        x = self.h_s2(x)
        x = self.h_s3(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


class NatCheckerboard2(NAT):
    r"""Nonlinear transform coding using checkerboard entropy model.
    The backbone of ga&gs are Neighborhood Attention Transformers.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=128, M=320, **kwargs):
        super().__init__(N, M, **kwargs)

        self.num_iters = 1
        self.M = M

        self.entropy_bottleneck = EntropyBottleneck(N * 3 // 2)
        self.gaussian_conditional = GaussianConditional(None)

        self.sc_transform = MultistageMaskedConv2d(
            M,
            M * 2,
            kernel_size=5, padding=2, stride=1, mask_type='B'
        )

        self.entropy_parameters = nn.Sequential(
            conv(M * 12 // 3, M * 10 // 3, 1, 1),
            nn.GELU(),
            conv(M * 10 // 3, M * 8 // 3, 1, 1),
            nn.GELU(),
            conv(M * 8 // 3, M * 6 // 3, 1, 1),
        )
        self.apply(self._init_weights)

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        params = self.h_s(z_hat)
        scales_hat, means_hat = params.chunk(2, 1)
        y_hat = quantize_ste(y - means_hat) + means_hat

        y_half = y_hat.clone()
        y_half[:, :, 0::2, 0::2] = 0
        y_half[:, :, 1::2, 1::2] = 0
        sc_params = self.sc_transform(y_half)
        sc_params[:, :, 0::2, 1::2] = 0
        sc_params[:, :, 1::2, 0::2] = 0
        gaussian_params = self.entropy_parameters(
            torch.cat((params, sc_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat = quantize_ste(y - means_hat) + means_hat

        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net


class SwinVariableRateJSCC(nn.Module):
    r""" Variable Rate Joint Source Channel Coding based on the SwinTransformer.

    Introduced in `"Nonlinear Transform Source-Channel Coding for Semantic Communications"
    <https://arxiv.org/abs/2112.10961>`_,
    by Dai Jincheng, Wang Sixian, Tan Kailin, et. al. (2022).
    """

    def __init__(self, config, N=128, M=320,
                 input_resolution=(16, 16), depths=[4, 4], num_heads=[20, 12],
                 window_size=4, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.f_e0 = SwinBasicLayer(dim=M,
                                   input_resolution=input_resolution,
                                   depth=depths[0],
                                   num_heads=num_heads[0],
                                   window_size=window_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                                   norm_layer=norm_layer,
                                   )

        self.f_d0 = SwinBasicLayer(dim=M,
                                   input_resolution=input_resolution,
                                   depth=depths[0],
                                   num_heads=num_heads[0],
                                   window_size=window_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                                   norm_layer=norm_layer,
                                   )

        self.f_list = nn.Sequential(self.f_e0, self.f_d0)
        self.f_list.apply(self._init_weights)

        self.rate_adaption_enc = RateAdaptionEncoder(M, config.multiple_rate)
        self.rate_adaption_dec = RateAdaptionDecoder(M, config.multiple_rate)

        self.rate_choice = config.multiple_rate
        self.rate_num = len(config.multiple_rate)
        self.register_buffer("rate_choice_tensor", torch.tensor(np.asarray(config.multiple_rate)))

        self.rate_token_enc = nn.Parameter(torch.zeros(self.rate_num, M))
        trunc_normal_(self.rate_token_enc, std=.02)
        self.rate_token_dec = nn.Parameter(torch.zeros(self.rate_num, M))
        trunc_normal_(self.rate_token_dec, std=.02)

        self.channel = Channel(config)
        self.eta = config.eta

    def f_e(self, x, likelihoods, eta):
        """ Variable rate joint source channel encoder. """
        B, C, H, W = x.size()
        hx = torch.clamp_min(-torch.log(likelihoods) / math.log(2), 0)
        symbol_num = torch.sum(hx, dim=1).flatten(0) * eta
        indexes = torch.searchsorted(self.rate_choice_tensor, symbol_num).clamp(0, self.rate_num - 1)

        rate_token = torch.index_select(self.rate_token_enc, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + rate_token

        x = self.f_e0(x)
        x_masked, mask = self.rate_adaption_enc(x, indexes)
        return x_masked, mask, indexes

    def f_d(self, x, indexes):
        B, C, H, W = x.size()
        x = self.rate_adaption_dec(x, indexes)
        rate_token = torch.index_select(self.rate_token_dec, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + rate_token
        x = self.f_d0(x)
        return x

    def feature_pass_channel(self, s_masked, mask=None):
        if mask is None:
            mask = torch.ones_like(s_masked)
        avg_pwr = torch.sum(s_masked ** 2) / mask.sum()
        s_hat, _ = self.channel.forward(s_masked, avg_pwr)
        s_hat = s_hat * mask
        channel_usage = mask.sum() / 2
        return s_hat, channel_usage

    def feature_pass_channel_with_multiple_snr(self, s_masked, snr, mask=None):
        if mask is None:
            mask = torch.ones_like(s_masked)
        avg_pwr = torch.sum(s_masked ** 2) / mask.sum()
        s_hat, channel_usage = self.channel.forward_multiple_snr(s_masked, snr, avg_pwr)
        s_hat = s_hat * mask
        channel_usage = mask.sum() / 2
        return s_hat, channel_usage

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


class NTSCC_plus(SwinVariableRateJSCC):
    def __init__(self, config, N=128, M=320,
                 input_resolution=(16, 16), depths=[4],
                 num_heads=[20, 20], window_size=4):
        super().__init__(config)
        self.config = config
        self.ntc = NatCheckerboard2()
        self.trans_sep_enc = SwinBasicLayer(dim=M,
                                            input_resolution=(input_resolution[0] // 2, input_resolution[1]),
                                            depth=depths[0],
                                            num_heads=num_heads[0],
                                            window_size=window_size)

        self.trans_sep_dec = SwinBasicLayer(dim=M,
                                            input_resolution=(input_resolution[0] // 2, input_resolution[1]),
                                            depth=depths[0],
                                            num_heads=num_heads[0],
                                            window_size=window_size)

        self.trans_ctx_enc = CrossAttentionLayer(dim=M,
                                                 input_resolution=(input_resolution[0] // 2, input_resolution[1]),
                                                 depth=depths[0],
                                                 num_heads=num_heads[0],
                                                 window_size=window_size)
        self.trans_ctx_dec = CrossAttentionLayer(dim=M,
                                                 input_resolution=(input_resolution[0] // 2, input_resolution[1]),
                                                 depth=depths[0],
                                                 num_heads=num_heads[0],
                                                 window_size=window_size)

    def forward(self, x):
        # NTC forward
        y = self.ntc.g_a(x)
        z = self.ntc.h_a(y)
        z_tilde, z_likelihoods = self.ntc.entropy_bottleneck(z, training=True)
        params = self.ntc.h_s(z_tilde)
        y_tilde = self.ntc.gaussian_conditional.quantize(y, "noise")
        y_half = y_tilde.clone()
        x_hat_ntc = self.ntc.g_s(y_tilde)

        y_half[:, :, 0::2, 0::2] = 0
        y_half[:, :, 1::2, 1::2] = 0
        sc_params = self.ntc.sc_transform(y_half)
        sc_params[:, :, 0::2, 1::2] = 0
        sc_params[:, :, 1::2, 0::2] = 0
        gaussian_params = self.ntc.entropy_parameters(
            torch.cat((params, sc_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.ntc.gaussian_conditional(y, scales_hat, means=means_hat)

        # deep JSCC forward
        likelihoods_non_anchor, likelihoods_anchor = DEMUX(y_likelihoods)
        y_non_anchor, y_anchor = DEMUX(y)
        y_anchor_sep = self.trans_sep_enc(y_anchor)

        y_non_anchor_sep = self.trans_sep_enc(y_non_anchor)
        y_non_anchor_ctx = self.trans_ctx_enc(y_non_anchor_sep, y_anchor_sep)

        y_concat = torch.cat([y_anchor_sep, y_non_anchor_ctx], dim=0)
        likelihoods_concat = torch.cat([likelihoods_anchor, likelihoods_non_anchor], dim=0)
        s_masked, mask, indexes = self.f_e(y_concat, likelihoods_concat, self.eta)
        s_hat, channel_usage = self.feature_pass_channel(s_masked, mask)
        y_hat_concat = self.f_d(s_hat, indexes)
        y_hat_anchor, y_hat_non_anchor = y_hat_concat.chunk(2, 0)

        y_hat_anchor = self.trans_sep_dec(y_hat_anchor)
        y_hat_non_anchor = self.trans_ctx_dec(y_hat_non_anchor, y_hat_anchor)
        y_hat_non_anchor = self.trans_sep_dec(y_hat_non_anchor)

        y_hat = MUX(y_hat_non_anchor, y_hat_anchor)
        x_hat = self.ntc.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "x_hat_ntc": x_hat_ntc,
            "indexes": indexes,
            "k": channel_usage,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def encode(self, x):
        y = self.ntc.g_a(x)
        z = self.ntc.h_a(y)
        z_tilde, z_likelihoods = self.ntc.entropy_bottleneck(z, training=True)
        params = self.ntc.h_s(z_tilde)
        y_tilde = self.ntc.gaussian_conditional.quantize(y, "noise")
        y_half = y_tilde.clone()

        y_half[:, :, 0::2, 0::2] = 0
        y_half[:, :, 1::2, 1::2] = 0
        sc_params = self.ntc.sc_transform(y_half)
        sc_params[:, :, 0::2, 1::2] = 0
        sc_params[:, :, 1::2, 0::2] = 0
        gaussian_params = self.ntc.entropy_parameters(
            torch.cat((params, sc_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.ntc.gaussian_conditional(y, scales_hat, means=means_hat)
        likelihoods_non_anchor, likelihoods_anchor = DEMUX(y_likelihoods)
        y_non_anchor, y_anchor = DEMUX(y)
        y_anchor_sep = self.trans_sep_enc(y_anchor)

        y_non_anchor_sep = self.trans_sep_enc(y_non_anchor)
        y_non_anchor_ctx = self.trans_ctx_enc(y_non_anchor_sep, y_anchor_sep)

        y_concat = torch.cat([y_anchor_sep, y_non_anchor_ctx], dim=0)
        likelihoods_concat = torch.cat([likelihoods_anchor, likelihoods_non_anchor], dim=0)
        s_masked, mask, indexes = self.f_e(y_concat, likelihoods_concat, self.eta)
        return s_masked, indexes

    def decode(self, s_hat, indexes):
        y_hat_concat = self.f_d(s_hat, indexes)
        y_hat_anchor, y_hat_non_anchor = y_hat_concat.chunk(2, 0)

        y_hat_anchor = self.trans_sep_dec(y_hat_anchor)
        y_hat_non_anchor = self.trans_ctx_dec(y_hat_non_anchor, y_hat_anchor)
        y_hat_non_anchor = self.trans_sep_dec(y_hat_non_anchor)

        y_hat = MUX(y_hat_non_anchor, y_hat_anchor)
        x_hat = self.ntc.g_s(y_hat)
        return x_hat

    def load_pretrained_ntc(self, model_path, strict=True):
        state_dict = torch.load(model_path)['state_dict']
        self.ntc.load_state_dict(state_dict, strict=strict)


if __name__ == '__main__':
    class config:
        channel_type = 'awgn'
        SNR = 10
        device = torch.device("cuda:0")

        eta = 0.2
        multiple_rate = [16, 32, 48, 64, 80, 96, 102, 118, 134, 160, 186, 192, 208, 224, 240, 256, 272, 288, 304, 320]
        logger = False


    img = torch.zeros(1, 3, 256, 256).cuda()
    model = NTSCC_plus(config)
    model = model.cuda()
    with torch.no_grad():
        results = model(img)
