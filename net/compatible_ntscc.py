import math

import torch
import torch.nn as nn

from .ntscc import NTSCC_plus
from .utils import DEMUX, MUX
from .utils import LowerBound


class ChannelModNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.layer_num = layer_num = 2
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(embed_dim, hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = embed_dim
            else:
                outdim = hidden_dim
            self.bm_list.append(AdaptiveModulator(hidden_dim))
            self.sm_list.append(nn.Linear(hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, snr):
        x = x.permute(0, 2, 3, 1)  # BHWC
        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list[i](x)
            else:
                temp = self.sm_list[i](temp)
            bm = self.bm_list[i](snr.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        x = x * mod_val
        x = x.permute(0, 3, 1, 2)  # BCHW
        return x


class AdaptiveModulator(nn.Module):
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )

    def forward(self, snr):
        return self.fc(snr)


class CompatibleNTSCC(NTSCC_plus):
    """
        Note: this implementation is different from original NTSCC+ paper.
        Here we use the ChannelModNet for SNR adaptation, which is proposed in
         "Toward Adaptive Semantic Communications: Efficient Data Transmission via
         Online Learned Nonlinear Transform Source-Channel Coding".
    """
    def __init__(self, config, N=128, M=320, qr_anchor_num=5):
        super().__init__(config)
        self.qr_basic_A = nn.Parameter(torch.ones((1, M, 1, 1)))
        self.qr_basic_NA = nn.Parameter(torch.ones((1, M, 1, 1)))
        self.qr_scale = nn.Parameter(torch.ones((qr_anchor_num, 1, 1, 1)))

        self.modnet_enc = ChannelModNet(M, int(M * 1.5))
        self.modnet_dec = ChannelModNet(M, int(M * 1.5))

    def get_curr_qr(self, q_scale):
        qr_basic_A = LowerBound.apply(self.qr_basic_A, 0.5) * q_scale
        qr_basic_NA = LowerBound.apply(self.qr_basic_NA, 0.5) * q_scale
        return qr_basic_A, qr_basic_NA

    def f_e(self, x, likelihoods, eta, snr=10):
        """ Variable rate joint source channel encoder. """
        B, C, H, W = x.size()
        hx = torch.clamp_min(-torch.log(likelihoods) / math.log(2), 0)
        symbol_num = torch.sum(hx, dim=1).flatten(0) * eta
        indexes = torch.searchsorted(self.rate_choice_tensor, symbol_num).clamp(0, self.rate_num - 1)

        rate_token = torch.index_select(self.rate_token_enc, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + rate_token

        x = self.f_e0(x)
        x = self.modnet_enc(x, snr)
        x_masked, mask = self.rate_adaption_enc(x, indexes)
        return x_masked, mask, indexes

    def f_d(self, x, indexes, snr=10):
        B, C, H, W = x.size()
        x = self.rate_adaption_dec(x, indexes)
        x = self.modnet_dec(x, snr)
        x = self.f_d0(x)
        return x

    def get_q_matrix(self, x, data_a, data_na):
        q_scale_matrix = torch.ones_like(x)
        q_scale_matrix[:, :, 0::2, 0::2] = data_na
        q_scale_matrix[:, :, 1::2, 1::2] = data_na
        q_scale_matrix[:, :, 0::2, 1::2] = data_a
        q_scale_matrix[:, :, 1::2, 0::2] = data_a
        return q_scale_matrix

    def forward(self, x, qr_scale=None, snr=10):
        snr_tensor = torch.as_tensor(snr, dtype=torch.float).to(x.get_device()).reshape(-1)
        y = self.ntc.g_a(x)
        curr_qr_a, curr_qr_na = self.get_curr_qr(qr_scale)
        qr_matrix = self.get_q_matrix(y, curr_qr_a, curr_qr_na)
        y = y / qr_matrix

        z = self.ntc.h_a(y)
        z_tilde, z_likelihoods = self.ntc.entropy_bottleneck(z, training=True)
        params = self.ntc.h_s(z_tilde)
        y_tilde = self.ntc.gaussian_conditional.quantize(y, "noise")
        y_half = y_tilde.clone()

        y_tilde = y_tilde * qr_matrix
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

        likelihoods_non_anchor, likelihoods_anchor = DEMUX(y_likelihoods)
        y_non_anchor, y_anchor = DEMUX(y)
        y_anchor_sep = self.trans_sep_enc(y_anchor)

        y_non_anchor_sep = self.trans_sep_enc(y_non_anchor)
        y_non_anchor_ctx = self.trans_ctx_enc(y_non_anchor_sep, y_anchor_sep)

        y_concat = torch.cat([y_anchor_sep, y_non_anchor_ctx], dim=0)
        likelihoods_concat = torch.cat([likelihoods_anchor, likelihoods_non_anchor], dim=0)
        snr_tensor_concat = torch.cat([snr_tensor, snr_tensor], dim=0)

        s_masked, mask, indexes = self.f_e(y_concat, likelihoods_concat, self.eta, snr_tensor_concat)
        s_hat, channel_usage = self.feature_pass_channel_with_multiple_snr(s_masked, snr_tensor_concat, mask)
        y_hat_concat = self.f_d(s_hat, indexes, snr_tensor_concat)
        y_hat_anchor, y_hat_non_anchor = y_hat_concat.chunk(2, 0)

        y_hat_anchor = self.trans_sep_dec(y_hat_anchor)
        y_hat_non_anchor = self.trans_ctx_dec(y_hat_non_anchor, y_hat_anchor)
        y_hat_non_anchor = self.trans_sep_dec(y_hat_non_anchor)

        y_hat = MUX(y_hat_non_anchor, y_hat_anchor)
        y_hat = y_hat * qr_matrix
        x_hat = self.ntc.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "x_hat_ntc": x_hat_ntc,
            "indexes": indexes,
            "k": channel_usage,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
