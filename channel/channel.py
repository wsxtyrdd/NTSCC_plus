import torch.nn as nn
import numpy as np
import os
import torch


class Channel(nn.Module):
    def __init__(self, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = config.channel_type
        self.chan_param = config.SNR
        self.device = config.device
        if config.logger:
            config.logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                config.channel_type, config.SNR))

    def gaussian_noise_layer(self, input_layer, std):
        device = input_layer.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def forward(self, input, avg_pwr=None, power=1):
        B = input.size()[0]
        if avg_pwr is None:
            avg_pwr = torch.mean(input ** 2)
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(B, -1)
        channel_in = channel_in[:, ::2] + channel_in[:, 1::2] * 1j
        channel_usage = channel_in.numel()
        channel_output = self.channel_forward(channel_in)
        channel_rx = torch.zeros_like(channel_tx.reshape(B, -1))
        channel_rx[:, ::2] = torch.real(channel_output)
        channel_rx[:, 1::2] = torch.imag(channel_output)
        channel_rx = channel_rx.reshape(input_shape)
        return channel_rx * torch.sqrt(avg_pwr * 2), channel_usage

    def forward_multiple_snr(self, input, chan_params, avg_pwr=None, power=1):
        B = input.size()[0]
        device = input.get_device()
        if avg_pwr is None:
            avg_pwr = torch.mean(input ** 2)
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(B, -1)
        channel_in = channel_in[:, ::2] + channel_in[:, 1::2] * 1j
        channel_usage = channel_in.numel()

        sigmas = torch.sqrt(1.0 / (2 * 10 ** (chan_params / 10))).unsqueeze(-1)  # B 1
        noise_real = torch.normal(mean=0.0, std=1., size=np.shape(channel_in), device=device) * sigmas
        noise_imag = torch.normal(mean=0.0, std=1., size=np.shape(channel_in), device=device) * sigmas
        noise = noise_real + 1j * noise_imag
        channel_output = channel_in + noise

        channel_rx = torch.zeros_like(channel_tx.reshape(B, -1))
        channel_rx[:, ::2] = torch.real(channel_output)
        channel_rx[:, 1::2] = torch.imag(channel_output)
        channel_rx = channel_rx.reshape(input_shape)
        return channel_rx * torch.sqrt(avg_pwr * 2), channel_usage

    def channel_forward(self, channel_in):
        if self.chan_type == 0 or self.chan_type == 'noiseless':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (self.chan_param / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma)
            return chan_output