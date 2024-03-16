import argparse
import math
import os
import random
import sys
import time

import numpy as np
import png
import torch.utils.data
from omegaconf import OmegaConf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
from net.compatible_ntscc import CompatibleNTSCC
from utils import *
from data.datasets import get_test_loader
import torch.nn.functional as F


class RateDistortionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target, num_pixels=None):
        N, C, H, W = target.size()
        out = {}
        if num_pixels is None:
            num_pixels = N * H * W
        out["bpp_y"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["bpp_z"] = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels)
        out["cbr_y"] = output["k"] / (N * C * H * W)
        out["mse_loss_ntc"] = self.mse(output["x_hat_ntc"], target)
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["bpp_loss"] = out["bpp_y"] + out["bpp_z"]
        out["cbr"] = out["cbr_y"]
        return out


def interpolate_log(min_val, max_val, num, decending=True):
    assert max_val > min_val
    assert min_val > 0
    if decending:
        values = np.linspace(math.log(max_val), math.log(min_val), num)
    else:
        values = np.linspace(math.log(min_val), math.log(max_val), num)
    values = np.exp(values)
    return values


def test(net, eta_list, criterion, test_loader, device, logger):
    SNR_list = [0, 4, 10]
    result_dict = {}
    for i in SNR_list:
        result_dict[i] = {
            'psnr': [],
            'cbr': []
        }

    for _, SNR in enumerate(SNR_list):
        net.channel.chan_param = SNR
        logger.info("----------------------------------")
        for eta in eta_list:
            net.eta = eta
            q_scale_max = net.qr_scale[0].cpu().detach().numpy()
            q_scale_min = net.qr_scale[-1].cpu().detach().numpy()
            q_scales = interpolate_log(q_scale_min, q_scale_max, 10)
            for q_level in q_scales:
                with torch.no_grad():
                    net.eval()
                    elapsed, losses, psnrs, bppys, bppzs, psnr_ntcs, cbrs, cbr_ys = [AverageMeter() for _ in range(8)]
                    for batch_idx, input_image in enumerate(test_loader):
                        input_image = input_image.to(device)
                        b, _, h, w = input_image.shape

                        num_pixels = h * w
                        p = 256  # maximum 6 strides of 2, and window size 4 for the smallest latent fmap: 4*2^6=256
                        new_h = (h + p - 1) // p * p
                        new_w = (w + p - 1) // p * p
                        padding_left = (new_w - w) // 2
                        padding_right = new_w - w - padding_left
                        padding_top = (new_h - h) // 2
                        padding_bottom = new_h - h - padding_top
                        input_image_pad = F.pad(
                            input_image,
                            (padding_left, padding_right, padding_top, padding_bottom),
                            mode="constant",
                            value=0,
                        )

                        start_time = time.time()
                        results = net(input_image_pad, q_level, SNR)

                        # side information k compress
                        sideinfo = np.array(results['indexes'].reshape(new_h // 16, new_w // 16).cpu().numpy(),
                                            dtype=np.uint8)
                        if not os.path.exists('./tmp'):
                            os.makedirs('./tmp')
                        png.from_array(sideinfo, 'L').save("./tmp/k.png")
                        os.system('flif --overwrite -e ./tmp/k.png ./tmp/sideinfo.flif > /dev/null')
                        bits_num = os.path.getsize('./tmp/sideinfo.flif') * 8

                        # transmit side information k using an ideal capacity-achieving channel code
                        cbr_k = bits_num / np.log2(1 + 10 ** (net.channel.chan_param / 10)) / (num_pixels * 3)
                        # or with practical LDPC code, the LDPC rate and QAM level should refer to AMC table

                        results["x_hat"] = F.pad(
                            results["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
                        )
                        results["x_hat_ntc"] = F.pad(
                            results["x_hat_ntc"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
                        )
                        out_criterion = criterion(results, input_image, num_pixels=num_pixels)

                        elapsed.update(time.time() - start_time)
                        bppys.update(out_criterion["bpp_y"].item())
                        bppzs.update(out_criterion["bpp_z"].item())
                        psnr_ntc = 10 * (torch.log(1. / out_criterion["mse_loss_ntc"]) / np.log(10))
                        psnr_ntcs.update(psnr_ntc.item())

                        psnr = 10 * (torch.log(1. / out_criterion["mse_loss"]) / np.log(10))
                        psnrs.update(psnr.item())
                        cbrs.update(out_criterion["cbr"].item() + cbr_k)
                        cbr_ys.update(out_criterion["cbr_y"])

                        # log = (' | '.join([
                        #     f'Step [{(batch_idx + 1)}/{test_loader.__len__()}]',
                        #     f'Time {elapsed.avg:.2f}',
                        #     f'PSNR {psnrs.val:.2f} ({psnrs.avg:.2f})',
                        #     f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        #     f'CBR_y {cbr_ys.val:.4f} ({cbr_ys.avg:.4f})',
                        #     f'PSNR {psnr_ntcs.val:.2f} ({psnr_ntcs.avg:.2f})',
                        #     f'Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})',
                        #     f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})'
                        # ]))
                        # logger.info(log)

                log = (' | '.join([
                    f'Q_level {q_level}',
                    f'Eta {eta}',
                    f'SNR {SNR}',
                    f'Loss {losses.avg:.3f}',
                    f'PSNR {psnrs.avg:.2f}',
                    f'CBR {cbrs.avg:.4f}',
                    f'CBR_y {cbr_ys.avg:.4f}',
                    # f'PSNR {psnr_ntcs.avg:.2f}',
                    # f'Bpp_y {bppys.avg:.2f}',
                    # f'Bpp_z {bppzs.avg:.4f}'
                ]))
                logger.info(log)

                result_dict[SNR]['psnr'].append(psnrs.avg)
                result_dict[SNR]['cbr'].append(cbrs.avg)

        logger.info("======  SNR={}dB  =======".format(SNR))
        logger.info("PSNR: {}".format(result_dict[SNR]['psnr']))
        logger.info("CBR: {}".format(result_dict[SNR]['cbr']))

    return losses.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training/testing script.")
    parser.add_argument('--config', default='./config/compatible_ntscc.yaml',
                        help='Path to config file to replace defaults.')
    # parser.add_argument('--phase', type=str, default='test',
    #                     choices=['train', 'test'])
    # parser.add_argument('--method', type=str, default='compatible_ntscc',
    #                     choices=['compatible_ntscc'])
    # parser.add_argument('--exp-name', default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    #                     type=str, help='Result dir name')
    # parser.add_argument('--only-ntscc-params', action="store_false", default=False)
    # parser.add_argument('--no-wandb', action="store_false", default=True)
    # parser.add_argument('--save', action="store_true",
    #                     help="Save the model at every epoch (no overwrite).")
    # parser.add_argument("--checkpoint",
    #                     default=r'./checkpoint/compatible_NTSCC.pth.tar',
    #                     type=str, help="Path to a checkpoint of NTSCC model")
    # parser.add_argument('--eval-dataset', type=str, default='kodak',
    #                     choices=['kodak', 'clic21', 'clic22', 'tecnick'])
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    config = OmegaConf.load(args.config)

    if config.seed is not None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)

    device = "cuda"
    config.device = device
    job_type = config.phase
    exp_name = "[{}_SNR={}_lmbda={}] ".format(config.net, config.SNR, config.train_lambda) + config.exp_name
    global workdir
    workdir, logger = logger_configuration(exp_name,
                                           job_type,
                                           method=config.exp_name,
                                           save_log=(config.phase == 'train'))
    logger.info(config.__dict__)
    net = CompatibleNTSCC(config, qr_anchor_num=config.rate_num).to(device)

    if config.eval_dataset == 'kodak':
        config.testset_path = [config.base_path + '/kodak']

    assert config.phase == 'test'
    load_weights(net, config.pretrained)
    test_loader = get_test_loader(config.testset_path[0])
    eta_list = np.linspace(config.eta_min, config.eta_max, 4)
    criterion = RateDistortionLoss()
    for eta_sample in eta_list:
        net.eta = eta_sample
        logger.info("============= eta={} =============".format(eta_sample))
        test(net, eta_list, criterion, test_loader, device, logger)
        # test_snr(net, criterion, test_loader, device, logger)


if __name__ == '__main__':
    main(sys.argv[1:])
