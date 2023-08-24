import math
import os
import random
import sys
import time
from datetime import datetime

import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from data.datasets import get_loader, get_test_loader
from utils import logger_configuration, load_weights, load_checkpoint, AverageMeter, save_checkpoint


class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

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

        out["ntscc_loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        out["loss"] = self.lmbda * 255 ** 2 * (out["mse_loss"] + out["mse_loss_ntc"]) + out["bpp_loss"]
        return out


def train_one_epoch(epoch, net, criterion, train_loader, test_loader, optimizer_G, aux_optimizer,
                    lr_scheduler, device, logger):
    best_loss = float("inf")
    elapsed, losses, psnrs, bppys, bppzs, psnr_ntcs, cbrs, cbr_ys = [AverageMeter() for _ in range(8)]
    metrics = [elapsed, losses, psnrs, bppys, bppzs, psnr_ntcs, cbrs, cbr_ys]
    global global_step
    for batch_idx, input_image in enumerate(train_loader):
        net.train()
        input_image = input_image.to(device)
        optimizer_G.zero_grad()
        aux_optimizer.zero_grad()
        global_step += 1
        start_time = time.time()

        results = net(input_image)
        out_criterion = criterion(results, input_image)
        out_criterion["loss"].backward()

        if config.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.clip_max_norm)
        optimizer_G.step()

        aux_loss = net.ntc.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        elapsed.update(time.time() - start_time)

        losses.update(out_criterion["loss"].item())
        bppys.update(out_criterion["bpp_y"].item())
        bppzs.update(out_criterion["bpp_z"].item())

        psnr_ntc = 10 * (torch.log(1. / out_criterion["mse_loss_ntc"]) / np.log(10))
        psnr_ntcs.update(psnr_ntc.item())

        losses.update(out_criterion["loss"].item())
        psnr = 10 * (torch.log(1. / out_criterion["mse_loss"]) / np.log(10))
        psnrs.update(psnr.item())
        cbrs.update(out_criterion["cbr"].item())
        cbr_ys.update(out_criterion["cbr_y"])

        if (global_step % config.print_every) == 0:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            log_info = [
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.avg:.2f}',
                f'PSNR {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'CBR {cbrs.val:.3f} ({cbrs.avg:.4f})',
                f'CBR_y {cbr_ys.val:.3f} ({cbr_ys.avg:.4f})',
                f'PSNR {psnr_ntcs.val:.2f} ({psnr_ntcs.avg:.2f})',
                f'Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})',
                f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})',
                f'Epoch {epoch}'
            ]
            log = (' | '.join(log_info))
            logger.info(log)
            if config.wandb:
                log_dict = {"PSNR": psnrs.avg, "CBR": cbrs.avg, "CBR_y": cbr_ys.avg, "Bpp_y": bppys.avg,
                            "Bpp_z": bppzs.avg, "loss": losses.avg, "Step": global_step // config.print_every,
                            "PSNR_NTC": psnr_ntcs.avg}
                wandb.log(log_dict)
            for i in metrics:
                i.clear()

        lr_scheduler.step()

        if (global_step + 1) % config.test_every == 0:
            loss = test(net, criterion, test_loader, device, logger)
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            if is_best:
                save_checkpoint(
                    {
                        "global_step": global_step,
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer_G.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    workdir
                )

        if (global_step + 1) % config.save_every == 0:
            save_checkpoint(
                {
                    "global_step": global_step,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer_G.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                False,
                workdir,
                filename='EP{}.pth.tar'.format(epoch)
            )


def test(net, criterion, test_loader, device, logger):
    import png
    with torch.no_grad():
        net.eval()
        elapsed, losses, bst_losses, psnrs, bppys, bppzs, psnr_ntcs, cbrs, cbr_ys = [AverageMeter() for _ in range(9)]
        for batch_idx, input_image in enumerate(test_loader):
            input_image = input_image.to(device)

            _, _, h, w = input_image.shape

            ph = 128
            pw = 64

            new_h = (h + ph - 1) // ph * ph
            new_w = (w + pw - 1) // pw * pw
            num_pixels = new_h * new_w
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
            results = net(input_image_pad)

            # side information should be reliably transmitted to the receiver
            sideinfo = np.array(results['indexes'].reshape(new_h // 16, new_w // 16).cpu().numpy(), dtype=np.uint8)
            png.from_array(sideinfo, 'L').save("output.png")
            try:
                # lossless compression the side information k using FLIF (https://github.com/FLIF-hub/FLIF)
                os.system('flif --overwrite -e output.png output.flif')
                bits_num = os.path.getsize('output.flif') * 8
            except:
                # revert to PNG format
                bits_num = os.path.getsize('output.png') * 8

            # for AWGN channel at SNR=10dB, we use 3/4LDPC+16QAM
            cbr_k = bits_num / 8 / (new_h * new_w)

            results["x_hat"] = F.pad(
                results["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
            )
            results["x_hat_ntc"] = F.pad(
                results["x_hat_ntc"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
            )
            out_criterion = criterion(results, input_image, num_pixels=num_pixels)

            elapsed.update(time.time() - start_time)

            losses.update(out_criterion["loss"].item())
            bppys.update(out_criterion["bpp_y"].item())
            bppzs.update(out_criterion["bpp_z"].item())

            psnr_ntc = 10 * (torch.log(1. / out_criterion["mse_loss_ntc"]) / np.log(10))
            psnr_ntcs.update(psnr_ntc.item())

            losses.update(out_criterion["loss"].item())
            bst_losses.update(out_criterion["ntscc_loss"].item())

            psnr = 10 * (torch.log(1. / out_criterion["mse_loss"]) / np.log(10))
            psnrs.update(psnr.item())
            cbrs.update(out_criterion["cbr_y"].item() + cbr_k)
            cbr_ys.update(out_criterion["cbr_y"])

            log = (' | '.join([
                f'Step [{(batch_idx + 1)}/{test_loader.__len__()}]',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.avg:.2f}',
                f'PSNR {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                f'CBR_y {cbr_ys.val:.4f} ({cbr_ys.avg:.4f})',
                f'PSNR {psnr_ntcs.val:.2f} ({psnr_ntcs.avg:.2f})',
                f'Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})',
                f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})'
            ]))
            logger.info(log)

    if not config.test_only and config.wandb:
        wandb.log({"[Kodak] PSNR": psnrs.avg,
                   "[Kodak] CBR": cbrs.avg,
                   "[Kodak] CBR_y": cbr_ys.avg,
                   "[Kodak] PSNR_NTC": psnr_ntcs.avg,
                   "[Kodak] Bpp_y": bppys.avg,
                   "[Kodak] Bpp_z": bppzs.avg,
                   "[Kodak] loss": bst_losses.avg})
    return bst_losses.avg


# def online_adaptation_test(net, criterion, test_loader, device, logger, step_num=20):
#     elapsed, losses, psnrs, bppys, bppzs, psnr_ntcs, cbrs, cbr_ys = [AverageMeter() for _ in range(8)]
#     for batch_idx, input_image in enumerate(test_loader):
#         input_image = input_image.to(device)
#         start_time = time.time()
#         with torch.no_grad():
#             y = net.ntc.g_a(input_image)
#         params = [
#             {'params': net.f_e0.parameters(), 'lr': 1e-4},
#             {'params': net.trans_sep_enc.parameters(), 'lr': 1e-4},
#             {'params': net.trans_ctx_enc.parameters(), 'lr': 1e-4},
#             {'params': torch.nn.Parameter(y, requires_grad=True), 'lr': 5e-3}
#         ]
#         optimizer = torch.optim.Adam(params)
#         for i in range(step_num):


def parse_args(argv):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='./config/ntscc.yaml',
                        help='Path to config file to replace defaults.')
    parser.add_argument('--seed', type=int, default=1024,
                        help='Random seed.')
    parser.add_argument('--gpu-id', type=str, default='0',
                        help='GPU id to use.')
    parser.add_argument('--test-only', action='store_true',
                        help='Test only (and do not run training).')

    # logging
    parser.add_argument('--exp-name', type=str, default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        help='Experiment name, unique id for trainers, logs.')
    parser.add_argument('--wandb', action="store_true",
                        help='Use wandb for logging.')
    parser.add_argument('--print-every', type=int, default=30,
                        help='Frequency of logging.')

    # dataset
    parser.add_argument('--dataset-path', type=str,
                        help='Path to the dataset')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for dataloader.')
    parser.add_argument('--training-img-size', type=tuple, default=(256, 256),
                        help='Size of the training images.')
    parser.add_argument('--eval-dataset-path', type=str,
                        help='Path to the evaluation dataset')

    # optimization
    parser.add_argument('--train-lambda', type=float, default=0.18,
                        help='Rate distortion trade-off hyper-parameter.')
    parser.add_argument('--distortion-type', type=str, default='MSE',
                        help='Distortion type, MSE/SSIM/Perceptual.')
    parser.add_argument('--optimizer-type', type=str, default='adam',
                        help='Optimizer to be used, includes optimizer modules available within `torch.optim` '
                             'and fused optimizers from `apex`, if apex is installed.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Base optimizer learning rate.')
    parser.add_argument('--aux-lr', type=float, default=0.001,
                        help='Auxiliary optimizer learning rate for factorized entropy model.')
    parser.add_argument('--eps', type=float, default=1e-8, help='Eps value for numerical stability.')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='Weight decay, applied only to decoder weights.')
    parser.add_argument('--clip-max-norm', type=float, default=1.0,
                        help='Gradient clipping for stable training.')

    # trainer
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to run the training.')
    parser.add_argument('--batch-size', type=int, default=9,
                        help='Batch size for the training.')
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint model")
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model weights.')
    parser.add_argument('--pretrained-ntc', type=str, default=None,
                        help='Path to pretrained ntc model weights.')
    parser.add_argument('--save', action="store_true",
                        help="Save the model at every epoch (no overwrite).")
    parser.add_argument('--save-every', type=int, default=10000,
                        help='Frequency of saving the model.')
    parser.add_argument('--test-every', type=int, default=5000,
                        help='Frequency of running validation.')

    # channel
    parser.add_argument('--channel-type', type=str, default='awgn',
                        help='Wireless channel type.')
    parser.add_argument('--SNR', type=float, default=10,
                        help='Signal-to-noise ratio (dB) for training.')

    # network
    parser.add_argument('--net', type=str, default='hyperprior',
                        help='Network architecture.')
    parser.add_argument('--input_resolution', type=int, default=(16, 16),
                        help='Input resolution of the network.')
    parser.add_argument('--eta', type=float,
                        help='Scaling factor, determining the ratio between estimated entropy to the number of channel-input symbols')
    parser.add_argument('--multiple-rate', type=list,
                        default=[1, 4, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224,
                                 240, 256, 272, 288, 304, 320],
                        help='Length choices of channel-input vector.')
    args = parser.parse_args(argv)
    return args


def main(argv):
    global config
    config = parse_args(argv)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    import torch.optim as optim
    from net.ntscc import NTSCC_plus
    model_architectures = {
        "checkboard2": NTSCC_plus,
    }

    if config.seed is not None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)

    device = "cuda"
    config.device = device
    job_type = 'test' if config.test_only else 'train'
    exp_name = "[{}_SNR={}_lmbda={}] ".format(config.net, config.SNR, config.train_lambda) + config.exp_name
    global workdir
    workdir, logger = logger_configuration(exp_name, job_type,
                                           method=model_architectures[config.net], save_log=(not config.test_only))
    config.logger = logger
    logger.info(config.__dict__)

    net = model_architectures[config.net](config).to(device)
    criterion = RateDistortionLoss(lmbda=config.train_lambda)

    if config.test_only:
        if config.checkpoint is not None:
            load_weights(net, config.checkpoint)
        else:
            load_weights(net, config.pretrained)
        test_loader = get_test_loader(config.eval_dataset_path)
        test(net, criterion, test_loader, device, logger)

    else:
        if config.wandb:
            wandb_init_kwargs = {
                'project': 'NTSCC',
                'name': exp_name,
                'save_code': True,
                'job_type': job_type
            }
            wandb.init(**wandb_init_kwargs)

        train_loader, test_loader = get_loader(config.dataset_path, config.eval_dataset_path, config.num_workers,
                                               config.batch_size)

        G_params = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
        aux_params = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
        optimizer_G = optim.Adam(G_params, lr=config.lr)
        aux_optimizer = optim.Adam(aux_params, lr=config.aux_lr)
        tot_epoch = config.epochs
        global global_step
        global_step = 0

        if config.pretrained is not None:
            global_step = 0
            pre_dict = torch.load(config.pretrained, map_location=device)
            result_dict = {}
            for key, weight in pre_dict["state_dict"].items():
                result_key = key
                if 'mask' not in key:
                    result_dict[result_key] = weight
            net.load_state_dict(result_dict, strict=False)
        elif config.checkpoint is not None and config.checkpoint != 'None':
            global_step = load_checkpoint(logger, device, global_step, net, optimizer_G, aux_optimizer,
                                          config.checkpoint)
        elif config.pretrained_ntc is not None:
            net.load_pretrained_ntc(config.pretrained_ntc)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[600000], gamma=0.1)

        steps_epoch = global_step // train_loader.__len__()
        for epoch in range(steps_epoch, tot_epoch):
            logger.info('======Current epoch %s ======' % epoch)
            logger.info(f"Learning rate: {optimizer_G.param_groups[0]['lr']}")
            train_one_epoch(epoch, net, criterion, train_loader, test_loader, optimizer_G, aux_optimizer,
                            lr_scheduler, device, logger)


if __name__ == '__main__':
    main(sys.argv[1:])
