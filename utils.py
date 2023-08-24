import logging
import os

import torch


def logger_configuration(filename, phase, method='NTC', save_log=True):
    logger = logging.getLogger("NTSCC")
    workdir = './history/{}/{}'.format(method, filename)
    if phase == 'test':
        workdir += '_test'
    log = workdir + '/{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    if save_log:
        makedirs(workdir)
        makedirs(samples)
        makedirs(models)

    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    return workdir, logger


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, base_dir + "/checkpoint_best_loss.pth.tar")
    else:
        torch.save(state, base_dir + "/" + filename)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def load_weights(net, model_path, load_featenc=True):
    try:
        pretrained = torch.load(model_path)['state_dict']
    except:
        pretrained = torch.load(model_path)
    result_dict = {}
    for key, weight in pretrained.items():
        result_key = key
        if load_featenc:
            if 'mask' not in key:
                result_dict[result_key] = weight
        else:
            # if 'attn_mask' not in key and 'rate_adaption.mask' not in key\
            #         and 'fe' not in key and 'fd' not in key:
            result_dict[result_key] = weight
    print(net.load_state_dict(result_dict, strict=False))
    del result_dict, pretrained


def load_checkpoint(logger, device, global_step, net, optimizer_G, aux_optimizer, model_path):
    logger.info("Loading " + str(model_path))
    pre_dict = torch.load(model_path, map_location=device)

    global_step = pre_dict["global_step"]

    result_dict = {}
    for key, weight in pre_dict["state_dict"].items():
        result_key = key
        if 'mask' not in key:
            result_dict[result_key] = weight
    net.load_state_dict(result_dict, strict=False)

    # optimizer_G.load_state_dict(pre_dict["optimizer"])
    aux_optimizer.load_state_dict(pre_dict["aux_optimizer"])
    # lr_scheduler.load_state_dict(pre_dict["lr_scheduler"])

    return global_step
