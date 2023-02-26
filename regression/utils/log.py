import logging
import time
from collections import OrderedDict
from os.path import split, splitext

import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


def get_logger(filename, mode='a'):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(logging.FileHandler(filename, mode=mode))
    logger.addHandler(logging.StreamHandler())
    return logger


class RunningAverage(object):
    def __init__(self, *keys):
        self.sum = OrderedDict()
        self.cnt = OrderedDict()
        self.clock = time.time()
        for key in keys:
            self.sum[key] = 0
            self.cnt[key] = 0

    def update(self, key, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if self.sum.get(key, None) is None:
            self.sum[key] = val
            self.cnt[key] = 1
        else:
            self.sum[key] = self.sum[key] + val
            self.cnt[key] += 1

    def reset(self):
        for key in self.sum.keys():
            self.sum[key] = 0
            self.cnt[key] = 0
        self.clock = time.time()

    def clear(self):
        self.sum = OrderedDict()
        self.cnt = OrderedDict()
        self.clock = time.time()

    def keys(self):
        return self.sum.keys()

    def get(self, key):
        assert(self.sum.get(key, None) is not None)
        return self.sum[key] / self.cnt[key]

    def info(self, show_et=True):
        line = ''
        for key in self.sum.keys():
            val = self.sum[key] / self.cnt[key]
            if type(val) == float:
                line += f'{key} {val:.4f} '
            else:
                line += f'{key} {val} '
        if show_et:
            line += f'({time.time()-self.clock:.3f} secs)'
        return line


def get_log(fileroot):
    step = []
    loss = []
    train_time = []
    eval_time = []
    ctxll = []
    tarll = []
    file = open(fileroot, "r")
    lines = file.readlines()
    for line in lines:
        # training step
        if "step" in line:
            linesplit = line.split(" ")
            step += [int(linesplit[3])]
            _loss = linesplit[-3]
            loss += [100 if _loss == "nan" else float(_loss)]
            train_time += [float(linesplit[-2][1:])]
        # evaluation step
        elif "ctx_ll" in line:
            linesplit = line.split(" ")
            ctxll += [float(linesplit[-5])]
            tarll += [float(linesplit[-3])]
            eval_time += [float(linesplit[-2][1:])]
    
    return step, loss, None, ctxll, tarll


def plot_log(fileroot, x_begin=None, x_end=None):
    step, loss, _, _, _ = get_log(fileroot)
    step = list(map(int, step))
    loss = list(map(float, loss))

    if x_begin is None:
        x_begin = 0
    if x_end is None:
        x_end = step[-1]
    
    print_freq = 1 if len(step) == 1 else step[1] - step[0]

    plt.clf()
    plt.plot(step[x_begin//print_freq:x_end//print_freq],
             loss[x_begin//print_freq:x_end//print_freq])
    plt.xlabel('step')
    plt.ylabel('loss')

    directory, file = split(fileroot)
    filename = splitext(file)[0]
    plt.savefig(directory + "/" + filename + f"-{x_begin}-{x_end}.png")
    plt.clf()  # clear current figure


class CustomSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            if v is not None:
                self.add_scalar(k, v)
