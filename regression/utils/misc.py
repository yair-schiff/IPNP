import argparse
import os
from importlib.machinery import SourceFileLoader
import math
import torch
from attrdict import AttrDict
from io import StringIO
import sys

from matplotlib import pyplot as plt
import numpy as np


def gen_load_func(parser, func):
    def load(args, cmdline):
        sub_args, cmdline = parser.parse_known_args(cmdline)
        for k, v in sub_args.__dict__.items():
            args.__dict__[k] = v
        return func(**sub_args.__dict__), cmdline
    return load


def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    return SourceFileLoader(module_name, filename).load_module()
    # <module "module_name" from "filename">
    #
    # ex.
    # <module "cnp" from "models/cnp.py">


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def stack(x, num_samples=None, dim=0):
    return x if num_samples is None \
            else torch.stack([x]*num_samples, dim=dim)


def hrminsec(duration):
    hours, left = duration // 3600, duration % 3600
    mins, secs = left // 60, left % 60
    return f"{hours}hrs {mins}mins {secs}secs"


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


def forward_plot_func(nt, batch, mean, std, ll):
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(24, 24), tight_layout=True)
    for i in range(batch.xc.shape[0]):
        inds = torch.randperm(batch.xt[i].shape[0]).numpy()[:nt]
        b_cpu = AttrDict({k: v[i].squeeze().cpu().numpy() for k, v in batch.items()})
        b_cpu.xt = b_cpu.xt[inds]
        b_cpu.yt = b_cpu.yt[inds]
        m_cpu = mean[i].squeeze().detach().cpu().numpy()[inds]
        s_cpu = std[i].squeeze().detach().cpu().numpy()[inds]
        ll_cpu = ll[i].detach().cpu().numpy()[inds]
        # plt.scatter(batch_one.xc, batch_one.yc, color='black', marker='x', label='ctx gt')
        r = i // 4
        c = i % 4
        sort_inds = np.argsort(b_cpu.xt)
        ax[r, c].scatter(b_cpu.xt[sort_inds], b_cpu.yt[sort_inds], color='black', marker='*', label='tar gt')
        ax[r, c].plot(b_cpu.xt[sort_inds], m_cpu[sort_inds], color='blue', marker='o', label='pred')
        ax[r, c].fill_between(b_cpu.xt[sort_inds], m_cpu[sort_inds]-s_cpu[sort_inds], m_cpu[sort_inds]+s_cpu[sort_inds],
                              color='blue', alpha=0.1)
        for j in range(nt):
            ax[r, c].annotate(f'{ll_cpu[sort_inds][j]:0.2f}', (b_cpu.xt[sort_inds][j], m_cpu[sort_inds][j]))
        ax[r, c].set_title(f'LL: {ll_cpu[sort_inds].mean():0.3f}')
        ax[r, c].set_xticks([], [])
        ax[r, c].set_yticks([], [])
        ax[r, c].legend(loc='best')
    plt.show()
    return fig, ax


def get_argparser(exp):
    parser = argparse.ArgumentParser()

    # Experiment
    exp_parser = parser.add_argument_group('Experiment Args')
    exp_parser.add_argument('--expid', type=str, default='default')
    exp_parser.add_argument('--resume', type=str, default=None)

    # Data
    data_parser = parser.add_argument_group('Data Args')
    data_parser.add_argument('--max_num_ctx', type=int, default=64)
    data_parser.add_argument('--min_num_ctx', type=int, default=4)
    data_parser.add_argument('--max_num_tar', type=int, default=64)
    data_parser.add_argument('--min_num_tar', type=int, default=4)

    # Model
    model_parser = parser.add_argument_group('Model Args')
    model_parser.add_argument('--model', type=str,
                              choices=["np", "anp", "cnp", "canp", "bnp", "banp", "tnpd", "tnpa", "tnpnd",
                                       "ipanp", "ipcanp"])

    # Train
    train_parser = parser.add_argument_group('Train Args')
    train_parser.add_argument('--pretrain', action='store_true', default=False)
    train_parser.add_argument('--train_seed', type=int, default=0)
    train_parser.add_argument('--train_batch_size', type=int, default=16 if exp == 'gp' else 100)
    train_parser.add_argument('--train_num_samples', type=int, default=4)
    train_parser.add_argument('--lr', type=float, default=5e-4)
    train_parser.add_argument('--min_lr', type=float, default=0)
    train_parser.add_argument('--num_epochs', type=int, default=100000 if exp == 'gp' else 200)
    train_parser.add_argument('--annealer_mult', type=float, default=1.0)
    train_parser.add_argument('--print_freq', type=int, default=200 if exp == 'gp' else 1)
    train_parser.add_argument('--eval_freq', type=int, default=5000 if exp == 'gp' else 20)
    train_parser.add_argument('--save_freq', type=int, default=1000 if exp == 'gp' else 10)

    # Eval
    eval_parser = parser.add_argument_group('Eval Args')
    eval_parser.add_argument('--eval_seed', type=int, default=0)
    eval_parser.add_argument('--eval_batch_size', type=int, default=16)
    eval_parser.add_argument('--eval_num_samples', type=int, default=50)
    eval_parser.add_argument('--eval_logfile', type=str, default=None)

    # Plot
    plot_parser = parser.add_argument_group('Plot Args')
    plot_parser.add_argument('--plot_seed', type=int, default=0)
    plot_parser.add_argument('--plot_batch_size', type=int, default=16)
    plot_parser.add_argument('--plot_num_samples', type=int, default=30)
    plot_parser.add_argument('--plot_num_ctx', type=int, default=30)
    plot_parser.add_argument('--plot_num_tar', type=int, default=10)
    plot_parser.add_argument('--start_time', type=str, default=None)

    ood_parser = parser.add_argument_group('OOD Args')
    ood_parser.add_argument('--t_noise', type=float, default=None)

    if exp == 'gp':
        gp_parser = parser.add_argument_group('GP Args')
        gp_parser.add_argument('--eval_kernel', type=str, default='rbf', choices=['matern', 'periodic', 'rbf'])
        gp_parser.add_argument('--eval_num_batches', type=int, default=3000)

    elif exp == 'celeba':
        celeba_parser = parser.add_argument_group('CelebA Args')
        celeba_parser.add_argument('--resize', type=int, default=64)
        celeba_parser.add_argument('--target_all', action='store_true')
        celeba_parser.add_argument('--plot_num_imgs', type=int, default=16)

    return parser
