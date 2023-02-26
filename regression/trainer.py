import os
import os.path as osp
import sys
import time

import torch
import yaml
from attrdict import AttrDict
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummaryX import summary
from tqdm import tqdm

from regression.data.celeba import CelebA
from regression.data.gp import GPSampler, RBFKernel, Matern52Kernel, PeriodicKernel
from regression.data.image import img_to_task
from regression.utils.log import get_logger, RunningAverage, CustomSummaryWriter
from regression.utils.misc import get_argparser, load_module, Capturing
from regression.utils.paths import results_path, evalsets_path

EXPERIMENTS = ['gp', 'celeba', 'emnist']
MODES = ['train', 'eval']


class RegressionTrainer:
    def __init__(self, exp, argv):
        parser = get_argparser(exp)
        args = parser.parse_args(argv)
        args.exp_title = self.gen_exp_title(exp, args)

        if args.expid is not None:
            args.root = osp.join(results_path, args.exp_title, args.model, args.expid)
        else:
            args.root = osp.join(results_path, args.exp_title, args.model)
        os.makedirs(args.root, exist_ok=True)
        tb = CustomSummaryWriter(log_dir=args.root)

        model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
        if args.resume and osp.exists(osp.join(args.root, 'config.yaml')):
            with open(osp.join(args.root, 'config.yaml'), 'r') as f:
                config = yaml.safe_load(f)
        else:
            with open(f'configs/{exp}/{args.model}.yaml', 'r') as f:
                config = yaml.safe_load(f)
            if 'ip' in args.model:  # TODO: For inducing point models, use h = (context size lower bound // 2)
                args.num_induce_ignore = config['num_induce']
                config['num_induce'] = args.min_num_ctx // 2
            args.__dict__.update(config)
            with open(osp.join(args.root, 'config.yaml'), 'w') as f:
                yaml.dump(config, f)

        if args.pretrain:
            assert args.model == 'tnpa'
            config['pretrain'] = args.pretrain

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model_cls(**config).to(self.device)
        self.args = args
        self.tb = tb

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer,
                                           T_max=self.args.num_epochs * self.args.annealer_mult,
                                           eta_min=self.args.min_lr)
        self.logfilename = os.path.join(self.args.root, f'train_{time.strftime("%Y%m%d-%H%M")}.log')
        self.start_epoch = 1
        self.train_steps = 0
        self.total_train_time = 0

        torch.manual_seed(self.args.train_seed)
        torch.cuda.manual_seed(self.args.train_seed)

    @staticmethod
    def gen_exp_title(exp, args):
        return osp.join(f'{exp}',
                        f'ctx-{args.min_num_ctx}-{args.max_num_ctx}_' +
                        f'tar-{args.min_num_tar}-{args.max_num_tar}')

    def load_checkpoint(self):
        ckpt = torch.load(os.path.join(self.args.root, 'ckpt.tar'))
        self.model.load_state_dict(ckpt.model)
        self.optimizer.load_state_dict(ckpt.optimizer)
        self.scheduler.load_state_dict(ckpt.scheduler)
        torch.random.set_rng_state(ckpt.torch_random_state)
        self.logfilename = ckpt.logfilename
        self.start_epoch = ckpt.epoch
        self.train_steps = ckpt.train_steps
        self.total_train_time = ckpt.total_train_time

    def save_checkpoint(self, epoch):
        ckpt = AttrDict()
        ckpt.model = self.model.state_dict()
        ckpt.optimizer = self.optimizer.state_dict()
        ckpt.scheduler = self.scheduler.state_dict()
        ckpt.torch_random_state = torch.random.get_rng_state()
        ckpt.logfilename = self.logfilename
        ckpt.epoch = epoch + 1
        ckpt.train_steps = self.train_steps
        ckpt.total_train_time = self.total_train_time
        torch.save(ckpt, os.path.join(self.args.root, 'ckpt.tar'))

    def train_step(self, epoch, batch_idx, batch, ravg):
        # On first step write param summary
        if epoch == self.start_epoch == 1 and batch_idx == 0:
            with torch.no_grad():
                with Capturing() as output:
                    if self.args.model in ["np", "anp", "cnp", "bnp", "banp", "ipanp"]:
                        summary(self.model, batch,
                                num_samples=self.args.train_num_samples)
                    else:
                        summary(self.model, batch)
            with open(os.path.join(self.args.root, 'param_count.txt'), 'w') as pf:
                for line in output:
                    pf.write(line + '\n')

        self.train_steps += 1
        step_start_time = time.time()
        self.optimizer.zero_grad()
        if self.args.model in ["np", "anp", "cnp", "bnp", "banp", "ipanp"]:
            outs = self.model(batch, num_samples=self.args.train_num_samples)
        else:
            outs = self.model(batch)
        outs.loss.backward()

        self.optimizer.step()
        self.scheduler.step()
        self.total_train_time += time.time() - step_start_time
        for k, v in outs.items():
            self.tb.add_scalar(f'train/{k}', v.item(), self.train_steps)
        self.tb.add_scalar('train/time', self.total_train_time, self.train_steps)

        for key, val in outs.items():
            ravg.update(key, val)

    def train_epoch(self, epoch, ravg):
        raise NotImplementedError

    def train(self):
        # Init the eval set:
        self.get_eval_path()

        if osp.exists(self.args.root + '/ckpt.tar'):
            if self.args.resume is None:
                raise FileExistsError(self.args.root)

        with open(osp.join(self.args.root, 'args.yaml'), 'w') as f:
            yaml.dump(self.args.__dict__, f)

        torch.manual_seed(self.args.train_seed)
        torch.cuda.manual_seed(self.args.train_seed)

        if self.args.resume and osp.exists(osp.join(self.args.root, 'ckpt.tar')):
            self.load_checkpoint()

        logger = get_logger(self.logfilename)
        ravg = RunningAverage()

        # if not args.resume:
        logger.info(f'Experiment: {self.args.model}-{self.args.expid}' +
                    (' (resume)' if self.args.resume and self.start_epoch > 1 else ''))
        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f'Total number of parameters: {sum(p.numel() for p in self.model.parameters())}\n')

        if self.start_epoch == 1:
            self.tb.add_scalar('info/params', params, self.start_epoch - 1)

        # Train loop
        epoch = 0
        for epoch in range(self.start_epoch, self.args.num_epochs + 1):
            self.model.train()
            self.train_epoch(epoch, ravg)
            if epoch % self.args.print_freq == 0:
                line = f'{self.args.model}:{self.args.expid} epoch {epoch} '
                line += f'lr {self.optimizer.param_groups[0]["lr"]:.3e} '
                line += f"[train_loss] "
                line += ravg.info()
                logger.info(line)

                if epoch % self.args.eval_freq == 0:
                    lines = self.eval(epoch=epoch)
                    for line in lines:
                        logger.info(line)
                    logger.info('\n')
                ravg.reset()

            if epoch % self.args.save_freq == 0 or epoch == self.args.num_epochs:
                self.save_checkpoint(epoch)
            self.tb.add_scalar('train/lr', self.scheduler.get_last_lr()[0], epoch)

        with open(osp.join(self.args.root, 'timing.log'), 'w') as f:
            f.write(f'Total train time: {self.total_train_time}\n')
            f.write(f'Avg epoch time: {self.total_train_time / self.args.num_epochs}')

        self.args.mode = 'eval'
        self.eval(epoch=epoch)

    def eval(self, epoch=None):
        # eval a trained model on log-likelihood
        if self.args.mode == 'eval':
            ckpt = torch.load(os.path.join(self.args.root, 'ckpt.tar'), map_location=self.device)
            self.model.load_state_dict(ckpt.model)
            eval_logfile = self.get_eval_logfile()
            filename = os.path.join(self.args.root, eval_logfile)
            logger = get_logger(filename, mode='a')
        else:
            logger = None

        if self.args.mode == "eval":
            torch.manual_seed(self.args.eval_seed)
            torch.cuda.manual_seed(self.args.eval_seed)

        path, filename = self.get_eval_path()
        eval_batches = torch.load(osp.join(path, filename), map_location=self.device)

        ravg = RunningAverage()
        self.model.eval()
        lines = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_batches), ascii=True):
                for key, val in batch.items():
                    batch[key] = val.to(self.device)
                if self.args.model in ["np", "anp", "bnp", "banp", "ipanp"]:
                    outs = self.model(batch, self.args.eval_num_samples)
                else:
                    outs = self.model(batch)

                for key, val in outs.items():
                    ravg.update(key, val)
        metric_dict = {f'val/{k}': None for k in ravg.keys()}
        torch.manual_seed(time.time())
        torch.cuda.manual_seed(time.time())

        line = f'{self.args.model}:{self.args.expid} '
        if self.args.t_noise is not None:
            line += f'tn {self.args.t_noise} '
        line += ravg.info()
        if epoch > -1:
            for k in ravg.keys():
                self.tb.add_scalar(f'val/{k}', ravg.sum[k] / ravg.cnt[k], epoch)
                self.tb.add_hparams(metric_dict=metric_dict,
                                    hparam_dict=self.args.__dict__,
                                    run_name=os.path.dirname(os.path.realpath(__file__)) + os.sep + self.args.root)

        if logger is not None:
            logger.info(line)

        lines.append(line)
        return lines

    def get_eval_logfile(self):
        raise NotImplementedError

    def get_eval_path(self):
        raise NotImplementedError

    def gen_evalset(self, path, filename):
        raise NotImplementedError


class GPTrainer(RegressionTrainer):
    def __init__(self, argv):
        super(GPTrainer, self).__init__('gp', argv)
        self.sampler = GPSampler(RBFKernel())

    def train_epoch(self, epoch, ravg):
        batch = self.sampler.sample(
            batch_size=self.args.train_batch_size,
            max_num_ctx=self.args.max_num_ctx,
            min_num_ctx=self.args.min_num_ctx,
            max_num_tar=self.args.max_num_tar,
            min_num_tar=self.args.min_num_tar,
            device=self.device)
        self.train_step(epoch, batch, ravg)

    def get_eval_logfile(self):
        if not self.args.eval_logfile:
            eval_logfile = f'eval_{self.args.eval_kernel}'
            if self.args.t_noise is not None:
                eval_logfile += f'_tn_{self.args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = self.args.eval_logfile
        return eval_logfile

    def get_eval_path(self):
        path = osp.join(evalsets_path, self.args.exp_title)
        filename = f'{self.args.eval_kernel}-seed{self.args.eval_seed}'
        if self.args.t_noise is not None:
            filename += f'_{self.args.t_noise}'
        filename += '.tar'
        if not osp.isfile(osp.join(path, filename)):
            print('generating evaluation sets...')
            self.gen_evalset(path, filename)
        return path, filename

    def gen_evalset(self, path, filename):
        if self.args.eval_kernel == 'rbf':
            kernel = RBFKernel()
        elif self.args.eval_kernel == 'matern':
            kernel = Matern52Kernel()
        elif self.args.eval_kernel == 'periodic':
            kernel = PeriodicKernel()
        else:
            raise ValueError(f'Invalid kernel {self.args.eval_kernel}')
        print(f"Generating Evaluation Sets with {self.args.eval_kernel} kernel")

        sampler = GPSampler(kernel, t_noise=self.args.t_noise, seed=self.args.eval_seed)
        batches = []
        for _ in tqdm(range(self.args.eval_num_batches), ascii=True):
            batches.append(sampler.sample(
                batch_size=self.args.eval_batch_size,
                max_num_ctx=self.args.max_num_ctx,
                min_num_ctx=self.args.min_num_ctx,
                max_num_tar=self.args.max_num_tar,
                min_num_tar=self.args.min_num_tar,
                device=self.device))

        torch.manual_seed(time.time())
        torch.cuda.manual_seed(time.time())

        if not osp.isdir(path):
            os.makedirs(path)
        torch.save(batches, osp.join(path, filename))


class CelebATrainer(RegressionTrainer):
    def __init__(self, argv):
        super(CelebATrainer, self).__init__('celeba', argv)
        train_ds = CelebA(train=True, resize=self.args.resize)
        self.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.args.train_batch_size,
                                                        shuffle=True, num_workers=0)
        # Override scheduler from parent class
        self.scheduler = CosineAnnealingLR(self.optimizer,
                                           T_max=len(
                                               self.train_loader) * self.args.num_epochs * self.args.annealer_mult,
                                           eta_min=self.args.min_lr)

    @staticmethod
    def gen_exp_title(exp, args):
        if args.target_all:
            return osp.join(f'{exp}',
                            f'{args.resize}x{args.resize}',
                            f'ctx-{args.min_num_ctx}-{args.max_num_ctx}_tar-all')
        return osp.join(f'{exp}',
                        f'{args.resize}x{args.resize}',
                        f'ctx-{args.min_num_ctx}-{args.max_num_ctx}_' +
                        f'tar-{args.min_num_tar}-{args.max_num_tar}')

    def train_epoch(self, epoch, ravg):
        for batch_idx, (x, _) in enumerate(tqdm(self.train_loader, ascii=True)):
            x = x.to(self.device)
            batch = img_to_task(x,
                                min_num_ctx=self.args.min_num_ctx,
                                max_num_ctx=self.args.max_num_ctx,
                                min_num_tar=self.args.min_num_tar,
                                max_num_tar=self.args.max_num_tar,
                                target_all=self.args.target_all)
            self.train_step(epoch, batch_idx, batch, ravg)
            self.tb.add_scalar(f'train/epoch', epoch, self.train_steps)

    def get_eval_logfile(self):
        if not self.args.eval_logfile:
            eval_logfile = f'eval'
            if self.args.t_noise is not None:
                eval_logfile += f'_{self.args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = self.args.eval_logfile
        return eval_logfile

    def get_eval_path(self):
        path = osp.join(evalsets_path, self.args.exp_title)
        if not osp.isdir(path):
            os.makedirs(path)
        filename = f'no-noise_seed{self.args.eval_seed}.tar' if self.args.t_noise is None \
            else f'{self.args.t_noise}-noise_seed{self.args.eval_seed}.tar'
        if not osp.isfile(osp.join(path, filename)):
            print('generating evaluation sets...')
            self.gen_evalset(path, filename)
        return path, filename

    def gen_evalset(self, path, filename):
        eval_ds = CelebA(train=False, resize=self.args.resize)
        eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=self.args.eval_batch_size,
                                                  shuffle=False, num_workers=0)
        torch.manual_seed(self.args.eval_seed)
        torch.cuda.manual_seed(self.args.eval_seed)

        batches = []
        for x, _ in tqdm(eval_loader, ascii=True):
            batches.append(img_to_task(x,
                                       max_num_ctx=self.args.max_num_ctx,
                                       min_num_ctx=self.args.min_num_ctx,
                                       max_num_tar=self.args.max_num_tar,
                                       min_num_tar=self.args.min_num_tar,
                                       target_all=self.args.target_all,
                                       t_noise=self.args.t_noise))

        torch.manual_seed(time.time())
        torch.cuda.manual_seed(time.time())

        torch.save(batches, osp.join(path, filename))


def get_trainer(exp, argv):
    if exp == 'gp':
        return GPTrainer(argv)
    elif exp == 'celeba':
        return CelebATrainer(argv)
    else:
        raise ValueError('Invalid experiment provided. Use one of:', EXPERIMENTS)


if __name__ == '__main__':
    experiment = sys.argv[1]
    mode = sys.argv[2]
    trainer = get_trainer(experiment, sys.argv[3:])
    trainer.args.mode = mode

    if mode == 'train':
        trainer.train()
    elif mode == 'eval':
        trainer.eval()
    else:
        raise ValueError('Invalid mode provided. Use one of:', MODES)
