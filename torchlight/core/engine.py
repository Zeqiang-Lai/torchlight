from typing import NamedTuple
from pathlib import Path
import os
import time
import shutil

import torch
from torchvision.utils import make_grid
from tqdm import tqdm
from qqdm import qqdm
from numpy import inf
from colorama import Fore, init
import readchar

from .logging.logger import Logger
from .module import Module

init(autoreset=True)


class Timer:
    def __init__(self):
        self._start_time = 0

    def tic(self):
        self._start_time = time.time()

    def tok(self):
        now = time.time()
        used = int(now - self._start_time)
        second = used % 60
        used = used // 60
        minutes = used % 60
        used = used // 60
        hours = used
        return "{}:{}:{}".format(hours, minutes, second)


timer = Timer()


class Experiment:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.log_dir = self.save_dir / 'log'
        self.ckpt_dir = self.save_dir / 'ckpt'

    def create(self):
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        return self


class EngineConfig(NamedTuple):
    max_epochs: int = 10
    log_step: int = 20
    log_img_step: int = 20
    save_per_epoch: int = 1
    valid_per_epoch: int = 1

    mnt_mode: str = 'min'
    mnt_metric: str = 'loss'

    pbar: str = 'tqdm'
    num_fmt: str = '{:8.5g}'


class Engine:
    def __init__(self, module: Module, save_dir):
        self.module = module
        self.experiment = Experiment(save_dir).create()
        self.cfg = EngineConfig()
        self.logger = Logger(self.experiment.log_dir)
        self.monitor = PerformanceMonitor(self.cfg.mnt_mode)
        self.start_epoch = 1
        self.debug_mode = False

    def config(self, **kwargs):
        self.cfg = EngineConfig(**kwargs)
        return self

    def set_debug_mode(self):
        self.debug_mode = True

    def resume(self, model_name, base_dir=None):
        ckpt_dir = self.experiment.ckpt_dir if base_dir is None else Experiment(base_dir).ckpt_dir
        resume_path = ckpt_dir / 'model-{}.pth'.format(model_name)
        assert resume_path.exists(), '{} not found'.format(resume_path)
        self._resume_checkpoint(resume_path)
        return self

    def train(self, train_loader, valid_loader=None):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.cfg.max_epochs + 1):
            self._show_divider('=', text='Epoch[{}]'.format(epoch))
            try:
                self._train(train_loader, epoch, valid_loader)
            except KeyboardInterrupt:
                print(Fore.RED + "Oops!" + Fore.RESET + " Training was terminated by Ctrl-C. "
                      "Do you want to save the latest checkpoint?" " Press y/n")
                res = readchar.readchar()
                if res == 'y':
                    self._save_checkpoint(epoch)
                return

    # TODO: use this function to support mannuly control training dataloader outside of engine
    def _train(self, train_loader, epoch, valid_loader=None):
        timer.tic()
        result = self._train_epoch(epoch, train_loader)
        self.epoch = epoch

        # save logged informations into log dict
        log = {'epoch': epoch, 'time': timer.tok()}
        log.update(result)

        # print logged informations to the screen
        if valid_loader is not None and (epoch % self.cfg.valid_per_epoch == 0 or self.debug_mode):
            val_log = self._valid_epoch(epoch, valid_loader)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.debug_mode:
            return

        self._log_log(log)

        # # evaluate model performance according to configured metric, save best checkpoint as model_best
        if self.cfg.mnt_mode != 'off':
            assert self.cfg.mnt_metric in log.keys(), '%s not in log keys' % self.cfg.mnt_metric
            self.monitor.update(log[self.cfg.mnt_metric])
        #     if self.monitor.should_early_stop():
        #         self.logger.info("Validation performance didn\'t improve for {} epochs. "
        #                          "Training stops.".format(self.early_stop))
        #         break

        # save checkpoint
        if epoch % self.cfg.save_per_epoch == 0:
            is_best = self.monitor.is_best() if self.cfg.mnt_mode != 'off' else False
            self._save_checkpoint(epoch, save_best=is_best)

    def test(self, test_loader, name=None):
        old_logger = self.logger
        self.test_log_dir = self.experiment.save_dir / 'test' / 'epoch{}'.format(self.epoch-1)
        if name is not None:
            self.test_log_dir = self.test_log_dir / name
        self.test_log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(self.test_log_dir)

        val_log = self._valid_epoch(1, test_loader)
        log = {'test_'+k: v for k, v in val_log.items()}

        self._log_log(log)

        self.logger = old_logger

    def _log_log(self, log):
        self._show_divider('-')
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))
        self._show_divider('-')

    def _train_epoch(self, epoch, train_loader):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and met
        ric in this epoch.
        """
        metric_tracker = MetricTracker()

        len_epoch = len(train_loader)
        pbar = self._progress_bar(total=len_epoch)
        pbar.set_description('Train [{}|{}]'.format(epoch, self.cfg.max_epochs))
        self.module.on_epoch_start(train=True)
        for batch_idx, data in enumerate(train_loader):
            gstep = (epoch - 1) * len_epoch + batch_idx + 1
            results = self.module.step(data, train=True, epoch=epoch, step=gstep)

            # self.logger.tensorboard.set_step(gstep, 'train')
            for name, value in results.metrics.items():
                metric_tracker.update(name, value)
            #     self.logger.tensorboard.add_scalar(name, value, gstep)

            if gstep % self.cfg.log_step == 0:
                self.logger.debug('Train Epoch: {} {} {}'.format(epoch,
                                                                 _progress(batch_idx, train_loader),
                                                                 metric_tracker.summary()))
            if gstep % self.cfg.log_img_step == 0:
                for name, img in results.imgs.items():
                    img_name = os.path.join('train', name, '{}_{}.png'.format(epoch, gstep))
                    self.logger.save_img(img_name, make_grid(img, nrow=8, normalize=True))

            pbar.set_postfix(self.format_nums(metric_tracker.result()))
            pbar.update()

            if self.debug_mode:
                break

        pbar.close()
        self.module.on_epoch_end(train=True)

        return metric_tracker.result()

    def _valid_epoch(self, epoch, valid_loader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        metric_tracker = MetricTracker()

        len_epoch = len(valid_loader)
        pbar = self._progress_bar(total=len_epoch)
        pbar.set_description('Valid [{}|{}]'.format(epoch, self.cfg.max_epochs))
        self.module.on_epoch_start(train=False)
        with torch.no_grad():
            len_epoch = len(valid_loader)
            for batch_idx, data in enumerate(valid_loader):
                gstep = (epoch - 1) * len_epoch + batch_idx + 1
                results = self.module.step(data, train=False, epoch=epoch, step=gstep)

                for name, value in results.metrics.items():
                    metric_tracker.update(name, value, gstep)

                for name, img in results.imgs.items():
                    img_name = os.path.join('valid', name, '{}_{}.png'.format(epoch, gstep))
                    self.logger.save_img(img_name, make_grid(img, nrow=8, normalize=True))

                pbar.set_postfix(self.format_nums(metric_tracker.result()))
                pbar.update()

                if self.debug_mode:
                    break
            pbar.close()
        self.module.on_epoch_end(train=False)
        return metric_tracker.result()

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'epoch': epoch,
            'module': self.module.state_dict(),
            'monitor': self.monitor.state_dict() if self.cfg.mnt_mode != 'off' else None,
            'config': self.cfg
        }
        filename = str(self.experiment.ckpt_dir / 'model-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.experiment.ckpt_dir / 'model-best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format(best_path))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.epoch = self.start_epoch

        if self.cfg.mnt_mode != 'off':
            if 'monitor' in checkpoint:
                self.monitor.load_state_dict(checkpoint['monitor'])
            else:
                self.logger.info("Monitor state dict not found, default value will be used.")

        self.module.load_state_dict(checkpoint['module'])

        self.logger.info("Checkpoint loaded. Resume from epoch {}".format(self.start_epoch-1))

    def _progress_bar(self, total):
        if self.cfg.pbar == 'qqdm':
            bar = qqdm
        elif self.cfg.pbar == 'tqdm':
            bar = tqdm
        else:
            raise ValueError('choice of progressbar [qqdm, tqdm]')
        return bar(total=total, dynamic_ncols=True)

    def _show_divider(self, sym, text='', ncols=None):
        print(text_divider(sym, text, ncols))

    def format_nums(self, d):
        return {k: format_num(v, fmt=self.cfg.num_fmt) for k, v in d.items()}


def _progress(batch_idx, loader):
    current = batch_idx * loader.batch_size
    total = len(loader) * loader.batch_size
    return '[{}/{} ({:.0f}%)]'.format(current, total, 100.0 * current / total)


def format_num(n, fmt='{0:.3g}'):
    f = fmt.format(n).replace('+0', '+').replace('-0', '-')
    n = str(n)
    return f if len(f) < len(n) else n


def text_divider(sym, text='', ncols=None):
    if ncols is None:
        ncols = shutil.get_terminal_size()[0]
    left = ncols // 2
    right = ncols - left
    divider = sym*(left-len(text)) + text + sym*right
    return divider


class MetricTracker:
    def __init__(self):
        self._data = {}
        self.reset()

    def reset(self):
        self._data = {}

    def update(self, key, value, n=1):
        if key not in self._data.keys():
            self._data[key] = {'total': 0, 'count': 0}
        self._data[key]['total'] += value * n
        self._data[key]['count'] += n

    def avg(self, key):
        return self._data[key]['total'] / self._data[key]['count']

    def result(self):
        return {k: self._data[k]['total'] / self._data[k]['count'] for k in self._data.keys()}

    def summary(self):
        items = ['{}: {:.8f}'.format(k, v) for k, v in self.result().items()]
        return ' '.join(items)


class PerformanceMonitor:
    def __init__(self, mnt_mode, early_stop_threshold=0.1):
        self.mnt_mode = mnt_mode
        self.early_stop_threshold = early_stop_threshold

        assert self.early_stop_threshold > 0, 'early_stop_threshold should be greater than 0'
        assert self.mnt_mode in ['min', 'max']

        self.reset()

    def update(self, metric):
        improved = (self.mnt_mode == 'min' and metric <= self.mnt_best) or \
                   (self.mnt_mode == 'max' and metric >= self.mnt_best)
        self.best = False
        if improved:
            self.mnt_best = metric
            self.not_improved_count = 0
            self.best = True
        else:
            self.not_improved_count += 1

    def is_best(self):
        return self.best == True

    def should_early_stop(self):
        return self.not_improved_count > self.early_stop_threshold

    def reset(self):
        self.not_improved_count = 0
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.best = False

    def state_dict(self):
        return {'not_improved_count': self.not_improved_count, 'mnt_best': self.mnt_best, 'best': self.best}

    def load_state_dict(self, states):
        self.not_improved_count = states['not_improved_count']
        self.mnt_best = states['mnt_best']
        self.best = states['best']