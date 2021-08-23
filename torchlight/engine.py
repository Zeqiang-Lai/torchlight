from abc import abstractmethod
import time
import logging

from numpy import inf
import torch
from torchvision.utils import make_grid
from tqdm import tqdm

from .utils.util import PerformanceMonitor
from .utils.logging import TensorboardWriter, get_logger
from .utils.config import Config
from .utils import MetricTracker
from .module import Module

class BaseEngine:
    def __init__(self, module:Module, config:dict):
        self.module = module
        
        self.config = config
        self.logger = get_logger('trainer', config.log_dir, config['logger']['verbosity'])

        self.max_epochs = config['max_epochs']
        self.save_per_epoch = config['save_per_epoch']
        self.valid_per_epoch = config['valid_per_epoch']
        self.do_validation = config['do_validation']
        
        # configuration to monitor model performance and save best
        self.mnt_mode = config.get('monitor', 'off')
        if self.mnt_mode != 'off':
            self.mnt_mode, self.mnt_metric = self.mnt_mode.split()
            self.monitor = PerformanceMonitor(mnt_mode=self.mnt_mode, 
                                              early_stop_threshold=config.get('early_stop', inf))

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir
        
        if self.config.resume is not None:
            self._resume_checkpoint(self.config.resume)
            
    def train(self, train_loader, valid_loader=None):
        """
        Full training logic
        """

        for epoch in range(self.start_epoch, self.max_epochs + 1):

            result = self._train_epoch(epoch, train_loader)
            
            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            if self.do_validation and epoch % self.valid_per_epoch == 0:
                assert valid_loader is not None, 'valid loader is not provided.'
                val_log = self._valid_epoch(epoch, valid_loader)
                log.update(**{'val_'+k : v for k, v in val_log.items()})
            
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.mnt_mode != 'off':
                if self.mnt_metric in log.keys():
                    self.monitor.update(log[self.mnt_metric])
                if self.monitor.should_early_stop():
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
            
            # save checkpoint
            if epoch % self.save_per_epoch == 0:
                is_best = self.monitor.is_best() if self.mnt_mode != 'off' else False
                self._save_checkpoint(epoch, save_best=is_best)
        
        
    def test(self, test_loader):
        val_log = self._valid_epoch(0, test_loader)
        log = {'val_'+k : v for k, v in val_log.items()}
            
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))
    
    
    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError
    
    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Validation logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError
    
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
            'monitor': self.monitor.state_dict() if self.mnt_mode != 'off' else None,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        if self.mnt_mode != 'off':
            self.monitor.load_state_dict(checkpoint['monitor'])
            
        self.module.load_state_dict(checkpoint['module'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Engine(BaseEngine):
    def __init__(self, module: Module, config: dict):
        super().__init__(module, config)
        
        self.log_step = config['log_step']
        self.log_img_step = config['log_img_step']
        
        # setup visualization writer instance                
        self.writer = TensorboardWriter(config['log_dir'], self.logger, config['tensorboard'])

        self.train_metrics = MetricTracker(writer=self.writer)
        self.valid_metrics = MetricTracker(writer=self.writer)

    
    def _train_epoch(self, epoch, train_loader):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and met
        ric in this epoch.
        """
        self.train_metrics.reset()
        
        len_epoch = len(train_loader)
        pbar = tqdm(total=len_epoch)
        for batch_idx, data in enumerate(train_loader):
            gstep = (epoch - 1) * len_epoch + batch_idx + 1
            
            results = self.module.step(data, train=True, epoch=epoch, step=gstep)

            self.writer.set_step(gstep, 'train')
            
            for name, value in results['metrics'].items():
                self.train_metrics.update(name, value)

            # if gstep % self.log_step == 0:
            #     self.logger.debug('Train Epoch: {} {} {}'.format(epoch,
            #                                                      _progress(batch_idx, train_loader),
            #                                                      self.train_metrics.summary()))
            if gstep % self.log_img_step == 0:
                for name, img in results['save_imgs'].items():
                    self.writer.add_image(name, make_grid(img.cpu(), nrow=8, normalize=True))
            
            pbar.set_postfix({'epoch': epoch, 'metrics': self.train_metrics.summary()})
            pbar.update()
            
        pbar.close()
        self.module.on_epoch_end(train=True)   
         
        return self.train_metrics.result()

    def _valid_epoch(self, epoch, valid_loader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """ 
        self.valid_metrics.reset()
        with torch.no_grad():
            len_epoch = len(valid_loader)
            for batch_idx, data in enumerate(valid_loader):
                gstep = (epoch - 1) * len_epoch + batch_idx + 1 
                
                results = self.module.step(data, train=False, epoch=epoch, step=gstep)

                self.writer.set_step(gstep, 'valid')
                
                for name, value in results['metrics'].items():
                    self.valid_metrics.update(name, value, gstep)
                
                for name, img in results['save_imgs'].items():
                    self.writer.add_image(name, make_grid(img.cpu(), nrow=8, normalize=True))
                
        return self.valid_metrics.result()


def _progress(batch_idx, loader):
    base = '[{}/{} ({:.0f}%)]'
    current = batch_idx * loader.batch_size
    total = loader.n_samples
    return base.format(current, total, 100.0 * current / total)

