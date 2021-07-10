from abc import abstractmethod, abstractproperty
from torchlight.utils.util import PerformanceMonitor

from numpy import inf
import torch
from torchvision.utils import make_grid

from ..logging import TensorboardWriter
from ..utils.config import ConfigParser
from ..utils import MetricTracker

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config:ConfigParser):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_per_epoch = cfg_trainer['save_per_epoch']
        self.valid_per_epoch = cfg_trainer['valid_per_epoch']
        self.do_validation = cfg_trainer['do_validation']
        
        # configuration to monitor model performance and save best
        self.mnt_mode = cfg_trainer.get('monitor', 'off')
        if self.mnt_mode != 'off':
            self.mnt_mode, self.mnt_metric = self.mnt_mode.split()
            self.monitor = PerformanceMonitor(mnt_mode=self.mnt_mode, 
                                              early_stop_threshold=cfg_trainer.get('early_stop', inf))

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir
    
    @property
    @abstractproperty
    def networks(self):
        return NotImplementedError
    
    @property
    @abstractproperty
    def optimizers(self):
        return NotImplementedError
    
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
    
    def train(self):
        """
        Full training logic
        """
        if self.config.resume is not None:
            self._resume_checkpoint(self.config.resume)
            
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            if self.do_validation and epoch % self.valid_per_epoch == 0:
                val_log = self._valid_epoch(epoch)
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

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'epoch': epoch,
            'networks': {name: network.state_dict() for name, network in self.networks.items()},
            'optimizers': {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
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

        # load architecture params from checkpoint.
        for name in self.networks:
            self.networks[name].load_state_dict(checkpoint['networks'][name])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        for name in self.optimizers:
            self.optimizers[name].load_state_dict(checkpoint['optimizers'][name])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    

class AbstractTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, metric_ftns, config, device,
                 train_loader, valid_loader=None, lr_scheduler=None):
        super().__init__(config)
        self.device = device
        
        cfg_trainer = config['trainer']
        self.log_step = cfg_trainer['log_step']
        self.log_img_step = cfg_trainer['log_img_step']
        
        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_scheduler = lr_scheduler
        self.len_epoch = len(self.train_loader)

        self.metric_ftns = metric_ftns
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        for network in self.networks.values():
            network.train()
        self.train_metrics.reset()
        
        for batch_idx, data in enumerate(self.train_loader):
            results = self._step(data, train=True)

            gstep = (epoch - 1) * self.len_epoch + batch_idx + 1
            self.writer.set_step(gstep, 'train')
            
            for name, value in results['metrics'].items():
                self.train_metrics.update(name, value)

            if gstep % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} {}'.format(epoch,
                                                                 self._progress(batch_idx),
                                                                 self.train_metrics.summary()))
            if gstep % self.log_img_step == 0:
                for name, img in results['save_imgs'].items():
                    self.writer.add_image(name, make_grid(img.cpu(), nrow=8, normalize=True))
            
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        return self.train_metrics.result()

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        for network in self.networks.values():
            network.eval()
            
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_loader):
                results = self._step(data, train=False)

                gstep = (epoch - 1) * self.len_epoch + batch_idx + 1 
                self.writer.set_step(gstep, 'valid')
                
                for name, value in results['metrics'].items():
                    self.valid_metrics.update(name, value, gstep)
                    
                for name, img in results['save_imgs'].items():
                    self.writer.add_image(name, make_grid(img.cpu(), nrow=8, normalize=True))

        return self.valid_metrics.result()
    
    @abstractmethod
    def _step(self, data, train=True):
        raise NotImplementedError
    
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_loader, 'n_samples'):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)