import torch
from torchvision.utils import make_grid
from tqdm import tqdm
from pathlib import Path

from .utils.logging.logger import Logger
from .utils.util import PerformanceMonitor
from .utils import MetricTracker
from .module import Module

class Experiment:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.log_dir = self.save_dir / 'log'
        self.ckpt_dir = self.save_dir / 'ckpt'
        
    def create(self):
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        return self

class EngineConfig:
    max_epochs = 10
    log_step = 20
    log_img_step = 20
    save_per_epoch = 1
    valid_per_epoch = 1
    
    mnt_mode = 'min'
    mnt_metric = 'loss'
    
    
    @classmethod
    def update(cls, new:dict):
        for k, v in new.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
    
    @classmethod
    def save(cls, path):
        pass


def _progress(batch_idx, loader):
    current = batch_idx * loader.batch_size
    total = len(loader) * loader.batch_size
    return '[{}/{} ({:.0f}%)]'.format(current, total, 100.0 * current / total)


class Engine:
    def __init__(self, module:Module, save_dir):
        self.module = module
        self.experiment = Experiment(save_dir).create()
        self.cfg = EngineConfig
        self.logger = Logger(self.experiment.log_dir)
        self.monitor = PerformanceMonitor(self.cfg.mnt_mode)
        self.start_epoch = 1
    
    def config(self, cfg:dict):
        EngineConfig.update(cfg)
        return self
    
    def resume(self, model_name):
        resume_path = self.experiment.ckpt_dir / 'model-{}.pth'.format(model_name)
        assert resume_path.exists(), '{} not found'.format(resume_path)
        self._resume_checkpoint(resume_path)
        return self
    
    def train(self, train_loader, valid_loader=None):
        """
        Full training logic
        """

        for epoch in range(self.start_epoch, self.cfg.max_epochs + 1):
            result = self._train_epoch(epoch, train_loader)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            if valid_loader is not None and epoch % self.cfg.valid_per_epoch == 0:
                val_log = self._valid_epoch(epoch, valid_loader)
                log.update(**{'val_'+k : v for k, v in val_log.items()})
            
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

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
        
        
    def test(self, test_loader):
        val_log = self._valid_epoch(0, test_loader)
        log = {'val_'+k : v for k, v in val_log.items()}
            
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))
    
    
    def _train_epoch(self, epoch, train_loader):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and met
        ric in this epoch.
        """
        metric_tracker = MetricTracker()
        
        len_epoch = len(train_loader)
        pbar = tqdm(total=len_epoch)
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
                    img_name = '{}_{}_{}.png'.format(name, epoch, gstep)
                    self.logger.save_img(img_name, make_grid(img.cpu(), nrow=8, normalize=True))
            
            pbar.set_postfix({'epoch': epoch, 'metrics': metric_tracker.summary()})
            pbar.update()
            
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
         
        with torch.no_grad():
            len_epoch = len(valid_loader)
            for batch_idx, data in enumerate(valid_loader):
                gstep = (epoch - 1) * len_epoch + batch_idx + 1 
                results = self.module.step(data, train=False, epoch=epoch, step=gstep)
                
                for name, value in results.metrics.items():
                    metric_tracker.update(name, value, gstep)
                
                for name, img in results.imgs.items():
                    img_name = 'valid_{}_{}_{}.png'.format(name, epoch, gstep)
                    self.logger.save_img(img_name, make_grid(img.cpu(), nrow=8, normalize=True))
                
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

        if self.cfg.mnt_mode != 'off':
            self.monitor.load_state_dict(checkpoint['monitor'])
            
        self.module.load_state_dict(checkpoint['module'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
