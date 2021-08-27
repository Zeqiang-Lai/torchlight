import logging
import logging.config
import os
from pathlib import Path
import os
import json

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_logging(save_dir, log_config=os.path.join(CURRENT_DIR, 'logger_config.json'), default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        with log_config.open('rt') as handle:
            config = json.load(handle)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)


log_levels = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}

def get_logger(name, save_dir, verbosity=2):
    setup_logging(save_dir)
    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
        verbosity, log_levels.keys())
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(name)
    logger.setLevel(log_levels[verbosity])
    return logger


class Logger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.tensorboard_ = None
        self.text = get_logger('Logger', log_dir)
        self.img_dir = log_dir / 'img'
        
    @property
    def tensorboard(self):
        if self.tensorboard_ is None:
            tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')
            self.tensorboard_ = SummaryWriter(log_dir=tensorboard_dir)
        return self.tensorboard_
    
    def info(self, msg):
        self.text.info(msg)
    
    def debug(self, msg):
        self.text.debug(msg)
    
    def save_img(self, name, img):
        if not self.img_dir.exists():
            self.img_dir.mkdir()
        save_image(img, self.img_dir / name)