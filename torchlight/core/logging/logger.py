import logging
import logging.config
import os
from pathlib import Path
import os
import yaml


from torchvision.utils import save_image


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_logging(save_dir, log_config=os.path.join(CURRENT_DIR, 'logger_config.yaml'), default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        with log_config.open('rt') as handle:
            config = yaml.load(handle, Loader=yaml.FullLoader)
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
        self.text = get_logger('Torchlight', log_dir)
        self.img_dir = log_dir / 'img'

    @property
    def tensorboard(self):
        if self.tensorboard_ is None:
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')
            self.tensorboard_ = SummaryWriter(log_dir=tensorboard_dir)
        return self.tensorboard_

    def info(self, msg):
        self.text.info(msg)

    def debug(self, msg):
        self.text.debug(msg)

    def warning(self, msg):
        self.text.warning(msg)

    def save_img(self, name, img):
        save_path: Path = self.img_dir / name
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        save_image(img, save_path)
