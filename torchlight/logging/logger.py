import importlib
import logging
import logging.config
import logging.handlers
import os
from datetime import datetime
from pathlib import Path


class Handler:
    @classmethod
    def console(cls):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        import colorlog
        formatter = colorlog.ColoredFormatter(fmt="%(log_color)s%(message)s",
                                              log_colors={'WARNING': 'yellow', "ERROR": 'red'})
        handler.setFormatter(formatter)
        return handler

    @classmethod
    def file(cls, level, filename):
        handler = logging.handlers.RotatingFileHandler(filename, maxBytes=10485760,
                                                       backupCount=20, encoding='utf-8')
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        return handler


log_levels = {
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}


def get_logger(name, save_dir, save2file, verbosity='debug'):
    import colorlog
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
        verbosity, log_levels.keys())
    assert verbosity in log_levels, msg_verbosity

    logger = colorlog.getLogger(name)
    logger.setLevel(log_levels[verbosity])
    logger.addHandler(Handler.console())
    if save2file:
        logger.addHandler(Handler.file(logging.INFO, save_dir / 'info.log'))
        logger.addHandler(Handler.file(logging.DEBUG, save_dir / 'debug.log'))
    return logger


class Logger:
    def __init__(self, log_dir, save2file=True, enable_tensorboard=False):
        self.log_dir = Path(log_dir)
        self.tensorboard_ = None
        self.enable_tensorboard = enable_tensorboard
        self.text = get_logger('Torchlight', self.log_dir, save2file)
        self.img_dir = self.log_dir / 'img'

    @property
    def tensorboard(self):
        if self.tensorboard_ is None:
            tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')
            self.tensorboard_ = TensorboardWriter(tensorboard_dir, logger=self,
                                                  enabled=self.enable_tensorboard)
        return self.tensorboard_

    def info(self, msg):
        self.text.info(msg)

    def debug(self, msg):
        self.text.debug(msg)

    def warning(self, msg):
        self.text.warning(msg)

    def save_img(self, name, img):
        # TODO: remove torchvision dependency
        from torchvision.utils import save_image
        save_path: Path = self.img_dir / name
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        save_image(img, save_path)


class TensorboardWriter:
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            # if enable is false, writer is None, then add_data would be none
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr
