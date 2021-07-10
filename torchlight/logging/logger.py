import logging
import logging.config
import os
from pathlib import Path

from ..utils import read_json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_logging(save_dir, log_config=os.path.join(CURRENT_DIR, 'logger_config.json'), default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
