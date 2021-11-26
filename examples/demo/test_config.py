import json
from torchlight.utils.config import ConfigParser

if __name__ == '__main__':
    # ConfigParser.backend(json, ext='json')
    config = ConfigParser.default("test")
    print(config['trainer'])