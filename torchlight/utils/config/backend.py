from pathlib import Path
import json
import yaml

Choices = set(['yaml', 'json'])

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def read_yaml(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)

def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        yaml.dump(content, handle, indent=4, sort_keys=False)
