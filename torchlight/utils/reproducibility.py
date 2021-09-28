import torch
import random
import numpy as np
from torch.utils.data.dataloader import DataLoader

# This module provide utils for controling reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html

def setup_randomness(seed=1234, deterministic=True):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)


def SeedDataLoader(*args, seed=0, **kwargs):
    return DataLoader(*args,
        worker_init_fn=seed_worker,
        generator=seed_generator(seed),
        **kwargs
    )