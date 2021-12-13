from torch.utils.data import Dataset

from multiprocessing import Manager

class CacheDataset(Dataset):
    def __init__(self):
        super().__init__()
        self._cache = Manager().dict()
        
    def __getitem__(self, index):
        if index not in self._cache:
            self._cache[index] = self.get_data(index)
        return self._cache[index]
    
    def get_data(self, index):
        raise NotImplementedError