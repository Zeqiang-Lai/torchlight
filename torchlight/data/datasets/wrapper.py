from torch.utils.data import Dataset

class ResizeDataset(Dataset):
    def __init__(self, dataset: Dataset, size: int):
        super(ResizeDataset, self).__init__()
        self.dataset = dataset
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        index = index % self.size
        return self.dataset.__getitem__(index)
    