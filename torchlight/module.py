from abc import ABC, abstractmethod


class Module(ABC):
    @abstractmethod
    def step(self, data, train, epoch, step):
        """ return a dict that contains the imgs and metrics you want to save and log
            in the format of {'save_imgs': dict, 'metrics': dict} """
        raise NotImplementedError
    
    def on_epoch_end(self, train):
        pass
    
    @abstractmethod
    def state_dict(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_state_dict(self, state):
        raise NotImplementedError