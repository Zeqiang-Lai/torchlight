from abc import ABC, abstractmethod
from typing import NamedTuple


class Module(ABC):
    class StepResult(NamedTuple):
        imgs: dict = {}
        metrics: dict = {}

    def __init__(self):
        super().__init__()
        from .engine import Engine
        self._engine: Engine = None

    def register_engine(self, engine):
        self._engine = engine
        return self

    @property
    def engine(self):
        if self._engine is None:
            raise RuntimeError("Engine is not registered yet.")
        return self._engine

    # ---------------------------------------------------------------------------- #
    #         Abstract methods that must be implemented by the subclasses.         #
    # ---------------------------------------------------------------------------- #

    @abstractmethod
    def step(self, data, train, epoch, step) -> StepResult:
        """ return a StepResult that contains the imgs and metrics you want to save and log """
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state):
        raise NotImplementedError

    # ---------------------------------------------------------------------------- #
    #                                Callback hooks                                #
    # ---------------------------------------------------------------------------- #

    def on_epoch_start(self, train):
        pass

    def on_epoch_end(self, train):
        pass


class SMSOModule(Module):
    """ Single Model Single Optimizer Module"""

    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def step(self, data, train, epoch, step):
        if train:
            self.model.train()
            self.optimizer.zero_grad()

        loss, result = self._step(data, train, epoch, step)
        result.metrics['loss'] = loss.item()
        
        if train:
            loss.backward()
            self.optimizer.step()

        return result

    def _step(self, data, train, epoch, step):
        """ return (loss, self.StepResult) 
        """
        raise NotImplementedError

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        if 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
