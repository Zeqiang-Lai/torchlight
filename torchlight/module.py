from abc import ABC, abstractmethod
from typing import NamedTuple


class Module(ABC):
    class StepResult(NamedTuple):
        imgs: dict = {}
        metrics: dict = {}

    @abstractmethod
    def step(self, data, train, epoch, step) -> StepResult:
        """ return a StepResult that contains the imgs and metrics you want to save and log """
        raise NotImplementedError
    
    def on_epoch_start(self, train):
        pass
    
    def on_epoch_end(self, train):
        pass

    @abstractmethod
    def state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state):
        raise NotImplementedError


class SimpleModule(Module):
    def step(self, data, train, epoch, step):
        if train:
            self.model.train()
            self.optimizer.zero_grad()

        input, target = data
        input, target = input.to(self.device), target.to(self.device)
        output = self.model(input)
        loss = self.criterion(output, target)

        if train:
            loss.backward()
            self.optimizer.step()

        metrics = {'loss': loss.item()}
        results = self.on_step_end(input, output, target)
        metrics.update(results.metrics)

        return self.StepResult(imgs=results.imgs, metrics=metrics)

    def on_step_end(self, input, output, target) -> Module.StepResult:
        return self.StepResult()

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
