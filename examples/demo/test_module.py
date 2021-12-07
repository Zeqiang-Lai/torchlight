import torchlight as tl

class TestModule(tl.Module):
    def step(self, data, train, epoch, step):
        return self.StepResult()
    
    def state_dict(self):
        return {}
    
    def load_state_dict(self, state):
        pass
    
module = TestModule()
engine = tl.Engine(module, save_dir='tmp/test_module')
module.engine.logger.info('test_module')
