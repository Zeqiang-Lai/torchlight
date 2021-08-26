

from torchlight.experiment.core import Experiment

# python run.py train -c config.yaml -s experiments -n test_run
# python run.py train -c config.yaml -s experiments -n test_run --debug # no log
# python run.py train -r experiments/test_run
# python run.py test -r experiments/test_run

if __name__ == '__main__':
    config = Config.parse()
    experiment = Experiment.New(savedir=config.savedir, name=config.name)
    engine = Engine(module, experiment, config)
    
    # 训练和测试都要用
    # 需要能够保存文本，图片，generic文件
    # 训练: 训练日志, 中间结果(图片)
    logger = Logger(experiment.log_dir)
    logger.write_img()
    
    # checkpoint
    ckpt = Checkpoint(experiment.ckpt_dir)
    ckpt.save()