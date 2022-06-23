import torchlight.data.datasets.pair as pair
from os.path import join
import torchvision.utils as utils

basedir = '/media/extssd/datasets/SIDD/medium/train/'

dataset = pair.PairDatasetTrain(
    imgA_dir=join(basedir, 'input'),
    imgB_dir=join(basedir, 'groundtruth'),
    patch_size=64
)

imgA, imgB = dataset.__getitem__(0)

print(imgA.shape, imgB.shape)
utils.save_image([imgA, imgB], 'train.png')


basedir='/media/exthdd/datasets/rgb/SIDD/valid/'
dataset = pair.PairDatasetVal(
    imgA_dir=join(basedir, 'input'),
    imgB_dir=join(basedir, 'target'),
)

imgA, imgB = dataset.__getitem__(0)

print(imgA.shape, imgB.shape)
utils.save_image([imgA, imgB], 'val.png')


dataset = pair.PairDatasetTest(
    inp_dir='sample'
)

imgA, path = dataset.__getitem__(0)

print(imgA.shape, path)
utils.save_image([imgA], 'test.png')
