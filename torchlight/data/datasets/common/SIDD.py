from os.path import join

from ..pair import DatasetTrain, DatasetVal


class Train(DatasetTrain):
    def __init__(self, basedir, patch_size):
        basedir = join(basedir, 'train')
        imgA_dir = join(basedir, 'input')
        imgB_dir = join(basedir, 'groundtruth')
        super().__init__(imgA_dir, imgB_dir, patch_size)


class Valid(DatasetVal):
    def __init__(self, basedir, patch_size=None):
        basedir = join(basedir, 'valid')
        imgA_dir = join(basedir, 'input')
        imgB_dir = join(basedir, 'target')
        super().__init__(imgA_dir, imgB_dir, patch_size)
