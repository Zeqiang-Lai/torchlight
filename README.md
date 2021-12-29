# Torchlight

[![Documentation Status](https://readthedocs.org/projects/torchlight/badge/?version=latest)](https://torchlight.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/torchlights)](https://pypi.org/project/torchlights/)

Torchlight is an ultra light-weight pytorch wrapper for fast prototyping of computer vision models.

[![asciicast](https://asciinema.org/a/441271.svg)](https://asciinema.org/a/441271)

## Installation

- Install via [PyPI](https://pypi.org/project/torchlights/).

```shell
pip install torchlights
```

- Install the latest version from source.

```shell
git clone https://github.com/Zeqiang-Lai/torchlight.git
cd torchlight
pip install .
pip install -e . # editable installation
```

## Features

- Most modules are self-contained.
- Debug Mode.
- User friendly progress bar .
- Save latest checkpoint if interrupted by Ctrl-C.
- Override any option in configuration file with cmd args.


## Useful Tools

- [kornia](https://github.com/kornia/kornia): Open Source Differentiable Computer Vision Library.
- [huggingface/accelerate](https://github.com/huggingface/accelerate/): A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision.
- [einops](https://github.com/arogozhnikov/einops): Flexible and powerful tensor operations for readable and reliable code.
- [image-similarity-measures](https://github.com/up42/image-similarity-measures): Implementation of eight evaluation metrics to access the similarity between two images. The eight metrics are as follows: RMSE, PSNR, SSIM, ISSM, FSIM, SRE, SAM, and UIQ.
- [huggingface/datasets](https://github.com/huggingface/datasets/): original design for NLP, but also include some vision datasets.