# Torchlight

[![Documentation Status](https://readthedocs.org/projects/torchlight/badge/?version=latest)](https://torchlight.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/torchlights)](https://pypi.org/project/torchlights/)

Torchlight provides  an light-weight PyTorch trainer, as well as many useful utils, including network components, transforms, metrics, etal. for fast prototyping of computer vision models.

**:sparkles: All top level packages are self-contained and independent. Feel free to steal any part into your own project.**

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

# or simply if you don't need editable installation
pip install git+https://github.com/Zeqiang-Lai/torchlight.git
```

## Features

- Most modules are self-contained.
- Debug Mode.
- User friendly progress bar .
- Save latest checkpoint if interrupted by Ctrl-C.
- Override any option in configuration file with cmd args.


## Useful Tools

- [kornia](https://github.com/kornia/kornia): Open Source Differentiable Computer Vision Library.
- [huggingface/datasets](https://github.com/huggingface/datasets/): original design for NLP, but also include some vision datasets.
- [huggingface/accelerate](https://github.com/huggingface/accelerate/): A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision.
- [einops](https://github.com/arogozhnikov/einops): Flexible and powerful tensor operations for readable and reliable code.
- [torch-fidelity](https://github.com/toshas/torch-fidelity): High-fidelity performance metrics for generative models in PyTorch.
- [piq](https://github.com/photosynthesis-team/piq): Measures and metrics for image2image tasks. PyTorch.
- [image-similarity-measures](https://github.com/up42/image-similarity-measures): Numpy implementation of eight evaluation metrics to access the similarity between two images. The eight metrics are as follows: RMSE, PSNR, SSIM, ISSM, FSIM, SRE, SAM, and UIQ.
- [ResizeRight](https://github.com/assafshocher/ResizeRight): The correct way to resize images or tensors. For Numpy or Pytorch (differentiable).
- [omegaconf](https://github.com/omry/omegaconf): Flexible Python configuration system. The last one you will ever need.