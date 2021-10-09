from setuptools import setup, find_packages

setup(
    name='torchlight',
    packages=find_packages(),
    version='0.1.0',
    install_requires=['munch', 'colorama', 'readchar', 'tqdm', 'qqdm', 'pyyaml', 'colorlog']
)