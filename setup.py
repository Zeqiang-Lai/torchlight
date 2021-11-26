from setuptools import setup, find_packages

setup(
    name='torchlights',
    packages=find_packages(),
    version='0.3.0',
    install_requires=['munch', 'colorama', 'readchar', 'tqdm', 'qqdm', 'pyyaml', 'colorlog']
)