from setuptools import setup, find_packages

setup(
    name='torchlights',
    packages=find_packages(),
    version='0.3.6',
    description='Torchlights - an light-weight PyTorch trainer, as well as many useful utils',
    author='Zeqiang Lai',
    author_email='laizeqiang@outlook.com',
    long_description_content_type = 'text/markdown',
    url='https://github.com/Zeqiang-Lai/torchlight',
    install_requires=['munch', 'colorama', 'readchar', 'tqdm', 'qqdm', 'pyyaml', 'colorlog'],
    include_package_data=True
)