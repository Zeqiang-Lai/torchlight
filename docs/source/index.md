---
hide-toc: true
---

# Welcome to Torchlight


```{toctree}
:maxdepth: 3
:hidden: true

getstart
modules/index
example
```


```{toctree}
:caption: Useful Links
:hidden:
PyPI page <https://pypi.org/project/torchlights/>
GitHub Repository <https://github.com/Zeqiang-Lai/torchlight>
```

Torchlight is an ultra light-weight pytorch wrapper for fast prototyping of computer vision models.

<script id="asciicast-441271" src="https://asciinema.org/a/441271.js" async data-autoplay="true" data-loop=1></script>

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