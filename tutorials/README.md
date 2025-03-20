# Python Tutorial

## Tech Stacks (Examples)

### Windows

- **OS**: Windows 10/11
- **Environment**: `virtualenv`/`venv` (Python 3.11)
- **Editor**: PyCharm
- **Package Management**: `pip` + `requirements.txt`
- **DL Framework**: PyTorch (CUDA if NVIDIA GPU available)
- **Version Control**: Git + GitHub Desktop
- **Extras**:
  - WSL 2 (optional for Linux-like environment)
  - CUDA Toolkit + cuDNN (GPU support)

### Linux

- **OS**: Ubuntu 20.04/21.04
- **Environment**: `virtualenv`/`venv` (Python 3.11)
- **Editor**: VS Code (Jupyter extension for `.ipynb`)
- **Package Management**: `pip` + `requirements.txt`
- **DL Framework**: PyTorch (CUDA if NVIDIA GPU available)
- **Version Control**: Git + GitHub
- **Extras**:
  - CUDA Toolkit + cuDNN (GPU support)

### macOS

- **OS**: macOS
- **Environment**: Pyenv + Pyenv-virtualenv (Python 3.11)
- **Editor**: VS Code (Jupyter extension for `.ipynb`)
- **Package Management**: `pip` + `requirements.txt`
- **DL Framework**: PyTorch (with MPS for Apple Silicon GPUs)
- **Version Control**: Git + GitHub
- **Extras**:
  - Homebrew for package management
  - Xcode Command Line Tools

### Cloud/Colab

- **Environment**: Google Colab (Browser-based)
- **DL Framework**: PyTorch (with GPU support)
- **Package Management**: `!pip install` (Colab cell)
- **Version Control**: Google Drive
- **Extras**:
  - Google Account
  - Google Drive
  - Free GPU runtime

## Install Python (Local)

### Windows

https://www.python.org/downloads/

### Linux

```bash
sudo apt-get update
sudo apt-get install python3 python3-pip
```

### MacOS

```bash
brew install python3
```

### Verify Installation

```bash
python3 --version
```

### Virtual Environment (Optional)

```bash
python3 -m venv dlo

# Linux/MacOS
source dlo/bin/activate

# Windows
.\dlo\Scripts\activate

which python
```

### Jupyter Notebook

```bash
pip install jupyter
jupyter notebook
```

### Install Libraries

```bash
# Single library
pip install <library>

# Requirements file
pip install -r requirements.txt
```

## Google Colab (Online)

https://colab.research.google.com/

# Tutorials

## Notebooks

https://cs231n.github.io/python-numpy-tutorial/

## Python

https://docs.python.org/3/tutorial/

## Numpy

https://numpy.org/doc/stable/user/absolute_beginners.html

https://assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf

## Pandas

https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html

https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

## PyTorch

https://pytorch.org/tutorials/beginner/basics/intro.html
