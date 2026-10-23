# Local Python setup

The project uses Python 3.12 in WSL with an NVIDIA GPU. The environment is kept
outside the OneDrive project folder, at `~/.venvs/cecs551-plant-disease`.
The package files in the repository describe the installation; the environment
itself is local and is not uploaded to GitHub.

Verified October 23, 2026: Python 3.12.3, PyTorch 2.13.0, torchvision 0.28.0,
and CUDA runtime 12.6. The package compatibility check passed, and
MobileNetV3-Small processed a synthetic image on the RTX 4060 Laptop GPU
(8,188 MiB). The notebook kernel is registered as **Plant Disease (WSL)**.
The requirements files pin the directly used packages, not every dependency.

## Opening the environment

In a WSL terminal:

```bash
source ~/.venvs/cecs551-plant-disease/bin/activate
```

From the project folder, the setup check is:

```bash
python scripts/check_environment.py --require-cuda
```

This checks the installed packages and runs MobileNetV3-Small on one synthetic
image. It downloads no weights or images and does no training. Passing this
check confirms basic GPU execution, not accuracy, training speed, or memory fit
for a real training batch.

## Recreating the environment

From the project folder in WSL, with Python 3.12 available:

```bash
python3 -m venv ~/.venvs/cecs551-plant-disease
source ~/.venvs/cecs551-plant-disease/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m pip install -r requirements.txt
python -m ipykernel install --user --name cecs551-plant-disease --display-name "Plant Disease (WSL)"
python scripts/check_environment.py --require-cuda
```

The PyTorch/torchvision pair and CUDA 12.6 wheel index follow the
[official installation instructions](https://pytorch.org/get-started/previous-versions/).
An NVIDIA driver must already be accessible in WSL. These instructions describe
this setup; a CPU-only or non-NVIDIA computer needs a different PyTorch build.

## VS Code

Python, Jupyter, and WSL extensions are already present on the checked machine;
Python and Jupyter are also present in the WSL extension installation. No extra
AI/chat plugins are needed for this work.

1. In the Command Palette, choose **WSL: Reopen Folder in WSL** to use the Linux
   environment from the editor.
2. Choose **Python: Select Interpreter**, then the environment's Python. On this
   machine it is `/home/troy/.venvs/cecs551-plant-disease/bin/python`.
3. For a notebook, choose **Select Kernel** and the **Plant Disease (WSL)** kernel
   or that same Python environment.

The existing native Windows editor can still edit the files, but its Windows
Python interpreter does not use packages installed in this Linux environment.
[VS Code notebook setup](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

## What needs image data

| Activity | Images needed? |
| --- | --- |
| Read or refine the model plan | No |
| Verify package imports and GPU execution | No; synthetic input is enough |
| Count categories, inspect quality, or make real example grids | Yes, the selected dataset files |
| Measure training memory and speed | Yes, initially a documented PlantVillage training subset |
| Run the final cross-dataset comparison | Yes, the agreed PlantVillage data and eligible cropped PlantDoc test data |

The dataset sources are [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset)
and [cropped PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset).
No datasets are downloaded by these setup commands. A later download should
record the source/version, preserve existing splits and same-leaf information,
and keep raw files under ignored `data/` or another local storage location.
The pretrained ImageNet weights are another later download, separate from the
Python packages. The first real timing trial can use PlantVillage alone; PlantDoc
test scores should not guide setup or model selection.

The installed environment occupies about 6.6 GiB. Windows C: had about 16 GiB
free after installation. Dataset download size and extracted size still need
checking before fetching full copies.
