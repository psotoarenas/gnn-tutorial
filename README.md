# gnn-tutorial
Tutorial for NoF 2022: "Building Network Digital Twins (NDTs) for Next-Generation WLANs using Graph Neural Networks (GNNs)"

## Requirements local installation 
This tutorial will be based on [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html#) for building the NDT-GNN which depends on [PyTorch](https://pytorch.org/). Additionally, we will build the NDT - ML using tensorflow and scikit-learn. Moreover, jupyter is needed to be able to run the notebooks. Therefore, we recommend that you install a virtual environment to avoid issues among dependencies. For example, to build this tutorial, we used a virtual environment created with conda using the following commands.

```conda create -n gnn-tutorial -c conda-forge scikit-learn```
```conda install -c conda-forge jupyterlab```
```conda install -c conda-forge matplotlib```
```conda install tensorflow```
```conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch``` 
```conda install pyg -c pyg```


Notice that, at the time of writing this tutorial, the Conda packages of PyTorch Geometric are not published for PyTorch 1.12 yet, thus, we need to install PyTorch 1.11 instead of PyTorch 1.12 (the latest release at that time). You can also use pyenv to build your virtual environment. 

You can install the libraries by following their installation guides. But please verify that the PyTorch Geometric version matches the installed PyTorch version.

## Requirements on Google Colab
The previously mentioned packages are already installed in an instance of Google Colab. However, PyTorch Geometric is not installed by default in a Google Colab instance. Thus, you need to install the PyTorch Geometric version that satisfies the PyTorch version.

Firstly, run the following commands to find out which PyTorch and cuda version you are using.
```
import torch

def format_pytorch_version(version):
  return version.split('+')[0]

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')


TORCH_version = torch.__version__
print(f"you are using pytorch and cuda {TORCH_version}")
TORCH = format_pytorch_version(TORCH_version)
print(f"you are using pytorch {TORCH}")

CUDA_version = torch.version.cuda
print(f"you are using cuda {CUDA_version}")
CUDA = format_cuda_version(CUDA_version)
```

Then you can run in a cell the corresponding pip command to install PyTorch Geometric.

```
! pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
```

## Dataset
The official dataset is located in [Zenodo](https://doi.org/10.5281/zenodo.4106127). The final version contains the throughput, which is the main variable to be predicted. You can download it and have it in your disk. However, to facilitate the development of the tutorial, the dataset is already preprocessed and can be downloaded from the [url](https://drive.google.com/file/d/14rD7TjcSLw6Qxouk2rdgW72Op-byfuU_/view?usp=sharing) we provide. To obtain the processed dataset, we downloaded the original files, extract each .zip and then use the script in ```auxiliaries/pre_processing_dataset.py``` to process the dataset.

The step-by-step procedure is as follows:
1. Download the original files and place them into a folder (e.g., data).
2. Extract the .zip files.
3. Run the ```auxiliaries/pre_processing_dataset.py``` file in that folder with
```
python auxiliaries/pre_processing_dataset.py [ARGS]

[ARGS]
--dir full path where the data from zenodo is located.

```

Additionally, we provided a fixed split (80% training and 20% validation), so we can carry the evaluation using the same data every time we run the model.

The feature matrix for each model is built in a different way. Therefore, we provided two different scripts to automate this process. 

## Notebooks
We only provide two notebooks, one for building an NDT using traditional AI/ML and another for GNNs. Please, follow the instructions of each notebook for a correct execution. 