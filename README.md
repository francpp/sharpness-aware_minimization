<h1 align="center"><b>SAM Optimizer</b></h1>
<h3 align="center"><b>EPFL Course - Optimization for Machine Learning - CS-439</b></h3> 
</p> 

--------------
This repository contains the code for the project of the Optimization for Machine Learning course. The project's name is **Sharpness Aware Minimization**, and it investigates the utilization of this new optimizer in three different classification models: WideResNet, AttentionModel, and GraphConvolutionalNetwork.

<br>
One of the major challenges in developing modern machine learning models, often over-parametrized, lies in studying their generalization capabilities. This refers to their ability to perform well on new data, thereby reducing the test loss of the model. Focusing solely on the training loss can lead to sub-optimal models with a high risk of overfitting. Inspired by the combined study of the loss landscape geometry and model generalization, we have exploited a new effective procedure to simultaneously minimize the value of the loss and its sharpness. This procedure, called Sharpness-Aware Minimization (SAM), seeks parameters that lie in the vicinity of uniformly low losses, leading to a min-max optimization problem that can be efficiently solved using the gradient algorithm.
<br>

## Usage
To train the model, use the following command from the main folder (replace `<model_name>` with the specific model you want to train, such as 'cifar', 'mitbih', or 'GCN'):

```
python .\trainings\my_train_<model_name>.py --optimizer=<opt> --rho=<rho>
```

Please note that you can also modify other default parameters set in the parser of each specific model. In order to customize these parameters during training, you can add additional arguments to the command line.

<br>
To build the file for evaluating the loss landscape, use the following command from the terminal, assuming you are inside the ./loss_landscape folder:

```
python plot_surface.py --model Transformer --dataset mitbih --x=-1:1:20 --y=-1:1:20 --model_file ../to_plot/model_<model_name>_<optimizer>_rho<rho>.pt --
dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot --batch_size=128 --loss_name smooth_crossentropy
```

To generate the surface plots and the eigenvalues plots, follow the instruction of the notebook `Loss_plots.ipynb`.

## Documentation

#### `DatasetClass`

| **File**    | **Description** |
| :-------------- | :-------------- |
| `cifar.py` | This file implements the CIFAR dataset class and a subset class for CIFAR-10, providing data loading and preprocessing functionality for the CIFAR-10 dataset. |
| `mitbih.py` | This file implements the MitBih dataset class and subset class for ECG signal classification. |
| `TUD.py` | This file implements a GraphDataset class for handling graph datasets from TUDataset using PyTorch Geometric. |

<br>
#### `Eigenvalues`
| **File**    | **Description** |
| :-------------- | :-------------- |
| `<model_name>_<optimizer>_eigen.npy` | |
| `<model_name>_<optimizer>_weight.npy` |  |

<br>
#### `loss_lanscape`

<br>
#### `models`

<br>
#### `outputs`

<br>
#### `plots`

<br>
#### `to_plot`

<br>
#### `trainings`

<br>
#### `utilities`


## References
Some code snippets used in the project have been sourced and adapted from the following repositories: 

1. Repository name: [SAM Optimizer](https://github.com/davda54/sam/)
   This repository provides the implementation of the SAM Optimizer and WideResNet model in PyTorch.

2. Repository name: [AttentionModel]()
   This repository contains a PyTorch implementation of the Attention Model.

3. Repository name: [GraphConvolutionalNetwork]()
   The Graph Convolutional Network repository is a PyTorch implementation of a graph convolutional neural network model.

4. Repository name: [LossLandscape]()
   ...

The code from these repositories has been used as a reference and modified to suit the requirements of the 'Sharpness Aware Minimization' project in the Optimization for Machine Learning course.
