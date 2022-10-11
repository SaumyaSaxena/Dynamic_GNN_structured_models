# Dynamic Inference on Graphs using Structured Transition Models
This repository contains code for our work on 'Dynamic Inference on Graphs using Structured Transition Models' accepted for publication at the International Conference on Intelligent Robots and Systems (IROS) 2022. 

Link to the paper: https://arxiv.org/abs/2209.15132

## Installation instuctions
### Clone this repo
```
git clone git@github.com:SaumyaSaxena/Dynamic_GNN_structured_models.git
```

### Create virtual env
```
virtualenv -p python3.6 venv_gnn
source venv_gnn/bin/activate
```
### Install this package

```
cd Dynamic_GNN_structured_models/
pip install -e .
```

### Install box2D environment
```
sudo apt-get install build-essential python-dev swig python-pygame
pip install git+https://github.com/pybox2d/pybox2d
pip install gym[box2d]
# Check if installed properly
python -c "import Box2D"
```

### Install isaac-gym environment (optional)
```
git clone git@github.com:iamlab-cmu/isaacgym.git
pip install -e isaacgym/python/
git clone git@github.com:iamlab-cmu/isaacgym-utils.git
pip install -e isaacgym-utils/[all]
```

### Install pytorch and pytorch geometric
```
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip install torch-geometric==1.7.2 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip install pytorch_lightning
# Check if installed properly
python -c "import torch_geometric"
```

## Data collection
To collect data for training the forward model run:
```
python scripts/collect_skill_data_box2D_envs.py
```

Use the environment config file `cfg/envs/block_grasp_box2D_env.yaml` to vary the number of blocks in the scene by changing the lists: `cfg['env_props']['blocks']['positions']` and `cfg['env_props']['blocks']['velocities']`.

## Training
To train the model run:
```
HYDRA_FULL_ERROR=1 python scripts/train_forward_model.py
```
Table below summarizes the our proposed model and ablation studies mentioned in the paper and respective datasets for training:

| Model | Model name | Dataset |
| ----------- | ----------- | ----------- |
| GIM_Temp (proposed model) | EPDTemporalObsLinearModelwithModesReactive | Box2DEnvPickup2ObjsTemporalDataset |
| No-GIM | EPDLinearObsSpringMassDamperModelwithModes | Box2DEnvPickup2ObjsDataset |
| No-GIM-Aug | EPDLinearObsSpringMassDamperModel | Box2DEnvPickup1Obj1DistractorPickup2ObjsDatasetMixed |
| GIM_Non-Temp | EPDLinearObsSpringMassDamperModelwithModes | Box2DEnvPickup2ObjsDataset |

Use the training configuration file ```cfg/train/train_forward_model.yaml``` to choose the dataset and model to train.

