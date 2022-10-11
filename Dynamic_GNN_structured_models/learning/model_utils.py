from collections import OrderedDict
import torch.nn as nn

def make_mlp(in_size, layer_sizes, act='relu', last_act=None, dropout=0, prefix=''):
    if act =='tanh':
        act_f = nn.Tanh()
    elif act in ('relu', 'ReLU'):
        act_f = nn.ReLU(inplace=True)
    elif act in ('leakyrelu', 'LeakyReLU'):
        act_f = nn.LeakyReLU(inplace=True)
    elif act in ('Softplus', 'softplus'):
        act_f = nn.Softplus()
    elif act in ('Sigmoid', 'sigmoid'):
        act_f = nn.Sigmoid()
    elif act in ('Softmax', 'softmax'):
        act_f = nn.Softmax()
    elif act in ('Silu', 'silu'):
        act_f = nn.SiLU(inplace=True)
    else:
        raise ValueError(f'Unknown act: {act}')
    
    if last_act =='tanh':
        last_act_f = nn.Tanh()
    elif last_act in ('relu', 'ReLU'):
        last_act_f = nn.ReLU(inplace=True)
    elif last_act in ('leakyrelu', 'LeakyReLU'):
        last_act_f = nn.LeakyReLU(inplace=True)
    elif last_act in ('Softplus', 'softplus'):
        last_act_f = nn.Softplus()
    elif last_act in ('Sigmoid', 'sigmoid'):
        last_act_f = nn.Sigmoid()
    elif last_act in ('Softmax', 'softmax'):
        last_act_f = nn.Softmax()
    elif last_act in ('Silu', 'silu'):
        last_act_f = nn.SiLU(inplace=True)
    else:
        last_act = None

    layers = []
    for i, layer_size in enumerate(layer_sizes):
        layers.append((f'{prefix}_linear{i}', nn.Linear(in_size, layer_size)))
        if i < len(layer_sizes) - 1:
            if dropout > 0:
                layers.append((f'{prefix}_dropout{i}', nn.Dropout(dropout)))
            layers.append((f'{prefix}_{act}{i}', act_f))
        else:
            if last_act is not None:
                layers.append((f'{prefix}_{last_act}{i}', last_act_f))
        in_size = layer_size
    return nn.Sequential(OrderedDict(layers))