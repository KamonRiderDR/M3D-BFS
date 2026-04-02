import argparse
import os
import numpy as np
import json
from itertools import product
import time

import torch
from torch.utils.data import Subset
from torch_geometric.data import DenseDataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score

from utils.dataloader import DataLoader as myDataLoader
from utils.utils import *
from utils.load_data import *
from utils.args import *
from utils.loss import *
# from models.backbone import GCN
from models.model import Model
from models.backbone import *

from Trainer import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
seeds = [25, 50, 100, 125, 150, 175, 200, 225, 250, 275]
device = ""


#* modify grid search parameters
param_names = ["alpha", "beta"]
params = [
    [0.2, 0.4, 0.6, 0.8],
    [0.3, 0.5, 0.7, 0.9]
]

# num_experts hyper-parameter
param_names = ["num_experts"]
params = [
    [3, 4, 5, 6]
]

# threshold hyper-parameter
param_names = ["threshold"]
params = [
    np.arange(0.02, 0.30, 0.02).tolist()
]

if __name__ == '__main__':
    args = get_args()
    # t_times_train(args)
    args.alpha = 0.6
    args.beta = 0.5
    if args.dataset == "HCP":
        args.times = 4
    # t_times_train(args)
    grid_search(args, param_names, params, function=t_times_train)
