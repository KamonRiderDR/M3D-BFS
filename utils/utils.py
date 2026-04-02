import time
import argparse
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import random
from itertools import product
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn.functional as F


def copy_module_weights(source_modules, target_modules):
    # 确保两个ModuleList的长度相同
    assert len(source_modules) == len(target_modules), "ModuleLists must have the same length"
    
    # 遍历两个ModuleList中的模块
    for source_module, target_module in zip(source_modules, target_modules):
        # 确保模块类型相同
        assert isinstance(target_module, type(source_module)), "Modules must be of the same type"
        
        # 如果模块有state_dict，则复制权重和偏置
        if hasattr(source_module, 'state_dict'):
            source_state_dict = source_module.state_dict()
            target_state_dict = target_module.state_dict()
            
            # 确保state_dict的键相同
            assert set(source_state_dict.keys()) == set(target_state_dict.keys()), "Module state_dicts must have the same keys"
            
            # 复制权重和偏置
            for key in source_state_dict:
                target_state_dict[key].copy_(source_state_dict[key])

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def KL_loss(teacher_output, student_output, T=3):   # T=10
    p_s = F.log_softmax(student_output / T, dim=1)
    p_t = F.softmax(teacher_output / T, dim=1)
    loss = F.kl_div(p_s, p_t, reduction='sum') * (T ** 2) / student_output.shape[0]

    return loss

def get_current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def K_Fold(folds, dataset, seed):
    skf = KFold(folds, shuffle=True, random_state=seed)
    test_indices = []
    for _, index in skf.split(torch.zeros(len(dataset))):
        test_indices.append(index)

    return test_indices

def cross_validate(folds, dataset):
    # 交叉验证
    skf = StratifiedKFold(folds, shuffle=True)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):
        test_indices.append(torch.from_numpy(idx))

    # val_indices = [test_indices[i - 1] for i in range(folds)]
    val_indices = test_indices

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, val_indices, test_indices

def kfold_split(self, test_index, args):
    self.k_fold = args.repetitions
    self.batch_size = args.batch_size
    assert test_index < self.k_fold
    valid_index = test_index
    test_split = self.k_fold_split[test_index]
    valid_split = self.k_fold_split[valid_index]

    train_mask = np.ones(len(self.choose_data))
    train_mask[test_split] = 0
    train_mask[valid_split] = 0
    train_split = train_mask.nonzero()[0]

    train_subset = Subset(self.choose_data, train_split.tolist())
    valid_subset = Subset(self.choose_data, valid_split.tolist())
    test_subset = Subset(self.choose_data, test_split.tolist())

    # train_subset = GraphDataset(train_subset, None, degree=True)
    # valid_subset = GraphDataset(valid_subset, None, degree=True)
    # test_subset = GraphDataset(test_subset, None, degree=True)

    # train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, collate_fn=train_subset.collate_fn())
    # val_loader = DataLoader(valid_subset, batch_size=self.batch_size, shuffle=False, collate_fn=valid_subset.collate_fn())
    # test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False, collate_fn=test_subset.collate_fn())
    
    train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
    val_loader = DataLoader(valid_subset, batch_size=self.batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)


    return train_split, train_subset, train_loader, valid_split, valid_subset, val_loader, test_split, test_subset, test_loader


def grid_search(args, param_names, params, function):
    """Grid search interface. 

    Args:
        param_names: name of all the hyper-parameters in `list` format.
            [name1, name2, ...]
        params: values of all hyper-parameters in `list` format
            [
                [val_11, val_12, val_13, ...],
                [val_21, val_22, val_23, ...],
            ]
    Return:
    """
    for upd_param in product(*params):
        print("Current parameter sets is :{}".format(upd_param))
        #* update args
        for idx, val in enumerate(upd_param):
            print(idx)
            setattr(args, param_names[idx], upd_param[idx]) 
            function(args)
