import argparse
import os
import numpy as np
import json
from itertools import product
import time
from copy import deepcopy

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
from models.backbone import *
from models.model import Model

seeds = [25, 50, 100, 125, 150, 175, 200, 225, 250, 275]
device = ""
print_interval=50

#* ============================ SC-FC eval PART ========================#
def sensitivity_specificity(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    tp = con_mat[1][1]
    fp = con_mat[0][1]
    fn = con_mat[1][0]
    tn = con_mat[0][0]
    if tn == 0 and fp == 0:
        specificity = 0
    else:
        specificity = tn / (fp + tn)

    if tp == 0 and fn == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)

    return sensitivity, specificity

def eval_FCSC(args, model, loader):
    model.eval()
    Y_test = []
    Y_pred = []
    correct = 0.
    test_loss = 0.
    for i, data in enumerate(loader):
        data = data.cuda()
        labels = data.y.cuda()
        
        output, _, _ = model(data)
        
        pred = output.data.argmax(dim=1)
        correct += torch.sum(pred == labels.view(-1)).item()
        test_loss += F.cross_entropy(output, labels.view(-1)).item() * output.shape[0]

        pred_num = pred.cpu().numpy()
        y_num = labels.cpu().numpy()
        for num in range(len(pred)):
            Y_pred.append(pred_num[num])
            Y_test.append(y_num[num])

    test_acc = correct / len(loader.dataset)
    test_loss = test_loss / len(loader.dataset)
    test_sen, test_spe = sensitivity_specificity(Y_test, Y_pred)
    test_f1 = f1_score(Y_test, Y_pred)
    test_auc = roc_auc_score(Y_test, Y_pred)

    # gating function
    moe_gating = [model.sc_gates, model.fc_gates, model.fu_gates]
    return test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, Y_test, Y_pred,\
            moe_gating


def eval_FCSC_for_tsne(args, model, loader, epoch):
    model.eval()
    Y_test = []
    Y_pred = []
    correct = 0.
    test_loss = 0.
    for i, data in enumerate(loader):
        data = data.cuda()
        labels = data.y.cuda()
        
        output, sc_x, fc_x, bottleneck = model(data)
        
        tsne = {
            'y': labels.cpu().numpy(),
            'embedding': bottleneck
        }
        
        np.save('{}/tsne/ZDXX/file_{}.npy'.format(args.root, epoch), tsne)


#* ============================ [train && test] PART ========================#

def unimodal_pretrain(
    args, 
    train_loader, 
    val_loader, 
    test_loader,
    modal="fc"
):
    """Pretained Loop for Uni-modal encoder (SC && FC)

    Args:
        args: _description_

    Return:
        pretrained weights of uni-modal (sc && fc)
    """
    global device
    
    max_acc = 0.
    patience = 0
    best_epoch = 0

    encoder     = GCN(args).to(device)
    classifier  = MLP(in_features=args.hidden_dim, 
                     hidden=args.hidden_dim, 
                     out_features=args.num_classes,
                     n_layers=1, 
                     dropout=args.dropout)
    pretrained  = ModuleAE(args, encoder, classifier, decoder=classifier).to(device)
    optimizer   = torch.optim.Adam(pretrained.parameters(), lr=0.001, weight_decay=args.weight_decay)
    ckpt = None
    
    for epoch in range(args.epochs):
        pretrained.train()
        train_loss = 0.
        train_correct = 0
        for idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            data.to(device)
            params = { 
                    "modal":    modal,
                    "pooling":  args.pooling,
                    "device":   args.device
                }
            out, _   = pretrained(data, params=params)
            loss     = F.cross_entropy(out, data.y.view(-1))
            loss.backward()
            optimizer.step()
            
            pred            =  out.data.argmax(dim=1)
            train_loss      += loss.item() * out.shape[0]
            train_correct   += torch.sum(pred == data.y.view(-1)).item()

        train_loss  = train_loss / len(train_loader.dataset)
        train_acc   = train_correct / len(train_loader.dataset)
        # val && test
        val_acc, val_loss   = unimodal_test(pretrained, params, val_loader)
        test_acc, test_loss = unimodal_test(pretrained, params, test_loader)
        if epoch % 50 == 0:
            print("Epoch: {:03d} \t train_loss: {:.6f} \t train_acc: {:.4f}\t val_loss: {:.4f} \t val_acc: {:.4f} \t test_acc: {:.4f}".format(
                epoch, train_loss, train_acc, val_loss, val_acc, test_acc
            ))
        if val_acc > max_acc:
            max_acc = val_acc
            patience = 0
            best_epoch = epoch
            ckpt = deepcopy(pretrained.encoder.state_dict())
        else: 
            patience += 1
        if patience >= 150:
            break

    torch.cuda.empty_cache()
    return ckpt


def multimodal_pretrain(args, train_loader, val_loader, test_loader):
    global device
    
    max_acc_sc = 0.
    max_acc_fc = 0.
    patience = 0
    best_epoch = 0

    encoder_sc = GCN(args).to(device)
    encoder_fc = GCN(args).to(device)
    classifier_sc = MLP(
        in_features=args.hidden_dim, 
        hidden = args.hidden_dim, 
        out_features=args.num_classes,
        n_layers=1, 
        dropout=args.dropout
    )
    classifier_fc = MLP(
        in_features=args.hidden_dim, 
        hidden = args.hidden_dim, 
        out_features=args.num_classes,
        n_layers=1, 
        dropout=args.dropout
    )
    pretrained_sc = ModuleAE(args, encoder_sc, classifier_sc, decoder=classifier_sc).to(device)
    pretrained_fc = ModuleAE(args, encoder_fc, classifier_fc, decoder=classifier_fc).to(device)
    pretrained_contrast = MBCP(args, encoder_fc, encoder_sc)

    optimizer = torch.optim.Adam([
        {'params': pretrained_sc.parameters(), 'lr': 0.001},
        {'params': pretrained_fc.parameters(), 'lr': 0.001}
        ], weight_decay=args.weight_decay)
    ckpt_sc = None
    ckpt_fc = None
    
    for epoch in range(args.epochs):
        pretrained_sc.train()
        pretrained_fc.train()
        train_loss = 0.
        train_correct = 0
        for idx, data in enumerate(train_loader):
            data.to(device)
            params_sc = { "modal": "sc", "pooling": args.pooling, "device": args.device}
            params_fc = { "modal": "fc", "pooling": args.pooling, "device": args.device}
            out_sc, logits_sc = pretrained_sc(data, params=params_sc)
            out_fc, logits_fc = pretrained_fc(data, params=params_fc)

            loss_ce     = F.cross_entropy(out_sc, data.y.view(-1)) + F.cross_entropy(out_fc, data.y.view(-1))
            loss_contra = pretrained_contrast(data)
            loss        = loss_ce 
            loss.backward()
            optimizer.step()
            
            pred_sc     = out_sc.data.argmax(dim=1)
            pred_fc     = out_fc.data.argmax(dim=1)
            train_loss      += loss.item() * out_fc.shape[0]
            train_correct   += torch.sum(pred_sc == data.y.view(-1)).item()
            train_correct   += torch.sum(pred_fc == data.y.view(-1)).item()

        train_loss  = train_loss / len(train_loader.dataset)
        train_acc   = train_correct / (len(train_loader.dataset) * 2)
        #* val && test
        val_acc_sc, val_loss_sc = unimodal_test(pretrained_sc, params_sc, val_loader)
        val_acc_fc, val_loss_fc = unimodal_test(pretrained_fc, params_fc, val_loader)
        if epoch % 50 == 0:
            print("Epoch: {:03d} \t train_loss: {:.4f} \t train_acc: {:.4f}\t loss_sc: {:.4f} \t acc_sc: {:.4f} \t loss_fc: {:.4f} \t acc_fc: {:.4f}".format(
                epoch, train_loss, train_acc, val_loss_sc, val_acc_sc, val_loss_fc, val_acc_fc))
        if val_acc_sc > max_acc_sc:
            max_acc_sc = val_acc_sc
            patience = 0
            best_epoch = epoch
            ckpt_sc = deepcopy(pretrained_sc.encoder.state_dict())
        if val_acc_fc > max_acc_fc:
            max_acc_fc = val_acc_fc
            patience = 0
            best_epoch = epoch
            ckpt_fc = deepcopy(pretrained_fc.encoder.state_dict())
        else: 
            patience += 1
        if patience >= args.patience // 2:
            break

    torch.cuda.empty_cache()
    return ckpt_sc, ckpt_fc


@torch.no_grad()
def unimodal_test(model, params, test_loader):
    model.eval()
    y_test = []
    y_pred = []
    correct = 0.
    test_loss = 0.

    for i, data in enumerate(test_loader):
        data.to(device)
        out,_ = model(data, params)
        pred = out.argmax(dim=1)
        correct += torch.sum(pred == data.y.view(-1)).item()
        test_loss += F.cross_entropy(out, data.y.view(-1)) * out.shape[0]
        
        pred_num = pred.cpu().numpy()
        y_num = data.y.cpu().numpy()
        for num in range(len(pred)):
            y_pred.append(pred_num[num])
            y_test.append(y_num[num])
    test_acc = correct / len(test_loader.dataset)
    test_loss = test_loss / len(test_loader.dataset)
    return test_acc, test_loss

def train(
    args, 
    model, 
    train_loader, val_loader, test_loader,
    optimizer, scheduler,
    loss_metrics,
    uni_fc=None,
    uni_sc=None,
    mode="train",
    max_patience=None,
    i_fold=0
):
    """train for epoches within [one fold(times)]

    Args:
        args (_type_): _description_
        model (_type_): _description_

        [uni_fc, uni_sc]: `ckpt` format
        mode: [train, finetune, KD]

    """
    global device
    
    min_loss = 1e10
    max_acc = 0.
    patience = 0
    best_epoch = 0
    begin = time.time()
    ckpt = None
    early_stop = args.patience if max_patience is None else max_patience

    #* Preprocess for different training methods
    if mode == "finetune":
        model.fc_encoder.load_state_dict(uni_fc)
        model.sc_decoder.load_state_dict(uni_sc)
        # pass
    elif mode == "KD":
        tea_fc = GCN(args).to(device)
        tea_sc = GCN(args).to(device)
        tea_fc.load_state_dict(uni_fc)
        tea_sc.load_state_dict(uni_sc)

    for epoch in range(args.epochs):
        model.train()
        cur_t = time.time()
        train_loss = 0.
        train_correct = 0
        for idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            data.to(device)
            loss = 0.
            out, [sc_x, fc_x, sc_spe, fc_spe, fusion, anchor_x], [loss_moe, loss_clip] = model(data)

            # TODO NOTE
            #* loss function define and backward process
            loss_ce = F.cross_entropy(out, data.y.view(-1))
            loss_disen = sc_fc_contrastive_loss(args, anchor_x, sc_spe, fc_spe, fusion)
            # mode == KD mean uni-encoding with guidance of KD
            ''' mode for training strategy: KD or not
                    - train
                    - KD
            '''
            if mode == "KD":
                tea_fc.eval()
                tea_sc.eval()
                tea_fc_logits = tea_fc(data, name="fc")
                tea_sc_logits = tea_sc(data, name="sc")
                loss_kd = (KL_loss(tea_fc_logits, fc_x) + KL_loss(tea_sc_logits, sc_x)) / 2
                loss += args.beta * loss_kd

            ''' 3 stages for training:
                    - contrast:     {CLIP}                  loss       
                    - ce:           {CE}                    loss         
                    - mix:          {disen} + {moe} + {CE}  loss  
                    - moe_pretrain: {CLIP} + {CE}           loss
            '''
            if args.stage == "contrast":
                loss += loss_clip
            elif args.stage == "ce":
                loss += loss_ce
            elif args.stage == "mix":
                loss = loss + loss_ce + \
                              loss_moe * args.alpha + \
                              loss_disen * (1 - args.alpha)
            elif args.stage == "moe_pretrain":
                loss = loss + loss_ce + \
                              (loss_clip) * args.beta
            else:
                loss = loss_ce + loss_disen * (1 - args.alpha)

            loss.backward()
            optimizer.step()

            pred = out.data.argmax(dim=1)
            train_loss += loss.item() * out.shape[0]
            train_correct += torch.sum(pred == data.y.view(-1)).item()
        scheduler.step()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # val && test
        val_acc, val_loss, val_sen, val_spe, val_f1, val_auc, _, _ = test(args, model, val_loader)
        test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, _, _ = test(args, model, test_loader)
        if epoch % 50 == 0:
            print("Epoch: {:03d} \t train_loss: {:.6f} \t train_acc: {:.4f}\t val_loss: {:.4f} \t val_acc: {:.4f} \t test_acc: {:.4f}".format(
                epoch, train_loss, train_acc, val_loss, val_acc, test_acc
            ))

        # early stop
        if val_acc > max_acc:
            max_acc = val_acc
            best_epoch = epoch
            patience = 0   
            ckpt = deepcopy(model.state_dict())
        else:
            patience += 1
        if patience >= early_stop:
            break

    torch.cuda.empty_cache()
    return best_epoch, ckpt


@torch.no_grad()
def test(args, model, test_loader):
    """_summary_

    Args:
        args (_type_): _description_
        test_loader (_type_): _description_
    Return:
        [test_acc, test_loss, 
         test_sen, test_spe, 
         test_f1, test_auc,  
         y_test, y_pred] from the ${test_loader}
    """
    model.eval()
    y_test = []
    y_pred = []
    correct = 0.
    test_loss = 0.

    for i, data in enumerate(test_loader):
        data.to(device)
        out, _, loss = model(data)
        pred = out.argmax(dim=1)
        correct += torch.sum(pred == data.y.view(-1)).item()
        test_loss += F.cross_entropy(out, data.y.view(-1)) * out.shape[0]
        pred_num = pred.cpu().numpy()
        y_num = data.y.cpu().numpy()
        for num in range(len(pred)):
            y_pred.append(pred_num[num])
            y_test.append(y_num[num])

    test_acc = correct / len(test_loader.dataset)
    test_loss = test_loss / len(test_loader.dataset)
    test_sen, test_spe = sensitivity_specificity(y_test, y_pred)
    test_f1=f1_score(y_test, y_pred)
    test_auc=roc_auc_score(y_test, y_pred)
    
    return test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, y_test, y_pred



def t_times_train(args):
    """[t times] * [k folds] for  training process. [default k = 10, t = 10]

    Args:
        args:
    Return:

    """
    global device

    acc = []
    loss = []
    sen = []
    spe = []
    f1 = []
    auc = []
    
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    
    #* t times loop
    for i in range(args.times):
        args.seed = seeds[i]
        setup_seed(args.seed)
        dataset = MyOwnDataset(root="{}/{}".format(args.root, args.dataset), args=args)
    
        acc_iter = []
        loss_iter = []
        sen_iter = []
        spe_iter = []
        f1_iter = []
        auc_iter = []

        #* k-fold loop
        for k, (train_split, valid_split, test_split) in enumerate(zip(*cross_validate(args.folds, dataset))):
            train_subset, valid_subset, test_subset = dataset[train_split], \
                                                      dataset[valid_split], \
                                                      dataset[test_split]
            train_loader    = myDataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            val_loader      = myDataLoader(valid_subset, batch_size=args.batch_size, shuffle=False)
            test_loader     = myDataLoader(test_subset,  batch_size=args.batch_size, shuffle=False)

            ckpt_fc = None
            ckpt_sc = None
            mode    = args.mode
            if mode != "train":
                ## * Stage 1: train uni-modal encoder
                print('=' * 50)
                print("Uni-modal pretraining")
                ckpt_fc = unimodal_pretrain(args, train_loader, val_loader, test_loader, modal="fc")
                ckpt_sc = unimodal_pretrain(args, train_loader, val_loader, test_loader, modal="sc")
                # ckpt_sc, ckpt_fc = multimodal_pretrain(args, train_loader, val_loader, test_loader)
                print("Uni-modal pretraining End")
                print('=' * 50)
            
            ## * Stage 2: train Model
            model = Model(args)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = StepLR(optimizer, step_size=50, gamma=0.8)
            # model.fc_encoder.load_state_dict(ckpt_fc)
            # model.sc_encoder.load_state_dict(ckpt_sc)

            print('=' * 50)
            print("Multi-modal training")
            stage_old = args.stage
            if args.stage == "contrast":
                ## stage 2.1 contrastive training
                print("CLIP training start.")
                best_epoch,_ = train(args, 
                                   model, 
                                   train_loader, val_loader, test_loader, 
                                   optimizer, 
                                   scheduler, 
                                   loss_metrics=F.cross_entropy,
                                   uni_fc=ckpt_fc, 
                                   uni_sc=ckpt_sc,
                                   mode=mode,
                                   i_fold=k)
                print("CLIP training end.")
                
                # stage 2.2 classification training
                # ["fc_proj", "sc_proj", "fusion_layers"]
                frozen_names = ["fc_encoder", "sc_encoder"]
                for name, param in model.named_parameters():
                    if name in frozen_names:
                        param.requires_grad = False
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                scheduler = StepLR(optimizer, step_size=50, gamma=0.8)
                
                mode="train"
                args.stage = "ce"

            elif args.stage == "moe_pretrain":
                print("MoE pretraining. Using single expert first.")
                args.fusion = "MLP"
                model_pretrain = Model(args)
                model_pretrain.to(device)
                optimizer_pretrain = torch.optim.Adam(model_pretrain.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                scheduler_pretrain = StepLR(optimizer_pretrain, step_size=50, gamma=0.8)
                best_epoch, ckpt_pretrain = train(
                                                args, 
                                                model_pretrain, 
                                                train_loader, val_loader, test_loader, 
                                                optimizer_pretrain, 
                                                scheduler_pretrain, 
                                                loss_metrics=F.cross_entropy,
                                                uni_fc=ckpt_fc, 
                                                uni_sc=ckpt_sc,
                                                mode=mode,
                                                i_fold=k
                                            )
                model_pretrain.load_state_dict(ckpt_pretrain)
                print("MoE pretraining ended.")
                # load model parameters from pretrain model to fine-tune model
                frozen_names = ["fc_encoder", "sc_encoder", 
                                "fc_spec_proj", "sc_spec_proj", 
                                "fc_share_proj", "sc_share_proj"]
                model_dict = model.state_dict()
                model_pretrain_dict = model_pretrain.state_dict()
                filter_dict = {k: v for k, v in model_pretrain_dict.items() \
                                if k in model_dict \
                                and k.split('.')[0] in frozen_names}
                model_dict.update(filter_dict)
                model.load_state_dict(model_dict)
                
                # TODO update moe expert weights
                for id, layer in enumerate(model_pretrain.fc_layers):
                    model.fc_layers[id].copy_expert_weights(deepcopy(layer.state_dict()))
                for id, layer in enumerate(model_pretrain.sc_layers):
                    model.sc_layers[id].copy_expert_weights(deepcopy(layer.state_dict()))
                for id, layer in enumerate(model_pretrain.fc_layers):
                    model.fu_layers[id].copy_expert_weights(deepcopy(layer.state_dict()))

                # frozen model parameters except MoE
                for name, param in model.named_parameters():
                    if name in frozen_names:
                        param.requires_grad = False
                
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                scheduler = StepLR(optimizer, step_size=50, gamma=0.8)
                
                mode="train"
                args.stage = "mix"
                args.fusion = "MoE"

            ## * FINAL TRAINING PART
            best_epoch, ckpt = train(
                                args, 
                                model, 
                                train_loader, val_loader, test_loader, 
                                optimizer, 
                                scheduler, 
                                loss_metrics=F.cross_entropy,
                                uni_fc=ckpt_fc, 
                                uni_sc=ckpt_sc,
                                mode=mode,
                                i_fold=k
                            )

            args.stage = stage_old
            model.load_state_dict(ckpt)
            test_acc, test_loss, test_sen, test_spe, test_f1, test_auc,\
                y, pred,\
                moe_gates = eval_FCSC(args, model, test_loader)
            print("Multi-modal training ended")
            print('=' * 50)

            acc_iter.append(test_acc)
            loss_iter.append(test_loss)
            sen_iter.append(test_sen)
            spe_iter.append(test_spe)
            f1_iter.append(test_f1)
            auc_iter.append(test_auc)
            
            #* log for each fold 
            print('[{}] fold test set results, best_epoch = {:03d}  loss = {:.6f}, accuracy = {:.6f}, sensitivity = {:.6f}, '
                  'specificity = {:.6f}, f1_score = {:.6f}, auc_score = {:.6f}'.format(
                   k, best_epoch, test_loss, test_acc, test_sen, test_spe, test_f1, test_auc))
            with open(args.result_path, 'a+') as f:
                f.write("[{}] fold: best_epoch = {:03d}  loss = {:.6f}, \
                accuracy = {:.6f}, sensitivity = {:.6f}, specificity = {:.6f}, \
                f1_score = {:.6f}, auc_score = {:.6f}\n".format(
                   k, best_epoch, test_loss, test_acc, test_sen, test_spe, test_f1, test_auc))
                f.write("sc_gates:{}\nfc_gates{}\nfu_gates{}\n".format(
                    moe_gates[0], moe_gates[1], moe_gates[2]
                ))
        
        #* record average results for each time 
        acc.append(np.mean(acc_iter))
        sen.append(np.mean(sen_iter))
        spe.append(np.mean(spe_iter))
        f1.append(np.mean(f1_iter))
        auc.append(np.mean(auc_iter))
        print('---------------------------------------------------------------\n \
            [{}] times test set results, accuracy = {:.2f}±{:.2f}, sen = {:.2f}±{:.2f},'
            'spe = {:.2f}±{:.2f}, f1 = {:.2f}±{:.2f}, auc = {:.2f}±{:.2f}'.format(
            i,
            np.mean(acc_iter)*100,  np.std(acc_iter)*100, 
            np.mean(sen_iter)*100,  np.std(sen_iter)*100,
            np.mean(spe_iter)*100,  np.std(spe_iter)*100, 
            np.mean(f1_iter)*100,   np.std(f1_iter)*100, 
            np.mean(auc_iter)*100,  np.std(auc_iter)*100))
        with open(args.result_path, 'a+') as f:
            f.write("----------------------------{}-----------------------------------\n[{}] times,\
                seed: {:03d} AVERAGE acc: {:.2f} ± {:.2f}, sen: {:.2f} ± {:.2f}, \
                spe: {:.2f} ± {:.2f}, f1: {:.2f} ± {:.2f}, auc: {:.2f} ± {:.2f}\n\n".format(
                get_current_time(),
                i, 
                args.seed,
                np.mean(acc_iter)*100,  np.std(acc_iter)*100, 
                np.mean(sen_iter)*100,  np.std(sen_iter)*100,
                np.mean(spe_iter)*100,  np.std(spe_iter)*100, 
                np.mean(f1_iter)*100,   np.std(f1_iter)*100, 
                np.mean(auc_iter)*100,  np.std(auc_iter)*100))

    #* record final average results for ${t} times
    print(args)
    print('FINAL Average test set results: \
        mean accuracy = {:.6f}, std = {:.6f}, \
        mean_sen = {:.6f},  std_sen = {:.6f}, \
        mean_spe = {:.6f},  std_spe = {:.6f}, \
        mean_f1 = {:.6f},   std_f1 = {:.6f}, \
        mean_auc = {:.6f},  std_auc = {:.6f}'.format(
        np.mean(acc),   np.std(acc), 
        np.mean(sen),   np.std(sen),
        np.mean(spe),   np.std(spe), 
        np.mean(f1),    np.std(f1), 
        np.mean(auc),   np.std(auc)))
    with open(args.result_path, 'a+') as f:
        f.write("{} \t FINAL AVERAGE acc: {:.2f} ± {:.2f}, sen:{:.2f} ± {:.2f}, spe:{:.2f} ± {:.2f}, f1:{:.2f} ± {:.2f}, auc:{:.2f} ± {:.2f}\n \
            ---------------------------------------------------------------------------------------------------------------------".format(
            get_current_time(),
            np.mean(acc)*100, np.std(acc)*100, 
            np.mean(sen)*100, np.std(sen)*100,
            np.mean(spe)*100, np.std(spe)*100, 
            np.mean(f1)*100,  np.std(f1)*100, 
            np.mean(auc)*100, np.std(auc)*100))

