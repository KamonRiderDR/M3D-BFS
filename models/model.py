import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv

from .backbone import *
from .MoE import MoE


fusion_backbone = {
    "MLP":  nn.Linear,
    "MoE":  MoE 
}

pooling_functions = {
    "add":  global_add_pool,
    "mean": global_mean_pool,
    "max":  global_max_pool,
}

GNN_factory = {
    "GCN":  GCN,
    "GIN":  GIN
}



def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        #* parameters init
        self.args               = args
        self.hidden             = args.hidden_dim
        self.fusion_hidden      = args.fusion_hidden
        self.num_fusion_layers  = args.num_fusion_layers
        self.num_classes        = args.num_classes
        self.dropout            = args.dropout
        self.num_experts        = args.num_experts
        self.k                  = args.k                    # moe heads
        self.batch_size         = args.batch_size

        #* fc/sc encoder init
        self.fc_encoder = GNN_factory[args.conv](args) # [N * hidden]
        self.sc_encoder = GNN_factory[args.conv](args)
        self.bn_enc     = nn.BatchNorm1d(self.hidden)

        self.logit_scale    = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.device         = torch.device('cuda:{}'.format(self.args.device) if torch.cuda.is_available() else 'cpu')

        #* fc/sc projection for fusion
        self.fc_spec_proj   = nn.Linear(self.hidden, self.fusion_hidden)  # hidden -> fusion_hidden
        self.sc_spec_proj   = nn.Linear(self.hidden, self.fusion_hidden)
        self.fc_share_proj  = nn.Linear(self.hidden, self.fusion_hidden)
        self.sc_share_proj  = nn.Linear(self.hidden, self.fusion_hidden)
        self.bn_proj        = nn.BatchNorm1d(self.fusion_hidden)
        
        #* fusion layer(s) init
        self.fc_layers      = nn.ModuleList()
        self.sc_layers      = nn.ModuleList()
        self.fu_layers      = nn.ModuleList()
        self.bns            = nn.ModuleList()
        self.init_fusion_encoders(args)

        #* FINAL projection and classification head
        self.anchor_encoder = nn.Linear(self.fusion_hidden * 3, self.fusion_hidden)
        self.classifier     = nn.Linear(self.fusion_hidden, self.num_classes)

        self.sc_gates = []
        self.fc_gates = []
        self.fu_gates = []

    def init_fusion_encoders(self, args):
        for i in range(self.num_fusion_layers):
            sc_encoder = FC(self.fusion_hidden, self.fusion_hidden, dropout=args.dropout, norm=True)
            fc_encoder = FC(self.fusion_hidden, self.fusion_hidden, dropout=args.dropout, norm=True)
            fu_encoder = FC(self.fusion_hidden, self.fusion_hidden, dropout=args.dropout, norm=True)

            if args.fusion == "MoE":
                self.fc_layers.append(MoE(input_dim=self.fusion_hidden, output_dim=self.fusion_hidden, num_experts=args.num_experts, k=1, expert_encoder=fc_encoder))
                self.sc_layers.append(MoE(input_dim=self.fusion_hidden, output_dim=self.fusion_hidden, num_experts=args.num_experts, k=1, expert_encoder=sc_encoder))
                self.fu_layers.append(MoE(input_dim=self.fusion_hidden, output_dim=self.fusion_hidden, num_experts=args.num_experts, k=1, expert_encoder=fu_encoder, num_nodes=180))
            else:
                self.fc_layers.append(fc_encoder)
                self.sc_layers.append(sc_encoder)
                self.fu_layers.append(fu_encoder)
            self.bns.append(nn.BatchNorm1d(self.fusion_hidden))

    def clip_loss(self, sc_x, fc_x):
        """_summary_

        Args:
            sc_x (_type_): _description_
            fc_x (_type_): _description_
        """
        sc_x = F.normalize(sc_x, dim=-1)
        fc_x = F.normalize(fc_x, dim=-1)
        tau = 0.01
        logits = (sc_x @ fc_x.T) / tau
        sc_similarity = sc_x @ sc_x.T
        fc_similarity = fc_x @ fc_x.T
        targets = F.softmax(
            (sc_similarity + fc_similarity) / 2 * tau, dim=-1
        )
        sc_loss = cross_entropy(logits, targets, reduction='none')
        fc_loss = cross_entropy(logits, targets, reduction='none')
        loss = (sc_loss + fc_loss) / 2.0
        return loss.mean()

    def mbcp_loss(self, sc_feat, fc_feat):
        """MBCP Loss based on clip-loss
        Args:
            sc_feat (_type_): _description_
            fc_feat (_type_): _description_
        Returns:
            mbcp_loss.mean()
        """
        sc_feat = F.normalize(sc_feat, dim=-1)
        fc_feat = F.normalize(fc_feat, dim=-1)
        logits_per_sc = self.logit_scale * sc_feat @ fc_feat.T
        logits_per_fc = self.logit_scale * fc_feat @ sc_feat.T
        labels = torch.arange(logits_per_fc.shape[0], device=self.device, dtype=torch.long)
        
        loss = (
            F.cross_entropy(logits_per_fc, labels) + 
            F.cross_entropy(logits_per_sc, labels)
        ) / 2
        
        return loss.mean()

    def forward(self, data):
        """_summary_

        Args:
            data (_type_): graph data type. 

        Return:
            cls_x: [1 * num_classes]
            middle_features: 
                [sc_x, fc_x,
                 sc_spec_x, fc_spec_x, fusion_x, 
                 anchor_x], [B * f_hidden]
            
            loss function: [MoE, CLIP]
        """
        #* Init data structure
        device = torch.device('cuda:{}'.format(self.args.device) if torch.cuda.is_available() else 'cpu')
        batch_size  = data.batch.max().item() + 1
        batch_spe   = torch.tensor([[i for j in range(90 * 1)] for i in range(batch_size)], dtype=torch.long).view(-1).to(device)
        batch_share = torch.tensor([[i for j in range(90 * 2)] for i in range(batch_size)], dtype=torch.long).view(-1).to(device)
        batch_cls   = torch.tensor([[i for j in range(3)]      for i in range(batch_size)], dtype=torch.long).view(-1).to(device)
        
        # init gating data structure
        self.sc_gates = []
        self.fc_gates = []
        self.fu_gates = []
        
        #* sc/fc embedding
        sc_x = self.sc_encoder(data, name="sc")
        fc_x = self.fc_encoder(data, name="fc")

        # multimodal projection [sc/fc] * [unique/shared]
        sc_spec_proj_ = F.relu(self.sc_spec_proj(sc_x))                                            
        fc_spec_proj_ = F.relu(self.fc_spec_proj(fc_x))
        sc_share_proj_ = F.relu(self.sc_share_proj(sc_x))
        fc_share_proj_ = F.relu(self.fc_share_proj(fc_x))
        sc_spec_proj_, fc_spec_proj_, sc_share_proj_, fc_share_proj_ =  self.bn_proj(sc_spec_proj_),\
                                                                        self.bn_proj(fc_spec_proj_),\
                                                                        self.bn_proj(sc_share_proj_),\
                                                                        self.bn_proj(fc_share_proj_)
        sc_spec_x_ = sc_spec_proj_
        fc_spec_x_ = fc_spec_proj_
        fusion_x_ = torch.cat((sc_share_proj_, fc_share_proj_), dim=0)
        loss_mbcp = self.mbcp_loss(sc_feat=global_add_pool(sc_spec_x_, batch_spe), 
                                   fc_feat=global_add_pool(fc_spec_x_, batch_spe))

        #* fusion
        loss_moe = 0.
        for i in range(self.num_fusion_layers):
            cur_sc = sc_spec_x_
            cur_fc = fc_spec_x_
            cur_sh = fusion_x_
            if self.args.fusion == "MoE":
                sc_spec_x_, loss_moe_sc_ = self.sc_layers[i](sc_spec_x_)
                fc_spec_x_, loss_moe_fc_ = self.fc_layers[i](fc_spec_x_)
                fusion_x_, loss_moe_fu_  = self.fu_layers[i](fusion_x_)
                loss_moe = loss_moe + \
                            (loss_moe_sc_ + loss_moe_fc_ + loss_moe_fu_) / 3
                self.sc_gates.append(self.sc_layers[i].gates.detach().cpu().numpy())
                self.fc_gates.append(self.fc_layers[i].gates.detach().cpu().numpy())
                self.fu_gates.append([
                                self.fu_layers[i].gates[0].detach().cpu().numpy(),
                                self.fu_layers[i].gates[1].detach().cpu().numpy()
                            ])

            else:
                sc_spec_x_  = self.sc_layers[i](sc_spec_x_)
                fc_spec_x_  = self.fc_layers[i](fc_spec_x_)
                fusion_x_   = self.fu_layers[i](fusion_x_)                

            sc_spec_x_ = torch.add(cur_sc, sc_spec_x_)
            fc_spec_x_ = torch.add(cur_fc, fc_spec_x_)
            fusion_x_  = torch.add(cur_sh, fusion_x_)
            
            #? normalization
            sc_spec_x_, fc_spec_x_, fusion_x_ = F.dropout(sc_spec_x_,   p=self.dropout, training=self.training),\
                                                F.dropout(fc_spec_x_,   p=self.dropout, training=self.training),\
                                                F.dropout(fusion_x_,    p=self.dropout, training=self.training)
            sc_spec_x_, fc_spec_x_, fusion_x_ = self.bns[i](sc_spec_x_),\
                                                self.bns[i](fc_spec_x_),\
                                                self.bns[i](fusion_x_)

        #* pooling
        pooling     = pooling_functions[self.args.pooling]
        sc_spec_x   = pooling(sc_spec_x_, batch_spe)
        fc_spec_x   = pooling(fc_spec_x_, batch_spe)
        fusion_x    = pooling(fusion_x_, batch_share)

        #* proj anchor embedding
        cls_x = torch.cat((sc_spec_x, fc_spec_x, fusion_x), dim=1)
        anchor_x = F.relu(self.anchor_encoder(cls_x))
        cls = self.classifier(anchor_x)


        return cls, \
               [sc_x, fc_x, sc_spec_x, fc_spec_x, fusion_x, anchor_x], \
               [loss_moe, loss_mbcp]



    def get_fusion_weights(self):
        """ Get fusion layer weights.

        Returns:
            [fc, sc, fusion]
        """
        
        fc_weights = copy.deepcopy(self.fc_layers.state_dict())
        sc_weights = copy.deepcopy(self.sc_layers.state_dict())
        fu_weights = copy.deepcopy(self.fu_layers.state_dict())
        print(fc_weights.keys())
        return fc_weights, sc_weights, fu_weights


    def update_moe_fusion_weights(self, fc_weights, sc_weights, fu_weights):
        """ Update fusion weights for MoE.

        Args:
            fc_weights (_type_): _description_
            sc_weights (_type_): _description_
            fu_weights (_type_): _description_
        """
        for i in range(self.num_fusion_layers):
            self.fc_layers[i].copy_expert_weights(fc_weights[i])
            self.sc_layers[i].copy_expert_weights(sc_weights[i])
            self.fu_layers[i].copy_expert_weights(fu_weights[i])