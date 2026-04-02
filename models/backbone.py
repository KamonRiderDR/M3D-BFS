import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv


def fc(in_features, out_features, dropout):
    return nn.Sequential(nn.BatchNorm1d(in_features),
                         nn.ReLU(),
                         nn.Dropout(dropout),
                         nn.Linear(in_features, out_features))

pooling_functions = {
    "add":  global_add_pool,
    "mean": global_mean_pool,
    "max":  global_max_pool,
}


class FC(nn.Module):
    def __init__(self, input, output, dropout, norm=False):
        super(FC, self).__init__()
        self.dropout = dropout
        self.bn1 = nn.BatchNorm1d(input)
        self.fc = nn.Linear(input, output)
        self.norm = None if norm is False else nn.BatchNorm1d(output)
    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.norm is None:
            return self.fc(x)
        else:
            return self.norm(self.fc(x))

class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.features = args.in_size
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        
        self.conv1 = GCNConv(self.features, self.hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
        self.bn = nn.BatchNorm1d(self.hidden_dim)

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

    def forward(self, data, name=None):
        """_summary_

        Args:
            data (_type_): _description_
        Returns:
            x: tensor format with [N * hidden]
        """
        x = getattr(data, f"{name}_x") if name is not None else data.x
        edge_index = getattr(data, f"{name}_edge_index") if name is not None else data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn(x)
        
        return x

    def __repr__(self):
        return self.__class__.__name__


class GIN(torch.nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        self.args           = args
        self.features       = args.in_size
        self.hidden_dim     = args.hidden_dim
        self.num_layers     = args.num_layers
        self.num_classes    = args.num_classes
        self.dropout        = args.dropout
        
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(self.features, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_dim),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(self.hidden_dim),
                    ), train_eps=True))

    def forward(self, data, name=None):
        x = getattr(data, f"{name}_x") if name is not None else data.x
        edge_index = getattr(data, f"{name}_edge_index") if name is not None else data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        return x

    def __repr__(self):
        return self.__class__.__name__


class MLP(torch.nn.Module):
    def __init__(self, in_features, hidden, out_features, n_layers, dropout):
        super(MLP, self).__init__()
        self.mlp = nn.ModuleList()
        if n_layers > 1:
            self.mlp.append(nn.Linear(in_features, hidden))
            for _ in range(n_layers - 2):
                self.mlp.append(fc(hidden, hidden, dropout=dropout))
            self.mlp.append(fc(hidden, out_features, dropout=dropout))
        else:
            self.mlp.append(nn.Linear(in_features, out_features))

    def forward(self, x, edge_index=None):
        for layer in self.mlp:
            x = layer(x)
        return x


"""_summary_

"""
class ModuleAE(nn.Module):
    def __init__(self, args, encoder, classifier, decoder=None):
        """_summary_

        Args:
            args: 
            encoder:    default: sc-fc GCN
            classifier: default: MLP
            decoder:    default: MLP
        """
        super(ModuleAE, self).__init__()
        self.args       = args
        self.encoder    = encoder 
        self.classifier = classifier
        self.decoder    = decoder if decoder is not None else self.classifier
        self.bn         = nn.BatchNorm1d(self.args.hidden_dim)

    def forward(self, data, params):
        """ [encoder => classifier](data)
        Args:
            data: `PYG` format
            params: parameter `dict` format:{name: value}
        Return:
            z_: [B * num_classes]
            z_: [B * hidden]
        """
        #? default [SC&&FC] => GCN
        modal       = "modal"
        device      = torch.device('cuda:{}'.format(self.args.device) if torch.cuda.is_available() else 'cpu')
        batch_size  = data.batch.max().item() + 1
        batch       = torch.tensor([[i for j in range(90 * 1)] for i in range(batch_size)], dtype=torch.long).view(-1).to(device)

        z = self.encoder(data, params[modal])
        # z = F.dropout(z, p=self.args.dropout, training=self.training)
        # z = self.bn(z)
        z = pooling_functions[params["pooling"]](z, batch)
        return self.classifier(z), z


""" Multi-modal Brain Contrastive Loss
"""
class MBCP(nn.Module):
    def __init__(self, args, encoder_fc, encoder_sc):
        """_summary_

        Args:
            args (_type_): _description_
            encoder_fc (_type_): [B * hidden]
            encoder_sc (_type_): [B * hidden]
        """
        super(MBCP, self).__init__()
        self.args           = args
        self.encoder_fc     = encoder_fc
        self.encoder_sc     = encoder_sc
        self.logit_scale    = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.device         = torch.device('cuda:{}'.format(self.args.device) if torch.cuda.is_available() else 'cpu')

    def forward(self, data):
        sc_feat = self.encoder_sc(data, name="sc")
        fc_feat = self.encoder_fc(data, name="fc")
        sc_feat = F.normalize(sc_feat, dim=-1)
        fc_feat = F.normalize(fc_feat, dim=-1)
        
        return self.mbcp_loss(sc_feat, fc_feat)

    def mbcp_loss(self, sc_feat, fc_feat):
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