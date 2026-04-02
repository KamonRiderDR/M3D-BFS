import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


def sc_fc_contrastive_loss(args, anchor, sc_spe, fc_spe, sc_fc_share):
    """_summary_

    Args:
        args (_type_): _description_
        sc_spe (_type_): [B * d]
        fc_spe (_type_): [B * d]
        sc_fc_share (_type_): [B * d]
    """
    loss = 0.
    batch_size = anchor.shape[0]
    loss_s2o = infoNCE(sc_spe, anchor, torch.cat([fc_spe, sc_fc_share], dim=0).view(batch_size, 2, -1))
    loss_f2o = infoNCE(fc_spe, anchor, torch.cat([sc_spe, sc_fc_share], dim=0).view(batch_size, 2, -1))
    loss_sh2o = infoNCE(sc_fc_share, anchor, torch.cat([fc_spe, sc_spe], dim=0).view(batch_size, 2, -1))

    return loss_s2o + loss_f2o + loss_sh2o


def infoNCE(anchor, positive, negative):
    """_summary_

    Args:
        anchor (_type_): [B, d]
        positive (_type_): [B, d]
        negative (_type_): [B, 2, d]
    """
    tau = 0.1
    batch_size = anchor.shape[0]
    d = anchor.shape[1]
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negative = F.normalize(negative, dim=2)

    positive_similarity = torch.exp(torch.sum(anchor * positive, dim=1) / tau)
    negative_similarity = torch.exp(torch.sum(anchor.unsqueeze(1) * negative, dim=2) / tau)
    negative_similarity = negative_similarity.sum(dim=1)

    loss = -torch.log(positive_similarity / (positive_similarity + negative_similarity))

    return loss.mean()


