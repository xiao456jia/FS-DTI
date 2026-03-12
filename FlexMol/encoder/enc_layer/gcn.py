import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from .base import *
from dgllife.model.gnn.gcn import GCN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax

__all__ = ['DGL_GCN', 'DGL_GCN_Chemberta','DGL_FragmentGCN']

#adapted from https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/gnn/gcn.py

@register_collate_func(dgl.batch)
@register_to_device(True)
class DGL_GCN(EncoderLayer):
    """
    DGL_GCN is a graph convolutional network implementation using DGL's GCN and WeightedSumAndMax.
    """

    def __init__(self, in_feats=74, hidden_feats=[64, 64, 64], activation=[F.relu, F.relu, F.relu], output_feats=64, device='cpu', max_nodes=50, readout=True):
        """
        Initialize the DGL_GCN model.

        Parameters:
            in_feats (int): Number of input features.
            hidden_feats (list): List of hidden feature sizes for each GCN layer.
            activation (list): List of activation functions for each GCN layer.
            output_feats (int): Number of output features.
            device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
            max_nodes (int): Maximum number of nodes in the graph.
            readout (bool): Whether to use a readout layer.
        """
        super(DGL_GCN, self).__init__()
        self.device = device
        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       activation=activation)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        if readout:
            self.readout = WeightedSumAndMax(gnn_out_feats)
            self.output_shape = output_feats
            self.transform = nn.Linear(gnn_out_feats * 2, output_feats)
        else:
            self.readout = None
            self.output_shape = (max_nodes, output_feats)
            self.transform = nn.Linear(gnn_out_feats, output_feats)

    def get_output_shape(self):
        return self.output_shape

    @staticmethod
    def default_config(task, method):
        config_map = {
            "drug": {
                "GCN": {"in_feats": 74}
            },
            "prot_3d": {
                "GCN": {"in_feats": 25},
                "GCN_ESM": {"in_feats": 1305}
            }
        }
    
        task_config = config_map.get(task, {})
        return task_config.get(method, {})

    def forward(self, bg):
        bg = bg.to(self.device)
        feats = bg.ndata.pop('h')
        feats = feats.to(torch.float32)
        node_feats = self.gnn(bg, feats)
        if self.readout:
            return self.transform(self.readout(bg, node_feats))
        else:
            batch_size = bg.batch_size
            node_feats = self.transform(node_feats)
            return node_feats.view(batch_size, -1, self.output_shape[1])


@register_collate_func(dgl.batch)
@register_to_device(True)
class DGL_GCN_Chemberta(EncoderLayer):
    """
    DGL_GCN is a graph convolutional network implementation using DGL's GCN and WeightedSumAndMax.
    """

    def __init__(self, in_feats=74, hidden_feats=[64, 64, 64], activation=[F.relu, F.relu, F.relu], output_feats=64, device='cpu', max_nodes=50, readout=True):
        """
        Initialize the DGL_GCN model.

        Parameters:
            in_feats (int): Number of input features.
            hidden_feats (list): List of hidden feature sizes for each GCN layer.
            activation (list): List of activation functions for each GCN layer.
            output_feats (int): Number of output features.
            device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
            max_nodes (int): Maximum number of nodes in the graph.
            readout (bool): Whether to use a readout layer.
        """
        super(DGL_GCN_Chemberta, self).__init__()
        self.device = device
        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       activation=activation)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        if readout:
            self.readout = WeightedSumAndMax(gnn_out_feats)
            self.output_shape = output_feats
            self.transform = nn.Linear(gnn_out_feats * 2, output_feats)
        else:
            self.readout = None
            self.output_shape = (max_nodes, output_feats)
            self.transform = nn.Linear(gnn_out_feats, output_feats)
        self.dim_reduction_layer = nn.Linear(384, 54)

    def get_output_shape(self):
        return self.output_shape

    @staticmethod
    def default_config(task, method):
        config_map = {
            "drug": {
                "GCN": {"in_feats": 74},
                "GCN_Chemberta": {"in_feats": 128}
            },
            "prot_3d": {
                "GCN": {"in_feats": 25},
                "GCN_ESM": {"in_feats": 1305}
            }
        }
    
        task_config = config_map.get(task, {})
        return task_config.get(method, {})

    def forward(self, bg):
        bg = bg.to(self.device)
        feats = bg.ndata.pop('h')
        feats_bert = bg.ndata.pop('bert')
        feats = feats.to(torch.float32)
        feats_bert = feats_bert.to(torch.float32)

        feats_bert = self.dim_reduction_layer(feats_bert)
        feats = torch.cat((feats, feats_bert), dim=1)

        node_feats = self.gnn(bg, feats)
        if self.readout:
            return self.transform(self.readout(bg, node_feats))
        else:
            batch_size = bg.batch_size
            node_feats = self.transform(node_feats)
            return node_feats.view(batch_size, -1, self.output_shape[1])


@register_collate_func(dgl.batch)
@register_to_device(True)
class DGL_FragmentGCN(EncoderLayer):
    def __init__(self, in_feats=8, hidden_feats=[64, 64, 64], activation=[F.relu, F.relu, F.relu], output_feats=128,
                 device='cpu', max_nodes=50, readout=True):
        super(DGL_FragmentGCN, self).__init__()
        self.device = device
        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       activation=activation)
        gnn_out_feats = self.gnn.hidden_feats[-1]


        self.frag_aggregate = nn.Sequential(
            nn.Linear(gnn_out_feats, 128),
            nn.ReLU(),
            nn.Linear(128, output_feats)
        )

        if readout:
            self.readout = WeightedSumAndMax(gnn_out_feats)
            self.output_shape = output_feats
            self.transform = nn.Sequential(
                nn.Linear(gnn_out_feats * 2, 128),
                nn.ReLU(),
                nn.Linear(128, output_feats)
            )
        else:
            self.readout = None
            self.output_shape = (max_nodes, output_feats)
            self.transform = nn.Sequential(
                nn.Linear(gnn_out_feats, 128),
                nn.ReLU(),
                nn.Linear(128, output_feats)
            )


        self.frag_id_key = 'frag_id'

    def get_output_shape(self):
        return self.output_shape

    @staticmethod
    def default_config(task, method):
        config_map = {
            "fragment": {
                "FragmentGNN": {"in_feats": 8}
            }
        }

        task_config = config_map.get(task, {})
        return task_config.get(method, {})

    def forward(self, bg):
        bg = bg.to(self.device)


        frag_ids = None
        if self.frag_id_key in bg.ndata:
            frag_ids = bg.ndata.pop(self.frag_id_key)


        if 'feat' in bg.ndata:
            feats = bg.ndata.pop('feat')
        else:
            feats = bg.ndata.pop('h')

        feats = feats.to(torch.float32)
        bg = dgl.add_self_loop(bg)
        node_feats = self.gnn(bg, feats)


        frag_features = []
        if frag_ids is not None:
            unique_frag_ids = torch.unique(frag_ids)
            for frag_id in unique_frag_ids:
                if frag_id == -1:  # 跳过虚拟碎片
                    continue
                mask = (frag_ids == frag_id)
                frag_feat = node_feats[mask]
                frag_feat = torch.mean(frag_feat, dim=0)
                frag_features.append(frag_feat)


        if frag_features:
            frag_features = torch.stack(frag_features)
            frag_features = self.frag_aggregate(frag_features)

            frag_global = torch.mean(frag_features, dim=0)
        else:

            frag_global = torch.zeros(self.output_feats, device=self.device)


        if frag_ids is not None:
            bg.ndata[self.frag_id_key] = frag_ids

        if self.readout:
            global_feat = self.transform(self.readout(bg, node_feats))

            frag_global = frag_global.unsqueeze(0)
            combined = global_feat + frag_global
            return combined
        else:
            batch_size = bg.batch_size
            node_feats = self.transform(node_feats)
            return node_feats.view(batch_size, -1, self.output_shape[1])
