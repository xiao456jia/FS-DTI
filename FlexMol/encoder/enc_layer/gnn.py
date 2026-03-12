import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

from .base import *
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax

__all__ = ['DGL_FragmentGNN']


@register_collate_func(dgl.batch)
@register_to_device(True)
class DGL_FragmentGNN(EncoderLayer):
    """
    DGL_FragmentGNN is a graph neural network implementation using DGL's GraphConv
    and WeightedSumAndMax for fragment-based molecular representation.
    """

    def __init__(self, in_feats=8, h_feats=64, gnn_out_feats=32,
                 output_feats=64, device='cpu', max_nodes=50, readout=True):
        """
        Initialize the DGL_FragmentGNN model.

        Parameters:
            in_feats (int): Number of input features per node
            h_feats (int): Hidden feature size
            gnn_out_feats (int): Output feature size of GNN layers
            output_feats (int): Final output feature size
            device (str): Device to run the model on ('cpu' or 'cuda')
            max_nodes (int): Maximum number of nodes per graph
            readout (bool): Whether to use readout for graph-level features
        """
        super(DGL_FragmentGNN, self).__init__()
        self.device = device
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, gnn_out_feats)

        if readout:
            self.readout = WeightedSumAndMax(gnn_out_feats)
            self.output_shape = output_feats
            self.transform = nn.Linear(gnn_out_feats * 2, output_feats)
        else:
            self.readout = None
            self.output_shape = (max_nodes, output_feats)
            self.transform = nn.Linear(gnn_out_feats, output_feats)

        self.gnn_out_feats = gnn_out_feats

    def get_output_shape(self):
        return self.output_shape

    @staticmethod
    def default_config(task, method):
        config_map = {
            "fragment": {
                "FragmentGNN": {"in_feats": 8, "h_feats": 64, "gnn_out_feats": 32}
            }
        }
        task_config = config_map.get(task, {})
        return task_config.get(method, {})

    def forward(self, bg):
        bg = bg.to(self.device)
        # Ensure graph has self-loops
        bg = dgl.add_self_loop(bg)

        # Get node features
        feats = bg.ndata['feat'].float()

        # GNN layers
        h = self.conv1(bg, feats)
        h = F.relu(h)
        node_feats = self.conv2(bg, h)

        # Readout or node feature processing
        if self.readout:
            return self.transform(self.readout(bg, node_feats))
        else:
            batch_size = bg.batch_size
            node_feats = self.transform(node_feats)
            return node_feats.view(batch_size, -1, self.output_shape[1])