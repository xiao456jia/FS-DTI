import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import TAGConv
from dgl.nn.pytorch.glob import MaxPooling, GlobalAttentionPooling
from .base import *

__all__ = ['FAG']

@register_collate_func(dgl.batch)
@register_to_device(True)

class FAG(EncoderLayer):
    """Molecular Fragment Feature Extraction Model"""

    def __init__(self, device='cpu', max_fragments=11, output_feats=128, pooling=False):
        """
        初始化模型
        :param device:
        :param max_fragments:
        :param output_feats:
        :param pooling:
        """
        super(FAG, self).__init__()
        #
        self.convs = nn.ModuleList([
            TAGConv(8, 64, 2),  #
            TAGConv(64, 64, 2),
            TAGConv(64, 64, 2),
            TAGConv(64, 64, 2)
        ])

        self.pooling_ligand = nn.Linear(64, 1)
        self.pooling_protein = nn.Linear(64, 1)
        self.max_pool = MaxPooling()
        self.attn_pool = GlobalAttentionPooling(self.pooling_protein)

        self.transform = nn.Linear(128, output_feats)

        self.device = device
        self.max_fragments = max_fragments
        self.output_feats = output_feats
        self.pooling = pooling
        self.output_shape = output_feats if pooling else (max_fragments, output_feats)

    def get_output_shape(self):

        return self.output_shape

    def forward(self, g):
        """
        Forward
        :param g: Batch processed DGL molecular fragment graph
        :return: Feature representation
        """
        g = g.to(self.device)
        h = g.ndata['h']


        for conv in self.convs:
            h = F.relu(conv(g, h))


        max_pooled = self.max_pool(g,h)
        attn_pooled = self.attn_pool(g,h).view(-1, 64)
        combined = torch.cat([max_pooled, attn_pooled], dim=1)
        transformed = self.transform(combined)


        total_fragments = transformed.size(0)



        B = total_fragments // self.max_fragments
        fragments_rep_reshaped = transformed.view(B, self.max_fragments, -1)

        if self.pooling:

            fragments_rep_reshaped = torch.mean(fragments_rep_reshaped, dim=1)
            return fragments_rep_reshaped
        else:

            positional_embeddings = g.ndata['fragment_number']


            node_counts = g.batch_num_nodes().tolist()
            start_indices = [0] + [sum(node_counts[:i + 1]) for i in range(len(node_counts))]


            pos_emb_list = []
            for i in range(len(node_counts)):
                start = start_indices[i]

                if start < positional_embeddings.size(0):
                    pos_emb_list.append(positional_embeddings[start])
                else:
                    pos_emb_list.append(torch.tensor(-1, device=positional_embeddings.device))

            positional_embeddings = torch.stack(pos_emb_list)


            if positional_embeddings.size(0) < B * self.max_fragments:

                padding_size = B * self.max_fragments - positional_embeddings.size(0)
                padding = torch.full((padding_size,), -1,
                                     device=positional_embeddings.device)
                positional_embeddings = torch.cat([positional_embeddings, padding])

            positional_embeddings = positional_embeddings.view(B, self.max_fragments)
            return fragments_rep_reshaped, positional_embeddings


