# mostly from
# https://github.com/TencentYoutuResearch/HIG-GraphClassification/blob/92a861ce7a753cb397b169dfd1d7dab130d33cc9/Graphormer_with_HIG/graphormer/collator.py
# with a bit of modification 
# to add GCN layer on top of the original model

import ogb
import torch
import numpy as np
from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    #adj = torch.zeros([N, N], dtype=torch.int64)
    adj[edge_index[0, :], edge_index[1, :]] = 1 # 1 or True 

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]
                   ] = convert_to_single_emb(edge_attr) + 1

    shortest_path_result, path = floyd_warshall(adj.numpy()) # algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    # max_dist = 1
    edge_input = gen_edge_input(max_dist, path, attn_edge_type.numpy())
    # edge_input = algos.gen_edge_input(max_dist, adj.numpy(), attn_edge_type.numpy())
    rel_pos = torch.from_numpy((shortest_path_result)).long()
    # rel_pos = torch.from_numpy((adj.numpy())).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.rel_pos = rel_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()

    return item


class MyGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        super(MyGraphPropPredDataset, self).download()

    def process(self):
        super(MyGraphPropPredDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)