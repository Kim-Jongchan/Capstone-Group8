# mostly from
# https://github.com/TencentYoutuResearch/HIG-GraphClassification/blob/92a861ce7a753cb397b169dfd1d7dab130d33cc9/Graphormer_with_HIG/graphormer/collator.py
# with a bit of modification 
# to add GCN layer on top of the original model

from collator import collator
from wrapper import MyGraphPropPredDataset

from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import ogb
import ogb.lsc
import ogb.graphproppred
import numpy as np
from functools import partial

# for AUC margin loss
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG

dataset = None

def get_dataset(dataset_name='abaaba'):
    global dataset
    if dataset is not None:
        return dataset
    # add AUC margin loss

    aucm_criterion = AUCMLoss()

    # max_node is set to max(max(num_val_graph_nodes), max(num_test_graph_nodes))

    dataset_name = 'ogbg-molhiv'
    dataset = {
        'num_class': 1,
        'loss_fn': aucm_criterion, #F.binary_cross_entropy_with_logits,
        'metric': 'rocauc',
        'metric_mode': 'max',
        'evaluator': ogb.graphproppred.Evaluator('ogbg-molhiv'),
        'dataset': MyGraphPropPredDataset('ogbg-molhiv', root='../dataset'),
        'max_node': 128,
    }

    print(f' > {dataset_name} loaded!')
    print(dataset)
    print(f' > dataset info ends')
    return dataset


class GraphDataModule(LightningDataModule):
    name = "OGB-GRAPH"

    def __init__(
        self,
        dataset_name: str = 'ogbg-molhiv',
        num_workers: int = 8,
        batch_size: int = 256,
        seed: int = 42,
        multi_hop_max_dist: int = 5,
        rel_pos_max: int = 1024,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.dataset = get_dataset(self.dataset_name)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...
        self.multi_hop_max_dist = multi_hop_max_dist
        self.rel_pos_max = rel_pos_max

    def setup(self, stage: str = None):

        split_idx = self.dataset['dataset'].get_idx_split()

        mgf_maccs_pred = np.load('../rf_preds_hiv/rf_final_pred.npy')
        self.dataset['dataset'].data.y = torch.cat((self.dataset['dataset'].data.y, torch.from_numpy(mgf_maccs_pred)), 1)
        # elif self.dataset_name == 'ogbg-molpcba':
        #     mgf_maccs_pred = np.load('../../rf_preds_pcba/rf_final_pred.npy')
        #     self.dataset['dataset'].data.y = torch.cat((self.dataset['dataset'].data.y, torch.from_numpy(mgf_maccs_pred)), 1)

        self.dataset_train = self.dataset['dataset'][split_idx["train"]]
        self.dataset_val = self.dataset['dataset'][split_idx["valid"]]
        self.dataset_test = self.dataset['dataset'][split_idx["test"]]

    def train_dataloader(self):
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, rel_pos_max=self.rel_pos_max),
        )
        print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, rel_pos_max=self.rel_pos_max),
        )
        print('len(val_dataloader)', len(loader))
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, rel_pos_max=self.rel_pos_max),
        )
        print('len(test_dataloader)', len(loader))
        return loader
