import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import rdkit
import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm

from hgraph import *
from hgraph.dataset import DatasetFromFolder, custom_collate

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

import pdb


class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = HierVAE(args)

        for param in self.model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

        self.beta = 0

        self.args = args

    def training_step(self, batch, batch_idx):
        total_step = self.current_epoch * self.args.dataset_size + batch_idx

        if total_step >= self.args.warmup and total_step % self.args.kl_anneal_iter == 0:
            self.beta = min(self.args.max_beta, self.beta + self.args.step_beta)       

        loss, kl_div, wacc, iacc, tacc, sacc = self.model.forward_with_gssl(*batch, beta=self.beta)

        self.log('loss', loss.item(), batch_size=self.args.batch_size, on_epoch=True, sync_dist=True)
        self.log('kl_div', kl_div, batch_size=self.args.batch_size, on_epoch=True, sync_dist=True)
        self.log('wacc', wacc.item(), batch_size=self.args.batch_size, on_epoch=True, sync_dist=True)
        self.log('iacc', iacc.item(), batch_size=self.args.batch_size, on_epoch=True, sync_dist=True)
        self.log('tacc', tacc.item(), batch_size=self.args.batch_size, on_epoch=True, sync_dist=True)
        self.log('sacc', sacc.item(), batch_size=self.args.batch_size, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, self.args.anneal_rate)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': self.args.anneal_iter} }



if __name__ == '__main__':

    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--atom_vocab', default=common_atom_vocab)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--seed', type=int, default=7)

    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--hidden_size', type=int, default=250)
    parser.add_argument('--embed_size', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--depthT', type=int, default=15)
    parser.add_argument('--depthG', type=int, default=15)
    parser.add_argument('--diterT', type=int, default=1)
    parser.add_argument('--diterG', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=5.0)
    parser.add_argument('--step_beta', type=float, default=0.001)
    parser.add_argument('--max_beta', type=float, default=1.0)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--kl_anneal_iter', type=int, default=2000)

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--anneal_rate', type=float, default=0.9)
    parser.add_argument('--anneal_iter', type=int, default=25000)
    parser.add_argument('--print_iter', type=int, default=50)
    parser.add_argument('--save_iter', type=int, default=5000)

    parser.add_argument('--load_checkpoint_for_continual_training', type=str, default=None)

    # number of nodes for parallel training
    parser.add_argument('--ddp_num_nodes', type=int, default=1)
    # number of devices in each node for parallel training
    parser.add_argument('--ddp_device', type=int, default=1)

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
    args.vocab = PairVocab(vocab)

    dataset = DatasetFromFolder(args.train)
    args.dataset_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers=4)

    model = Model(args)

    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir, save_last=True, filename='checkpoint-best', monitor='loss')
    trainer = pl.Trainer(max_epochs=args.epoch, gradient_clip_val=5, check_val_every_n_epoch=1, default_root_dir=args.save_dir, callbacks=[checkpoint_callback], num_sanity_val_steps=10,
                         accelerator='gpu', strategy='ddp', num_nodes=args.ddp_num_nodes, devices=args.ddp_device)
    trainer.fit(model, dataloader, ckpt_path=args.load_checkpoint_for_continual_training)

