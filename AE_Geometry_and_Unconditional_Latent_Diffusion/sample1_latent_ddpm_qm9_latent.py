import argparse
import math
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from models_diffusion.ddpm import Model as DDPMModel
from models_diffusion.ddpm_utils import DDPMSampler
import pdb
import os, sys


class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = DDPMModel(args.dim_in, args.dim_hidden, args.num_layer, args.n_steps, args.beta_1, args.beta_T)
        self.args = args

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        loss = self.model.loss_fn(batch)
        self.log('train_loss', loss, batch_size=batch.shape[0], on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        batch = batch[0]
        batch_size, _ = batch.shape
        nll = 0
        for idx in range(batch_size):
            nll += self.model.neg_loglikelihood(batch[idx])
        nll /= batch_size
        self.log('test_nll', nll, batch_size=batch_size, on_epoch=True, sync_dist=True)

        loss = self.model.loss_fn(batch)
        self.log('test_loss', loss, batch_size=batch_size, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./logs/debug')

    parser.add_argument('--sample_number', type=int, default=100000)

    # number of nodes for parallel training
    parser.add_argument('--ddp_num_nodes', type=int, default=1)
    # number of devices in each node for parallel training
    parser.add_argument('--ddp_device', type=int, default=1)
    args = parser.parse_args()

    print(args)

    device = 'cuda'

    # model
    args2 = torch.load(args.log_dir + '/args.pt')
    model = Model.load_from_checkpoint(args.log_dir + '/checkpoint-best.ckpt', args=args2)

    # calculate nll
    data = np.load('../e3_diffusion_for_molecules/qm9/latent_diffusion/emb_2d_3d/test.npz')
    dataset_test = torch.utils.data.TensorDataset(torch.tensor( np.concatenate([data['emb_2d'], data['emb_3d']], axis=1) ))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args2.batch_size*8, shuffle=False, num_workers=4)
    trainer = pl.Trainer(max_epochs=1, gradient_clip_val=5, check_val_every_n_epoch=5, default_root_dir=args.log_dir, num_sanity_val_steps=10, accelerator='gpu', strategy='ddp', num_nodes=args.ddp_num_nodes, devices=args.ddp_device)
    # trainer.test(model, dataloader_test)

    model = model.model.to(device)
    model.eval()

    # 1. sample z~p(z)
    sampler = DDPMSampler(args2.beta_1, args2.beta_T, args2.n_steps, model, device, (args2.dim_in,))
    samples = sampler.sampling(args.sample_number, True).to('cpu')

    # clip to [-1, 1]
    samples[samples>1] = 1
    samples[samples<-1] = -1

    torch.save(samples, args.log_dir + '/sample_z.pt')

