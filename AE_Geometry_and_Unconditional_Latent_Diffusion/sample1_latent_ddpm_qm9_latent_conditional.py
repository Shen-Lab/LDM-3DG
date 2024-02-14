import argparse
import math
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from models_diffusion.ddpm_conditional import Model as DDPMModel
from models_diffusion.ddpm_utils_conditional import DDPMSampler
import pdb
import os, sys


class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = DDPMModel(args.dim_in, args.dim_condition, args.dim_hidden, args.num_layer, args.n_steps, args.beta_1, args.beta_T)
        self.args = args

    def training_step(self, batch, batch_idx):
        batch, batch_condition = batch
        loss = self.model.loss_fn(batch, batch_condition)
        self.log('train_loss', loss, batch_size=batch.shape[0], on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch, batch_condition = batch
        batch_size, _ = batch.shape

        loss = self.model.loss_fn(batch, batch_condition)
        self.log('val_loss', loss, batch_size=batch_size, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./logs/debug')
    parser.add_argument('--condition', type=str, default='alpha')

    parser.add_argument('--data_dir', type=str, default='/scratch/user/yuning.you/project/graph_latent_diffusion/e3_diffusion_for_molecules/qm9/latent_diffusion/emb_2d_3d_4layer_new/')

    parser.add_argument('--sample_number', type=int, default=100000)
    parser.add_argument('--dim_condition', type=int, default=16)

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

    # load data
    data = np.load( os.path.join(args.data_dir, 'valid.npz') )
    cond_mean = data[args.condition].mean()
    cond_mad = np.abs(data[args.condition] - cond_mean).mean() * 10

    data = np.load( os.path.join(args.data_dir, 'train.npz') )
    np.random.seed(42)
    num_data = data['emb_2d'].shape[0]
    idx_perm = np.random.permutation(num_data)
    idx_train = idx_perm[num_data//2:]
    idx_holdout = idx_perm[:num_data//2]
    condition_train = torch.tensor((data[args.condition] - cond_mean) / cond_mad)[idx_train]
    cond_max, cond_min = condition_train.max().item(), condition_train.min().item()
    print(cond_max, cond_min, cond_mean, cond_mad)

    # condition_holdout = torch.tensor((data[args.condition] - cond_mean) / cond_mad)[idx_holdout]
    # cond_holdout_max, cond_holdout_min = condition_holdout.max().item(), condition_holdout.min().item()
    # print(cond_holdout_max, cond_holdout_min)
    # assert False

    # sampling number = 100 * 1000 * 3 = 300000, ood distribution for the last two
    # condition = torch.tensor( np.concatenate([np.linspace(cond_min, cond_max, 100) for _ in range(1000)]
    #                                        + [np.linspace(cond_min * 1.5, cond_min, 100) for _ in range(1000)]
    #                                        + [np.linspace(cond_max, cond_max * 1.5, 100) for _ in range(1000)] ), dtype=torch.float32 ).to(device)
    condition = torch.tensor( np.concatenate([np.linspace(cond_min, cond_max, 100) for _ in range(10)]
                                           + [np.linspace(cond_min * 1.5, cond_min, 100) for _ in range(10)]
                                           + [np.linspace(cond_max, cond_max * 1.5, 100) for _ in range(10)] ), dtype=torch.float32 ).to(device)
    # print(condition)

    model = model.model.to(device)
    model.eval()

    # 1. sample z~p(z)
    sampler = DDPMSampler(args2.beta_1, args2.beta_T, args2.n_steps, model, device, (args2.dim_in,))
    samples = sampler.sampling(condition, True).to('cpu')

    # clip to [-1, 1]
    samples[samples>1] = 1
    samples[samples<-1] = -1

    torch.save(samples, args.log_dir + '/sample_z.pt')
    torch.save(condition.cpu() * cond_mad + cond_mean, args.log_dir + '/condition.pt')
