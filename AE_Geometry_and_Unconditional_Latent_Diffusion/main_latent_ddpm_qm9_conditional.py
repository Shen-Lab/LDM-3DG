import argparse
import math
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from models_diffusion.ddpm_conditional import Model as DDPMModel
import pdb
import os


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
        '''
        nll = 0
        for idx in range(batch_size):
            nll += self.model.neg_loglikelihood(batch[idx])
        nll /= batch_size
        self.log('val_nll', nll, batch_size=batch_size, on_epoch=True, sync_dist=True)
        '''

        loss = self.model.loss_fn(batch, batch_condition)
        self.log('val_loss', loss, batch_size=batch_size, on_epoch=True, sync_dist=True)

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
    # parser.add_argument('--data_dir', type=str, default='/scratch/user/yuning.you/project/graph_latent_diffusion/e3_diffusion_for_molecules/qm9/latent_diffusion/emb_2d_3d/')
    parser.add_argument('--data_dir', type=str, default='/scratch/user/yuning.you/project/graph_latent_diffusion/e3_diffusion_for_molecules/qm9/latent_diffusion/emb_2d_3d_4layer_new/')

    parser.add_argument('--log_dir', type=str, default='./logs/debug')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epoch_num', type=int, default=10000)

    parser.add_argument('--learning_rate', type=float, default=1e-5) # to be tuned
    parser.add_argument('--dim_hidden', type=int, default=2048)
    parser.add_argument('--num_layer', type=int, default=32)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    parser.add_argument('--n_steps', type=int, default=500)

    # condition from {alpha, gap, homo, lumo, mu, Cv}
    parser.add_argument('--condition', type=str, default='alpha')
    parser.add_argument('--dim_condition', type=int, default=16)

    # continual training
    parser.add_argument('--load_checkpoint_for_warm_start', type=str, default=None)
    parser.add_argument('--load_checkpoint_for_continual_training', type=str, default=None)

    # number of nodes for parallel training
    parser.add_argument('--ddp_num_nodes', type=int, default=1)
    # number of devices in each node for parallel training
    parser.add_argument('--ddp_device', type=int, default=1)
    args = parser.parse_args()


    # dataset and dataloader
    data = np.load( os.path.join(args.data_dir, 'valid.npz') )
    cond_mean = data[args.condition].mean()
    cond_mad = np.abs(data[args.condition] - cond_mean).mean() * 10
    print(cond_mean, cond_mad) # ; assert False
    dataset_val = torch.utils.data.TensorDataset(torch.tensor( np.concatenate([data['emb_2d'], data['emb_3d']], axis=1) ), torch.tensor((data[args.condition] - cond_mean) / cond_mad) )
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size*8, shuffle=False, num_workers=4)

    data = np.load( os.path.join(args.data_dir, 'train.npz') )
    np.random.seed(42)
    num_data = data['emb_2d'].shape[0]
    idx_train = np.random.permutation(num_data)[num_data//2:]
    dataset_train = torch.utils.data.TensorDataset(torch.tensor( np.concatenate([data['emb_2d'], data['emb_3d']], axis=1) )[idx_train], torch.tensor((data[args.condition] - cond_mean) / cond_mad)[idx_train] ) # shape = (N, 250+250)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # cond = torch.tensor((data[args.condition] - cond_mean) / cond_mad)[idx_train]
    # print(cond.max().item(), cond.min().item())
    # assert False

    data = np.load( os.path.join(args.data_dir, 'test.npz') )
    dataset_test = torch.utils.data.TensorDataset(torch.tensor( np.concatenate([data['emb_2d'], data['emb_3d']], axis=1) ), torch.tensor((data[args.condition] - cond_mean) / cond_mad) )
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size*8, shuffle=False, num_workers=4)

    # get input dimension
    args.dim_in = dataset_train[0][0].shape[0]

    # print args
    print(args)
    os.system('mkdir ' + args.log_dir)
    torch.save(args, args.log_dir + '/args.pt')


    # model
    model = Model(args)
    if not args.load_checkpoint_for_warm_start is None:
        model = Model.load_from_checkpoint(args.load_checkpoint_for_warm_start, args=args)

    # trainer
    checkpoint_callback = ModelCheckpoint(dirpath=args.log_dir, save_last=True, filename='checkpoint-best', monitor='val_loss')
    trainer = pl.Trainer(max_epochs=args.epoch_num, gradient_clip_val=5, check_val_every_n_epoch=5, default_root_dir=args.log_dir, callbacks=[checkpoint_callback], num_sanity_val_steps=10, accelerator='gpu', strategy='ddp', num_nodes=args.ddp_num_nodes, devices=args.ddp_device)

    # trainer.test(model, dataloader_test)

    trainer.fit(model, dataloader_train, dataloader_val, ckpt_path=args.load_checkpoint_for_continual_training)

    # trainer.test(model, dataloader_test)
