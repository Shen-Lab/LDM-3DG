import argparse
import math
import numpy as np
import torch
from torch import nn
import torch_geometric as tgeom
# dataset url: https://github.com/divelab/MoleculeX/tree/molx/Molecule3D
from datasets.datasets_molecule3d_new import Molecule3D
from dmcg_utils.model.gnn import GNN as Decoder_2Dto3D
from geometric_gnn_dojo_utils.models import MACEModel, EGNNModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
import pdb
import os


class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        # self.encoder_3d = MACEModel(r_max=args.radius_cutoff, num_layers=2, emb_dim=250, in_dim=args.input_x_dim, out_dim=250)
        self.encoder_3d = EGNNModel(num_layers=4, emb_dim=256, in_dim=118, out_dim=250, aggr='mean', pool='mean')

        hparams = {
            "mlp_hidden_size": 1024,
            "mlp_layers": 2,
            "latent_size": 1024,
            "use_layer_norm": False,
            "num_message_passing_steps": 4,
            "global_reducer": "sum",
            "node_reducer": "sum",
            "dropedge_rate": 0.1,
            "dropnode_rate": 0.1,
            "dropout": 0.1,
            "layernorm_before": False,
            "encoder_dropout": 0.0,
            "use_bn": True,
            "vae_beta": 1.0,
            "decoder_layers": None,
            "reuse_prior": True,
            "cycle": 1,
            "pred_pos_residual": True,
            "node_attn": True,
            "global_attn": False,
            "shared_decoder": False,
            "use_global": False,
            "sg_pos": False,
            "shared_output": False,
            "use_ss": False,
            "rand_aug": False,
            "no_3drot": True,
            "not_origin": False,
            "clamp_dist": None,
            "aux_loss": 0.2,
            "ang_lam": 0,
            "bond_lam": 0,
            "dim_in": args.input_x_dim,
        }
        self.decoder_2dto3d = Decoder_2Dto3D(**hparams)

        self.args = args

    def calculate_loss(self, batch, batch_idx):
        return loss_atom_pos * 0.5 + loss_atom_neg * 0.5, loss_bond_pos * 0.5 + loss_bond_neg * 0.5, loss_coor_pos

    def training_step(self, batch, batch_idx):
        batch.pos = batch.positions

        emb_2d = batch.emb_2d
        # emb_3d = self.encoder_3d(batch, True)
        _, emb_3d = self.encoder_3d(batch.x, batch.pos, batch.radius_edge_index, batch.batch, True)

        # spatial graphs
        batch.edge_index = batch.radius_edge_index
        edge_attr = torch.exp( - torch.norm(batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]], dim=1) )
        batch.edge_attr = torch.einsum('i,j->ij', edge_attr, torch.linspace(1, 5, 16).to(edge_attr.device))
        batch.n_edges = batch.n_edges_radius

        atom_pred_list, extra_output = self.decoder_2dto3d(batch, emb_3d)

        gt_poss = tgeom.utils.unbatch(batch.pos, batch.batch)
        atom_pred_list = tgeom.utils.unbatch(atom_pred_list[-1], batch.batch)

        loss = 0
        for pred_pos, gt_pos in zip(atom_pred_list, gt_poss):
            pred_pos = self.kbash_alignment(pred_pos, gt_pos)
            _loss = (pred_pos - gt_pos).pow(2).mean()
            loss += _loss

            # print(_loss)
            # pdb.set_trace()
        loss /= len(atom_pred_list)

        # avoiding collapse
        emb_3d_norm = emb_3d.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', emb_3d, emb_3d) / torch.einsum('i,j->ij', emb_3d_norm, emb_3d_norm)
        sim_matrix = torch.exp(sim_matrix)
        pos_sim = sim_matrix[range(emb_3d.shape[0]), range(emb_3d.shape[0])]
        loss_neg_contrast = torch.log( (sim_matrix.sum(dim=1) - pos_sim) ).mean()
        loss += loss_neg_contrast * 0.1

        # loss, loss_dict = self.decoder_2dto3d.compute_loss(atom_pred_list, extra_output, batch)
        self.log('train_loss', loss, batch_size=batch.num_graphs, on_epoch=True, sync_dist=True)

        return loss
    
    def kbash_alignment(self, coor1, coor2):
        A = ( coor2 - coor2.mean(dim=0) ).t() @ ( coor1 - coor1.mean(dim=0) )
        U, S, Vt = torch.linalg.svd(A)
        rotation = ( U @ torch.diag( torch.Tensor([1, 1, torch.sign(torch.det(A))]).to(coor1.device) ) ) @ Vt
        translation = coor2.mean(dim=0) - torch.t(rotation @ coor1.mean(dim=0, keepdim=True).t())

        rotation = rotation.detach()
        translation = translation.detach()

        coor1 = coor1 @ rotation.t() + translation
        return coor1

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./logs/debug')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch_num', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=5e-5) # to be tuned

    parser.add_argument('--decode_len', type=int, default=200)
    parser.add_argument('--radius_cutoff', type=float, default=5.0)
    parser.add_argument('--graph_readout', type=str, default="add", choices=["mean", "add"])
    parser.add_argument('--painn_n_rbf', type=int, default=20)
    parser.add_argument('--gat_attn_head_num', type=int, default=8)
    parser.add_argument('--transformer_attn_head_num', type=int, default=8)

    # loss weights
    parser.add_argument('--loss_weight_atom', type=float, default=1)
    parser.add_argument('--loss_weight_bond', type=float, default=1)
    parser.add_argument('--loss_weight_coordinate', type=float, default=1)

    # continual training
    parser.add_argument('--load_checkpoint_for_warm_start', type=str, default=None)
    parser.add_argument('--load_checkpoint_for_continual_training', type=str, default=None)

    # number of nodes for parallel training
    parser.add_argument('--ddp_num_nodes', type=int, default=1)
    # number of devices in each node for parallel training
    parser.add_argument('--ddp_device', type=int, default=1)
    args = parser.parse_args()

    # dataset and dataloader
    dataset_train = Molecule3D(root='../AE_geom_uncond_weights_and_data/', radius_cutoff=args.radius_cutoff, split='train', split_mode='random', args=args)
    dataloader_train = tgeom.loader.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # dataset_val = Molecule3D(root='../data/molecule3d/', radius_cutoff=args.radius_cutoff, split='val', split_mode='random', args=args)
    # dataloader_val = tgeom.loader.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # get input dimension
    args.input_x_dim = dataset_train[0].x.shape[1]
    args.input_edge_attr_dim = dataset_train[0].edge_attr.shape[1]

    # model
    model = Model(args)
    if not args.load_checkpoint_for_warm_start is None:
        model = Model.load_from_checkpoint(args.load_checkpoint_for_warm_start, args=args)
    
    # args2 = torch.load('./logs/job2_decoder_2d_to_3d_4layer_new/args.pt')
    # model = Model.load_from_checkpoint('./logs/job2_decoder_2d_to_3d_4layer_new/checkpoint-best.ckpt', args=args2)

    # print args
    print(args)
    os.system('mkdir ' + args.log_dir)
    torch.save(args, args.log_dir + '/args.pt')

    # trainer
    checkpoint_callback = ModelCheckpoint(dirpath=args.log_dir, save_last=True, filename='checkpoint-best', monitor='train_loss')
    trainer = pl.Trainer(max_epochs=args.epoch_num, gradient_clip_val=5, check_val_every_n_epoch=1, default_root_dir=args.log_dir, callbacks=[checkpoint_callback], num_sanity_val_steps=10,
                         accelerator='gpu', strategy='ddp', num_nodes=args.ddp_num_nodes, devices=args.ddp_device)

    trainer.fit(model, dataloader_train, ckpt_path=args.load_checkpoint_for_continual_training)

