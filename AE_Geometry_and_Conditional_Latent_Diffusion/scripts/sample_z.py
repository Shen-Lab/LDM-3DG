import argparse
import os
import shutil

import numpy as np
import torch
import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D, LDM_Cond
from geometric_gnn_dojo_utils.models import EGNNModel
from tqdm import tqdm

import pdb


def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    args = parser.parse_args()

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    config.train.batch_size = 64
    config.train.val_freq = 1000
    config.train.optimizer.lr = 1e-4
    config.model.hidden_dim = 128
    # log_dir = misc.get_new_log_dir(args.logdir, prefix='ldm', tag=args.tag)

    # config.train.batch_size = 32
    # config.train.val_freq = 100
    # config.train.optimizer.lr = 1e-4
    # config.model.hidden_dim = 128
    # log_dir = misc.get_new_log_dir(args.logdir, prefix='debug', tag=args.tag)

    # Logging
    os.system('rm -r ./debug')
    log_dir = './debug'
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform
    )
    # train_set, val_set = subsets['train'], subsets['test']
    train_set, val_set = subsets['test'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')
    

    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(val_set, 20, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model
    logger.info('Building model...')
    model = LDM_Cond(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)

    model.load_state_dict(torch.load('logs_diffusion/ldm_2023_11_16__18_01_30/checkpoints/30000.pt')['model'])

    # print(model)
    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    model.eval()
    num_sample = 500
    zs = []
    emb_prots = []
    for batch in val_loader:
        batch = batch.to(args.device)

        with torch.no_grad():
            z, emb_prot = model.sample_z(batch.protein_pos, batch.protein_atom_feature.float(), batch.protein_element_batch, num_sample)

        zs.append(z.to('cpu'))
        emb_prots.append(emb_prot.to('cpu'))

    zs = torch.cat(zs, dim=0)
    emb_prots = torch.cat(emb_prots, dim=0)

    print(zs.shape, emb_prots.shape)

    torch.save(zs, 'samples_latent/sample_z.pt')
    torch.save(emb_prots, 'samples_latent/emb_protein.pt')
