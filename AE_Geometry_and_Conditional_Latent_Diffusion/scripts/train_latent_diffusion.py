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
    log_dir = misc.get_new_log_dir(args.logdir, prefix='ldm', tag=args.tag)

    # config.train.batch_size = 32
    # config.train.val_freq = 100
    # config.train.optimizer.lr = 1e-4
    # config.model.hidden_dim = 128
    # log_dir = misc.get_new_log_dir(args.logdir, prefix='debug', tag=args.tag)

    # Logging
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
    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')
    

    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model
    logger.info('Building model...')
    model = LDM_Cond(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    # print(model)
    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    # scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    # load latent embedding
    emb2d_all = torch.load('emb2d.pt')
    emb3d_all = torch.load('emb3d.pt')

    from tqdm import tqdm
    idx2idx = - torch.ones(len(dataset), dtype=torch.int64)
    idx_train = torch.tensor(torch.load(config.data['split'])['train'])
    idx2idx[idx_train] = torch.arange(len(train_set))

    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            idxs = idx2idx[batch.id.to('cpu')]
            emb2d = emb2d_all[idxs].to(args.device)
            emb3d = emb3d_all[idxs].to(args.device)

            num_batch = batch.id.shape[0]

            gt_protein_pos = batch.protein_pos

            # pdb.set_trace()

            loss = model(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=torch.zeros(num_batch, 3).to(args.device),
                ligand_v=torch.ones(num_batch, config.model.hidden_dim-1).to(args.device),
                batch_ligand=torch.arange(num_batch).to(args.device),

                emb=torch.cat([emb2d, emb3d], dim=1)
            )

            # pdb.set_trace()

            # loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']
            # loss = loss / config.train.n_acc_batch
            loss.backward()
        # orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        # if it % args.train_report_iter == 0:
        #     logger.info(
        #         '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
        #             it, loss, loss_pos, loss_v, optimizer.param_groups[0]['lr'], orig_grad_norm
        #         )
        #     )
        #     for k, v in results.items():
        #         if torch.is_tensor(v) and v.squeeze().ndim == 0:
        #             writer.add_scalar(f'train/{k}', v, it)
        #     writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        #     writer.add_scalar('train/grad', orig_grad_norm, it)
        #     writer.flush()

        return loss.item()



    try:
        best_loss, best_iter = None, None
        loss_it = 0
        for it in range(1, config.train.max_iters + 1):
            # with torch.autograd.detect_anomaly():
            loss_it += train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = loss_it / config.train.val_freq
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(f'[Validate] Val loss {val_loss:.6f} is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')

                loss_it = 0

    except KeyboardInterrupt:
        logger.info('Terminating...')
