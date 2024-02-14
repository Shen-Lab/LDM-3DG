# multiprocess for functions
# reference: https://docs.python.org/3/library/multiprocessing.html
def multiprocess(func, data_list, batch_size=32, num_worker=8):
    '''
    func: callabel function
    data_list: list of data for multiprocessing
    batch_size: batch size
    num_worker: number of threads
    '''
    from tqdm import tqdm
    from multiprocessing import Pool

    data_batch_list = []
    output_list = []
    for idx, data in enumerate(tqdm(data_list)):
        data_batch_list.append(data)

        if idx + 1 == len(data_list) or len(data_batch_list) == batch_size:
            with Pool(num_worker) as p:
                out = p.map(func, data_batch_list)
            output_list += out
            data_batch_list = []

    return output_list


# perform kbash alignment for two point clouds
# reference: https://github.com/octavian-ganea/equidock_public
def kbash_alignment(coor1, coor2):
    '''
    coor1: point cloud 1, shape = (N, 3)
    coor2: point cloud 2, shape = (N, 3)
    '''
    import torch

    A = ( coor2 - coor2.mean(dim=0) ).t() @ ( coor1 - coor1.mean(dim=0) )
    U, S, Vt = torch.linalg.svd(A)
    rotation = ( U @ torch.diag(torch.Tensor([1, 1, torch.sign(torch.det(A))]).to(coor1.device)) ) @ Vt
    translation = coor2.mean(dim=0) - (rotation @ coor1.mean(dim=0, keepdim=True).t()).t()

    # detach the gradients if there are
    rotation = rotation.detach()
    translation = translation.detach()

    coor1 = coor1 @ rotation.t() + translation
    return coor1, rotation, translation


# alignment two protein sequences
# reference: https://github.com/Shen-Lab/gcWGAN
def sequence_alignment(seq1, seq2, return_seqid_only=False):
    '''
    seq1: protein sequence 1
    seq2: protein sequence 2
    return_seqid_only: whether only return sequence identity (to save the further computation cost for index mappings)
    '''
    from Bio import pairwise2
    from Bio.SubsMat import MatrixInfo as matlist
    import numpy as np

    alignments = pairwise2.align.globaldd(seq1, seq2, matlist.blosum62, -11, -1, -11, -1)

    align_best = None
    seqid_max = 0
    for align in alignments:
        s1 = np.array(list(align[0]))
        s2 = np.array(list(align[1]))

        s1, s2 = s1[s1!='-'], s2[s1!='-']
        seqid = float( (s1==s2).sum() ) / align[-1]
        if seqid > seqid_max:
            seqid_max = seqid
            align_best = align

    if return_seqid_only:
        return seqid_max

    # construct index alignment mapping
    idx_map_s1tos2_list = []
    idx_map_s2tos1_list = []

    for aa1, aa2 in zip(align_best[0], align_best[1]):
        if aa1 == '-':
            if aa2 == '-': # both are gaps
                pass
            else: # aa2 != '-', aa1 is gap and aa2 is not
                idx_map_s2tos1_list.append(-1)
        else: # aa1 != '-':
            if aa2 == '-': # aa1 is not gap and aa2 is
                idx_map_s1tos2_list.append(-1)
            elif aa1 != aa2: # and aa2 != '-', both are not gaps and aa1 is different from aa2
                idx_map_s2tos1_list.append(-1)
                idx_map_s1tos2_list.append(-1)
            else: # aa2 != '-' and aa1 == aa2, both are not gaps and aa1 is the same as aa2
                idx_map_s1tos2 = len(idx_map_s2tos1_list)
                idx_map_s2tos1 = len(idx_map_s1tos2_list)
                idx_map_s1tos2_list.append(idx_map_s1tos2)
                idx_map_s2tos1_list.append(idx_map_s2tos1)

    return idx_map_s1tos2_list, idx_map_s2tos1_list, seqid_max


# construct edge index for a complete graph
def construct_complete_graph(num_node, return_index=True, add_self_loop=False):
    '''
    num_node: number of nodes in the graph
    add_self_loop: whether add self loop in the graph
    '''
    import torch
    import torch_geometric as tgeom

    adj = 1 - torch.eye(num_node)
    if add_self_loop:
        adj += torch.eye(num_node)

    if not return_index:
        return adj
    else:
        edge_index, _ = tgeom.utils.dense_to_sparse(adj)
        return adj, edge_index


def statistical_metric(vars1, vars2, num_bin=100, plot=False):
    import numpy as np
    from scipy.stats import wasserstein_distance

    metric_dict = {}

    # total variation distance
    # reference: https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
    v_max, v_min = np.max(np.concatenate([vars1, vars2])), np.min(np.concatenate([vars1, vars2]))
    bins = np.linspace(v_min, v_max, num=num_bin)

    pmf1, _ = np.histogram(vars1, bins=bins)
    pmf1 = pmf1 / float(pmf1.sum())
    pmf2, _ = np.histogram(vars2, bins=bins)
    pmf2 = pmf2 / float(pmf2.sum())

    metric_dict['tvd'] = np.max(np.abs(pmf1 - pmf2))

    # hellinger distance
    # reference: https://en.wikipedia.org/wiki/Hellinger_distance
    metric_dict['hd'] = 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(pmf1) - np.sqrt(pmf2))

    # wasserstein distance
    metric_dict['wd'] = wasserstein_distance(vars1, vars2)

    # plot for debug
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.hist(vars1, bins=num_bin, density=True, alpha=0.8)
        ax.hist(vars2, bins=num_bin, density=True, alpha=0.8)
        plt.savefig('./utils_yy/debug.png')

    return metric_dict
