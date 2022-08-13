import numpy as np
import argparse
import networkx as nx
import time
import os
import sys
import pickle
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import KDTree
from xnetmf_config import *
from scipy.linalg import block_diag
import scipy.sparse as sps
import xnetmf
import regal_utils
import refina
import refina_utils
import matlab_utils as utl
import math
import gwl_model
import torch.optim as optim
from torch.optim import lr_scheduler
import scipy

def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=50):
    kd_tree = KDTree(emb2, metric=distance_metric)

    row = np.array([])
    col = np.array([])
    data = np.array([])

    dist, ind = kd_tree.query(emb1, k=num_top)
    print("queried alignments")
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    #
    return sparse_align_matrix.todense()


def get_counterpart(alignment_matrix, true_alignments):
    n_nodes = alignment_matrix.shape[0]

    correct_nodes = []
    counterpart_dict = {}

    if not sps.issparse(alignment_matrix):
        sorted_indices = np.argsort(alignment_matrix)

    for node_index in range(n_nodes):
        target_alignment = node_index #default: assume identity mapping, and the node should be aligned to itself
        if true_alignments is not None: #if we have true alignments (which we require), use those for each node
            target_alignment = int(true_alignments[node_index])
        if sps.issparse(alignment_matrix):
            row, possible_alignments, possible_values = sps.find(alignment_matrix[node_index])
            node_sorted_indices = possible_alignments[possible_values.argsort()]
        else:
            node_sorted_indices = sorted_indices[node_index]
        if target_alignment in node_sorted_indices[-1:]:
            correct_nodes.append(node_index)
        counterpart = node_sorted_indices[-1]
        counterpart_dict[node_index] = counterpart

    return correct_nodes, counterpart_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Run CONE Align.")

    parser.add_argument('--true_align', nargs='?', default='data/synthetic-combined/arenas/arenas950-1/arenas_edges-mapping-permutation.txt',
                        help='True alignment file.')
    parser.add_argument('--combined_graph', nargs='?', default='data/synthetic-combined/arenas/arenas950-1/arenas_combined_edges.txt', help='Edgelist of combined input graph.')
    parser.add_argument("--level", default=3, type=int, help='Number of levels for coarseing')
    parser.add_argument('--output_alignment', nargs='?', default='output/alignment_matrix/arenas/arenas950-1', help='Output path for alignment matrix.')
    # Embedding Method
    parser.add_argument('--embmethod', nargs='?', default='netMF', help='Node embedding method.')
    # xnetmf parameters
    parser.add_argument('--attributes', nargs='?', default=None,help='File with saved numpy matrix of node attributes, or int of number of attributes to synthetically generate.  Default is 5 synthetic.')
    parser.add_argument('--attrvals', type=int, default=2,help='Number of attribute values. Only used if synthetic attributes are generated')
    parser.add_argument('--k', type=int, default=10,help='Controls of landmarks to sample. Default is 10.')
    parser.add_argument('--untillayer', type=int, default=2,help='Calculation until the layer for xNetMF.')
    parser.add_argument('--alpha', type=float, default = 0.01, help = "Discount factor for further layers")
    parser.add_argument('--gammastruc', type=float, default = 1, help = "Weight on structural similarity")
    parser.add_argument('--gammaattr', type=float, default = 1, help = "Weight on attribute similarity")
    parser.add_argument('--buckets', default=2, type=float, help="base of log for degree (node feature) binning")
    # REFINA parameters
    parser.add_argument('--n-iter', type=int, default=100, help='Maximum #iter for RefiNA. Default is 20.') 
    parser.add_argument('--token-match', type=float, default = -1, help = "Token match score for each node.  Default of -1 sets it to reciprocal of largest graph #nodes rounded up to smallest power of 10")
    parser.add_argument('--n-update', type=int, default=-1, help='How many possible updates per node. Default is -1, or dense refinement.  Positive value uses sparse refinement')   
    # Alignment methods
    parser.add_argument('--alignmethod', nargs='?', default='REGAL', help='Network alignment method.')
    # Refinement methods
    parser.add_argument('--refinemethod', nargs='?', default=None, help='Network refinement method, to overcome the shortcoming of MILE')
    # Whether doing coarsening or not
    parser.add_argument('--coarsen', default=False, action='store_true')
    return parser.parse_args()

def degree_matrix(adj):
    n, _ =adj.shape
    deg = np.zeros((n, n))
    for i in range(n):
        deg[i, i] = np.sum(adj[i, :])
    return deg

def main(args):
    true_align_name = args.true_align
    with open(true_align_name, "rb") as true_alignments_file:
        # for python3, you need to use latin1 as the encoding method
        true_align = pickle.load(true_alignments_file, encoding = "latin1")
    before_emb = time.time()

    ##################### Load data ######################################
    if args.coarsen is True:
        # running CAPER
        pickle_in = open(args.combined_graph, "rb")
        data = pickle.load(pickle_in)
        adjA = np.array(data['A_list'][0][args.level-1].todense())
        adjB = np.array(data['A_list'][1][args.level-1].todense())
        split_idx = adjA.shape[0]
    else:
        # running normal graph alignment methods
        combined_graph_name = args.combined_graph
        graph = nx.read_edgelist(combined_graph_name, nodetype=int, comments="%")
        adj = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()) ).todense().astype(float)
        node_num = int(adj.shape[0] / 2)
        adjA = np.array(adj[:node_num, :node_num])
        split_idx = adjA.shape[0]
        adjB = np.array(adj[node_num:, node_num:])

    ##################### Proprocess if needed ######################################
    if (args.embmethod == "xnetMF"):
        print("Generating xnetMF embeddings for REGAL")
        adj = block_diag(adjA, adjB)
        graph = Graph(adj, node_attributes = args.attributes)
        max_layer = args.untillayer
        if args.untillayer == 0:
            max_layer = None
        if args.buckets == 1:
            args.buckets = None
        rep_method = RepMethod(max_layer = max_layer, alpha = args.alpha, k = args.k, num_buckets = args.buckets, #BASE OF LOG FOR LOG SCALE
            normalize = True, gammastruc = args.gammastruc, gammaattr = args.gammaattr)
        if max_layer is None:
            max_layer = 1000
        print("Learning representations with max layer %d and alpha = %f" % (max_layer, args.alpha))
        embed = xnetmf.get_representations(graph, rep_method)
        after_emb = time.time()
        if (args.store_emb):
            np.save(args.embeddingA, embed, allow_pickle=False)
            np.save(args.embeddingB, embed, allow_pickle=False)
    elif (args.embmethod == "gwl"):
        # parse the data to be gwl readable format
        print("Parse the data to be gwl readable format")
        data_gwl = {}
        data_gwl['src_index'] = {}
        data_gwl['tar_index'] = {}
        data_gwl['src_interactions'] = []
        data_gwl['tar_interactions'] = []
        data_gwl['mutual_interactions'] = []
        for i in range(adjA.shape[0]):
            data_gwl['src_index'][float(i)] = i
        for i in range(adjB.shape[0]):
            data_gwl['tar_index'][float(i)] = i
        ma,mb = adjA.nonzero()
        for i in range(ma.shape[0]):
            data_gwl['src_interactions'].append([ma[i], mb[i]])
        ma,mb = adjB.nonzero()
        for i in range(ma.shape[0]):
            data_gwl['tar_interactions'].append([ma[i], mb[i]])
        after_emb = time.time()
    else:
        print("No preprocessing needed for FINAL")
        after_emb = time.time()

    ##################### Alignment ######################################
    before_align = time.time()
    # step2 and 3: align embedding spaces and match nodes with similar embeddings
    if args.alignmethod == 'REGAL':
        emb1, emb2 = regal_utils.get_embeddings(embed, graph_split_idx=split_idx)
        alignment_matrix = regal_utils.get_embedding_similarities(emb1, emb2, num_top = None)
    elif args.alignmethod == 'FINAL':
        graph1 = nx.from_numpy_matrix(adjA)
        graph2 = nx.from_numpy_matrix(adjB)
        degree_one = np.array(graph1.degree)[:, 1].reshape(adjA.shape[0], 1)
        degree_two = np.array(graph2.degree)[:, 1].reshape(adjB.shape[0], 1)
        k = int(math.floor(math.log((adjA.shape[0]+adjB.shape[0])/2, 2)))
        kd_sim = kd_align(degree_one, degree_two, distance_metric="euclidean", num_top=k)
        est_align, total_time, sim = utl.run_matlab(adjA, adjB, init_align=csr_matrix(kd_sim, dtype=float).tocoo(),
                                               configs={'maxiter': 50.0, 'alpha': 0.9, 'tol': 1e-7},
                                               path={'FINAL': './FINAL/matlab'}, attribs=None,
                                               method='final')
        alignment_matrix = np.zeros((adjA.shape[0], adjB.shape[0]))
        for i in est_align.keys():
            alignment_matrix[int(i), int(est_align[i])] = 1
    elif args.alignmethod == "gwl":
        result_folder = 'gwl_test'
        cost_type = ['cosine']
        method = ['proximal']
        opt_dict = {'epochs': 30,
                    'batch_size': 57000,
                    'use_cuda': False,
                    'strategy': 'soft',
                    'beta': 1e-2,
                    'outer_iteration': 200,
                    'inner_iteration': 1,
                    'sgd_iteration': 500,
                    'prior': False,
                    'prefix': result_folder,
                    'display': False}
        for m in method:
            for c in cost_type:
                hyperpara_dict = {'src_number': len(data_gwl['src_index']),
                                'tar_number': len(data_gwl['tar_index']),
                                'dimension': 256,
                                'loss_type': 'L2',
                                'cost_type': c,
                                'ot_method': m}
                gwd_model = gwl_model.GromovWassersteinLearning(hyperpara_dict)

                # initialize optimizer
                optimizer = optim.Adam(gwd_model.gwl_model.parameters(), lr=1e-2)
                scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

                # Gromov-Wasserstein learning
                gwd_model.train_without_prior(data_gwl, optimizer, opt_dict, scheduler=None)
                # save model
                gwd_model.save_model('{}/model_{}_{}.pt'.format(result_folder, m, c))
                gwd_model.save_recommend('{}/result_{}_{}.pkl'.format(result_folder, m, c))
                alignment_matrix = gwd_model.trans

    ##################### Refine Alignment embeddings ######################################
    for i in range(args.level-1, 0, -1):
        if args.refinemethod is not None and args.coarsen is True:
            if args.refinemethod == "RefiNA": 
                left_mat = np.array(data['A_list'][0][i].todense())
                right_mat = np.array(data['A_list'][1][i].todense())
                # soft alignment for coarser levels
                alignment_matrix = refina.refina(alignment_matrix, left_mat, right_mat, args, 100)
                alignment_matrix = np.array(data['matches'][0][i-1].todense())@alignment_matrix@np.array(data['matches'][1][i-1].todense()).T
    if args.refinemethod is not None:
        if args.refinemethod == "RefiNA":
            if sps.issparse(alignment_matrix):
                alignment_matrix = np.array(alignment_matrix.todense())
            if args.coarsen is True:
                if args.n_update > 0:
                    alignment_matrix = sps.csr_matrix(alignment_matrix)
                    left_mat = data['A_list'][0][0]
                    right_mat = data['A_list'][1][0]
                else:
                    left_mat = np.array(data['A_list'][0][0].todense())
                    right_mat = np.array(data['A_list'][1][0].todense())
                alignment_matrix = refina.refina(alignment_matrix, left_mat, right_mat, args, 100, true_alignments = true_align)
            else:
                if args.n_update > 0:
                    alignment_matrix = sps.csr_matrix(alignment_matrix)
                    adjA = sps.csr_matrix(adjA)
                    adjB = sps.csr_matrix(adjB)
                alignment_matrix = refina.refina(alignment_matrix, adjA, adjB, args, args.n_iter, true_alignments = true_align)         
    node_num = alignment_matrix.shape[0]
    after_align = time.time()

    if true_align is not None:
        score, _ = refina_utils.score_alignment_matrix(alignment_matrix, topk = 1, true_alignments = true_align)
        if args.coarsen is True:
            left_mat = np.array(data['A_list'][0][0].todense())
            right_mat = np.array(data['A_list'][1][0].todense())
            mnc = refina_utils.score_MNC(alignment_matrix, left_mat, right_mat)
        else:
            mnc = refina_utils.score_MNC(alignment_matrix, adjA, adjB)
        print("Top 1 accuracy: %.5f" % score)
        print("MNC: %.5f" % mnc)
    # evaluation
    total_time = (after_align - before_align) + (after_emb - before_emb)
    print(("score for CAPER: %f" % score))
    print(("time for CAPER (in seconds): %f" % total_time))


    with open(args.output_stats, "w") as log:
        log.write("score: %f\n" % score)
        log.writelines("time(in seconds): %f\n"% total_time)

if __name__ == "__main__":
    args = parse_args()
    main(args)
