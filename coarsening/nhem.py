from collections import defaultdict
from graph import Graph
import numpy as np
from utils import cmap2C
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import from_alignment_edgelist_to_graph, create_coarse_graph
import importlib
import logging
import numpy as np
import pdb
import os
import time
import pickle

#### Credit: codes partially borrowed from MILE: https://github.com/jiongqian/MILE
#### Also cited in paper

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--data', required=True, help='Input graph file')
    parser.add_argument('--coarsen-level', default=2, type=int, help='MAX number of levels of coarsening.')
    parser.add_argument('--workers', default=4, type=int, help='Number of workers.')
    parser.add_argument('--output-path', required=True, help='Path to save the output pickle file')
    args = parser.parse_args()
    return args

def normalized_adj_wgt(graph):
    adj_wgt = graph.adj_wgt
    adj_idx = graph.adj_idx
    norm_wgt = np.zeros(adj_wgt.shape, dtype=np.float32)
    degree = graph.degree
    for i in range(graph.node_num):
        for j in range(adj_idx[i], adj_idx[i + 1]):
            neigh = graph.adj_list[j]
            norm_wgt[j] = adj_wgt[neigh] / np.sqrt(degree[i] * degree[neigh])
    return norm_wgt

def normalized_heavy_edge_matching(args, graph):
    '''Generate matchings using the hybrid method. It changes the cmap in graph object, 
    return groups array and coarse_graph_size.'''
    node_num = graph.node_num
    adj_list = graph.adj_list  # big array for neighbors.
    adj_idx = graph.adj_idx  # beginning idx of neighbors.
    adj_wgt = graph.adj_wgt  # weight on edge
    node_wgt = graph.node_wgt  # weight on node
    cmap = graph.cmap
    norm_adj_wgt = normalized_adj_wgt(graph)
    coarsen_to = max(1, graph.node_num // (2 ** args.coarsen_level))  # rough estimation.
    max_node_wgt = int((5.0 * graph.node_num) / coarsen_to)

    groups = []  # a list of groups, each group corresponding to one coarse node.
    matched = [False] * node_num

    degree = [adj_idx[i + 1] - adj_idx[i] for i in range(0, node_num)]
    sorted_idx = np.argsort(degree)
    for idx in sorted_idx:
        if matched[idx]:
            continue
        max_idx = idx
        max_wgt = -1
        for j in range(adj_idx[idx], adj_idx[idx + 1]):
            neigh = adj_list[j]
            if neigh == idx:  # KEY: exclude self-loop. Otherwise, mostly matching with itself.
                continue
            curr_wgt = norm_adj_wgt[j]
            if ((not matched[neigh]) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt):
                max_idx = neigh
                max_wgt = curr_wgt
        # it might happen that max_idx is idx, which means cannot find a match for the node. 
        matched[idx] = matched[max_idx] = True
        if idx == max_idx:
            groups.append([idx])
        else:
            groups.append([idx, max_idx])
    coarse_graph_size = 0
    for idx in range(len(groups)):
        for ele in groups[idx]:
            cmap[ele] = coarse_graph_size
        coarse_graph_size += 1
    return (groups, coarse_graph_size)

def multilevel_embed(args, graph):
    '''This method defines the multilevel embedding method.'''
    start = time.time()

    # Step-1: Graph Coarsening.
    original_graph = graph
    coarsen_level = args.coarsen_level
    graphs = []
    graphs.append(graph)
    for i in range(coarsen_level):
        match, coarse_graph_size = normalized_heavy_edge_matching(args, graph)
        coarse_graph = create_coarse_graph(graph, match, coarse_graph_size)
        graph = coarse_graph
        graphs.append(graph)

    return graphs

if __name__ == "__main__":
    before_emb = time.time()
    seed = 123
    np.random.seed(seed)
    args = parse_args()

    input_graph_path = args.data
    graphs, mapping = from_alignment_edgelist_to_graph(input_graph_path)
    labels = None
    embed_As = []
    embed_Cs = []
    max_degree = 0
    for graph in graphs:
        As = []
        Cs = []
        # Generate multilevel projects
        graph_embeds = multilevel_embed(args, graph)
        for i in range(args.coarsen_level):
            As.append(graph_embeds[i].A)
            Cs.append(graph_embeds[i].C)
        embed_As.append(As)
        embed_Cs.append(Cs)
    to_save = {'A_list': embed_As, 'matches': embed_Cs, 'labels': labels}
    pickle_out = open(args.output_path, "wb")
    pickle.dump(to_save, pickle_out)
    pickle_out.close()
    after_emb = time.time()
    total_time = after_emb - before_emb
    print(("time for embed (in seconds): %f" % total_time))
    print("MILE completed")