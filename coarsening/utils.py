from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import logging
import networkx as nx
import numpy as np
import scipy.sparse as sp
import sys
from graph import Graph
import os
import pdb

def cmap2C(cmap): # fine_graph to coarse_graph, matrix format of cmap: C: n x m, n>m.
    node_num = len(cmap)
    i_arr = []
    j_arr = []
    data_arr = []
    for i in range(node_num):
        i_arr.append(i)
        j_arr.append(cmap[i])
        data_arr.append(1)
    return sp.csr_matrix((data_arr, (i_arr, j_arr)))    

def create_coarse_graph(graph, groups, coarse_graph_size):
    '''create the coarser graph and return it based on the groups array and coarse_graph_size'''
    coarse_graph = Graph(coarse_graph_size, graph.edge_num)
    coarse_graph.finer = graph
    graph.coarser = coarse_graph
    cmap = graph.cmap
    adj_list = graph.adj_list
    adj_idx = graph.adj_idx
    adj_wgt = graph.adj_wgt
    node_wgt = graph.node_wgt

    coarse_adj_list = coarse_graph.adj_list
    coarse_adj_idx = coarse_graph.adj_idx
    coarse_adj_wgt = coarse_graph.adj_wgt
    coarse_node_wgt = coarse_graph.node_wgt
    coarse_degree = coarse_graph.degree

    coarse_adj_idx[0] = 0
    nedges = 0  # number of edges in the coarse graph
    for idx in range(len(groups)):  # idx in the graph
        coarse_node_idx = idx
        neigh_dict = dict()  # coarser graph neighbor node --> its location idx in adj_list.
        group = groups[idx]
        for i in range(len(group)):
            merged_node = group[i]
            if (i == 0):
                coarse_node_wgt[coarse_node_idx] = node_wgt[merged_node]
            else:
                coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]

            istart = adj_idx[merged_node]
            iend = adj_idx[merged_node + 1]
            for j in range(istart, iend):
                k = cmap[adj_list[
                    j]]  # adj_list[j] is the neigh of v; k is the new mapped id of adj_list[j] in coarse graph.
                if k not in neigh_dict:  # add new neigh
                    coarse_adj_list[nedges] = k
                    coarse_adj_wgt[nedges] = adj_wgt[j]
                    neigh_dict[k] = nedges
                    nedges += 1
                else:  # increase weight to the existing neigh
                    coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]
                # add weights to the degree. For now, we retain the loop. 
                coarse_degree[coarse_node_idx] += adj_wgt[j]

        coarse_node_idx += 1
        coarse_adj_idx[coarse_node_idx] = nedges

    coarse_graph.edge_num = nedges

    coarse_graph.resize_adj(nedges)
    C = cmap2C(cmap)  # construct the matching matrix.
    graph.C = C
    coarse_graph.A = C.transpose().dot(graph.A).dot(C)
    return coarse_graph


def graph_to_adj(graph, self_loop=False):
    '''self_loop: manually add self loop or not'''
    node_num = graph.node_num
    i_arr = []
    j_arr = []
    data_arr = []
    for i in range(0, node_num):
        for neigh_idx in range(graph.adj_idx[i], graph.adj_idx[i+1]):
            i_arr.append(i)
            j_arr.append(graph.adj_list[neigh_idx])
            data_arr.append(graph.adj_wgt[neigh_idx])
    adj = sp.csr_matrix((data_arr, (i_arr, j_arr)), shape=(node_num, node_num), dtype=np.float32)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    return adj    

def from_alignment_edgelist_to_graph(dataset):
    # _,prefix = os.path.split(dataset)
    graph = nx.read_edgelist(dataset, nodetype = int, comments = "%")
    adj = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()))
    split_idx = None
    adj1, adj2 = split_adj(adj, split_idx)
    graph1 = nx.from_scipy_sparse_matrix(adj1)
    graph2 = nx.from_scipy_sparse_matrix(adj2)
    # create its self-defined graph class from networkx graphs
    g1 = Graph(graph1.number_of_nodes(), graph1.number_of_edges()*2)
    edge_cnt = 0
    for node in range(graph1.number_of_nodes()): # get the neighbor of each node
        g1.node_wgt[node] = 1
        for n in graph1.neighbors(node):
            try:
                g1.adj_list[edge_cnt] = n
            except:
                pdb.set_trace()
            g1.adj_wgt[edge_cnt] = 1.0
            edge_cnt += 1
        g1.adj_idx[node+1] = edge_cnt
    g1.A = graph_to_adj(g1, self_loop=False)
    g2 = Graph(graph2.number_of_nodes(), graph2.number_of_edges()*2)
    edge_cnt = 0
    for node in range(graph2.number_of_nodes()): # get the neighbor of each node
        g2.node_wgt[node] = 1
        for n in graph2.neighbors(node):
            try:
                g2.adj_list[edge_cnt] = n
            except:
                pdb.set_trace()
            g2.adj_wgt[edge_cnt] = 1.0
            edge_cnt += 1
        g2.adj_idx[node+1] = edge_cnt
    g2.A = graph_to_adj(g2, self_loop=False)
    graphs = [g1, g2]
    mapping = None
    return graphs, mapping



#Split adjacency matrix in two
def split_adj(combined_adj, split_index = None, increasing_size = True):
	if split_index is None: split_index = int(combined_adj.shape[0] / 2) #default: assume graphs are same size
	if sp.issparse(combined_adj):
		if not combined_adj.getformat() != "csc": combined_adj = combined_adj.tocsc() #start off with csc so that we end up as csr
		adj1 = combined_adj[:,:split_index]; adj2 = combined_adj[:,split_index:] #select columns as csc bc faster
		adj1 = adj1.tocsr(); adj2 = adj2.tocsr() #convert to CSR for fast row slicing
		adj1 = adj1[:split_index]; adj2 = adj2[split_index:]
	else:
		adj1 = combined_adj[:split_index,:split_index]
		adj2 = combined_adj[split_index:,split_index:]

	#Align larger graph to smaller one
	if increasing_size and adj1.shape[0] < adj2.shape[0]:
		tmp = adj1
		adj1 = adj2
		adj2 = tmp

	return adj1, adj2