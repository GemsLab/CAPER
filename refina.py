import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

import time
from collections import defaultdict
import math

from refina_utils import threshold_alignment_matrix, score_alignment_matrix, kd_align, score_MNC
import pdb

def refina(alignment_matrix, adj1, adj2, args, iter, true_alignments = None):
    '''Automatically set token match'''
    if args.token_match < 0: #automatically select
        #reciprocal of smallest power of 10 larger than largest graph #nodes
        pow_10 = math.log(max(adj1.shape[0], adj2.shape[0]), 10)
        args.token_match = 10**-int(math.ceil(pow_10))	

    #alignment_matrix = threshold_alignment_matrix(alignment_matrix, topk = args.init_threshold)

    for i in range(iter):
        '''DIAGNOSTIC/DEMO ONLY: keep track of alignment quality'''
        if alignment_matrix.shape[0] < 20000: #don't bother with intermediate diagnostics for big matrices
            print(("Scores after %d refinement iterations" % i))
            if true_alignments is not None:
                score, _ = score_alignment_matrix(alignment_matrix, true_alignments = true_alignments)
                print("Top 1 accuracy: %.5f" % score)
            mnc = score_MNC(alignment_matrix, adj1, adj2)
            print("MNC: %.5f" % mnc)

        '''Step 1: compute MNC-based update'''
        update = compute_update(adj1, adj2, alignment_matrix, args)
        update = compute_update(adj1, adj2, alignment_matrix, args)#min( int(5*(i+1)), adj1.shape[0]) )
		
        '''Step 2: apply update and token match'''
        if args.n_update > 0: #add token match score here so we can selectively update
            if sp.issparse(alignment_matrix):
                nonzero_updates = update.nonzero() #Indices of alignments to update
                updated_data = np.asarray(alignment_matrix[nonzero_updates]) #Alignment values we want to update
                updated_data += args.token_match #Add token match
                updated_data *= update.data #Multiplicatively update them

                alignment_matrix = alignment_matrix.tolil()
                alignment_matrix[nonzero_updates] = updated_data
                alignment_matrix.tocsr()
            else:
                alignment_matrix[update != 0] += args.token_match
                alignment_matrix[update != 0] *= update[update != 0]
        else:
            alignment_matrix = alignment_matrix * update
            alignment_matrix += args.token_match

        '''Step 3: normalize'''
        alignment_matrix = normalize_alignment_matrix(alignment_matrix)

    return alignment_matrix

def compute_update(adj1, adj2, alignment_matrix, args):
    update_matrix = adj1.dot(alignment_matrix).dot(adj2.T) #row i: counterparts of neighbors of i

    if args.n_update > 0 and args.n_update < adj1.shape[0]:
        if sp.issparse(update_matrix): 
            if update_matrix.shape[0] < 120000: update_matrix = update_matrix.toarray() #still fits in memory and dense is faster 
            update_matrix = threshold_alignment_matrix(update_matrix, topk = args.n_update, keep_dist = True)
            update_matrix = sp.csr_matrix(update_matrix)
        else:
            update_matrix = threshold_alignment_matrix(update_matrix, topk = args.n_update, keep_dist = True)
    return update_matrix

def normalize_alignment_matrix(alignment_matrix):
    alignment_matrix = normalize(alignment_matrix, norm = "l1", axis = 1)
    alignment_matrix = normalize(alignment_matrix, norm = "l1", axis = 0)
    return alignment_matrix