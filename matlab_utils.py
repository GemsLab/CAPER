import matlab
import matlab.engine
import sys
import time
import random
import operator
import logging
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as sps
import networkx as nx
import os

def convert_to_matlab(eng, sparse_mat):
    ''' convert scipy sparse matrix to matlab matrices '''
    data = matlab.double(list(sparse_mat.data))
    # row = matlab.double(list(sparse_mat.row))
    row = eng.plus(matlab.double(list(sparse_mat.row)), matlab.double([1]))
    col = eng.plus(matlab.double(list(sparse_mat.col)), matlab.double([1]))
    # col = matlab.double(list(sparse_mat.col))
    return data, row, col
def run_matlab(adj1, adj2, init_align, configs, path, attribs=None,
               method='netalign'):
    '''
    Run Netalign on the given graph1, graph2, and initial alignment matrix.
    '''
    # start matlab engine
    # os.environ['TMP'] = r'temp_folder'
    # os.environ['TEMP'] = r'temp_folder'
    #eng = matlab.engine.find_matlab()
    eng = matlab.engine.start_matlab()
    if method == 'final':
        eng.addpath(path['FINAL'])
    else:
        eng.addpath(eng.genpath('graclus1.2(linux)'))
        eng.addpath(path['MOANA'])
    if init_align != None:
        print('maximum values in init align row is: ', init_align.row.max())
        print('maximum values in init align col is: ', init_align.col.max())
    # pass sparse matrix to matlab
    num_nodes1 = adj1.shape[0]
    adjmat1 = sps.coo_matrix(adj1)
    adjmat2 = sps.coo_matrix(adj2)
    adj1_data, adj1_row, adj1_col = convert_to_matlab(eng, adjmat1)
    adj2_data, adj2_row, adj2_col = convert_to_matlab(eng, adjmat2)
    if init_align != None:
        ia_data, ia_row, ia_col = convert_to_matlab(eng, init_align)
    else:
        ia_data = []
        ia_row = 0
        ia_col = 0
    # run netalign
    similarity_matrix = None
    if method == 'final':
        logging.debug('Running %s ...' % method)
        if attribs is not None:
            # input data contains attributes
            # split the attribs for graph1 and graph2
            print('Attributes exists! feeding attributes')
            attribs1 = matlab.double(attribs[:num_nodes1,:].tolist())
            attribs2 = matlab.double(attribs[num_nodes1:,:].tolist())
        else:
            # input data does not contain attributes
            attribs1 = matlab.double(None)
            attribs2 = matlab.double(None)
        before_final = time.time()
        ma, mb, similarity_matrix = eng.run_final(adj1_row, adj1_col, adj1_data,
                                         adj2_row, adj2_col, adj2_data,
                                         ia_row, ia_col, ia_data,
                                         attribs1, attribs2,
                                         float(configs['alpha']),
                                         float(configs['maxiter']),
                                         float(configs['tol']),
                                         nargout=3)
        after_final = time.time()
        run_time = after_final - before_final
        eng.eval('exception = MException.last;', nargout=0)
        eng.eval('getReport(exception)')
        ma = np.asarray(ma).flatten() - 1
        mb = np.asarray(mb).flatten() - 1
    elif method == 'moana':
        logging.debug('Running %s ...' % method)
        if attribs is not None:
            # input data contains attributes
            # split the attribs for graph1 and graph2
            print('Attributes exists! feeding attributes')
            attribs1 = matlab.double(attribs[:num_nodes1,:].tolist())
            attribs2 = matlab.double(attribs[num_nodes1:,:].tolist())
        else:
            # input data does not contain attributes
            attribs1 = matlab.double(None)
            attribs2 = matlab.double(None)
        before_final = time.time()
        ma, mb, similarity_matrix = eng.run_moana(adj1_row, adj1_col, adj1_data,
                                         adj2_row, adj2_col, adj2_data,
                                         ia_row, ia_col, ia_data,
                                         attribs1, attribs2,
                                         float(configs['alpha']),
                                         float(configs['maxiter']),
                                         float(configs['tol']),
                                         nargout=3)
        after_final = time.time()
        run_time = after_final - before_final
        eng.eval('exception = MException.last;', nargout=0)
        eng.eval('getReport(exception)')
        ma = np.asarray(ma).flatten() - 1
        mb = np.asarray(mb).flatten() - 1        
    else:
        logging.error('Invalid method %s !', method)
        print("ERROR: Invalid method %s", method)
        sys.exit(-1)
    logging.debug('Complete.')
    return dict(zip(mb.tolist(), ma.tolist())), float(run_time), similarity_matrix
