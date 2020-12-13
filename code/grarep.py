#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 

mainly reference from ThuNLP

"""
import os
import time
import logging
import numpy as np
from numpy import linalg as la
from sklearn.preprocessing import normalize

import utils
import network


logger = logging.getLogger("NRL")


class GraRep(object):
    def __init__(self, nodes_size, edges_list, embedding_size = 128, Kstep = 1):
        self._nodes_size = nodes_size
        self._Kstep = Kstep
        assert embedding_size % Kstep == 0, 'error: dim({}) % Kstep({}) != 0'.format(embedding_size, Kstep)
        self._embedding_size = embedding_size // Kstep
        self._adj_matrix = np.zeros([nodes_size, nodes_size], dtype=np.float32)
        for x, y in edges_list:
            self._adj_matrix[x, y] = 1.
            self._adj_matrix[y, x] = 1.
        # ScaleSimMat
        self._adj_matrix = np.matrix(self._adj_matrix / np.sum(self._adj_matrix, axis=1))

    def _getProbTranMat(self,Ak):
        probTranMat = np.log(Ak / np.tile( np.sum(Ak, axis=0), (self._nodes_size, 1))) - np.log(1.0 / self._nodes_size)
        probTranMat[probTranMat < 0] = 0
        probTranMat[probTranMat == np.nan] = 0
        return probTranMat

    def _getRepUseSVD(self, probTranMat, alpha):
        U, S, VT = la.svd(probTranMat)
        Ud = U[:, 0:self._embedding_size]
        Sd = S[0:self._embedding_size]
        return np.array(Ud) * np.power(Sd, alpha).reshape((self._embedding_size))

    def train(self):
        Ak = np.matrix(np.identity(self._nodes_size))
        VecMat = np.zeros((self._nodes_size, self._embedding_size * self._Kstep))
        for i in range(self._Kstep):
            Ak = np.dot(Ak, self._adj_matrix)
            probTranMat = self._getProbTranMat(Ak)
            Rk = self._getRepUseSVD(probTranMat, 0.5)
            Rk = normalize(Rk, axis=1, norm='l2')
            VecMat[:, self._embedding_size * i:self._embedding_size * (i + 1)] = Rk[:, :]
        # get embeddings
        return VecMat


def save_word2vec_format(vector_path, vectors, idx2vocab):
    vocab_size = np.size(vectors, axis=0)
    vector_size = np.size(vectors, axis=1)
    with open(vector_path, 'w') as fr:
        fr.write('%d %d\n' % (vocab_size, vector_size))
        for index in range(vocab_size):
            fr.write('%d ' % idx2vocab[index])
            for one in vectors[index]:
                fr.write('{} '.format(one))
            fr.write('\n')


def train_vectors(options):
    if not utils.check_rebuild(options.vectors_path, descrip='vectors', always_rebuild=options.always_rebuild):
        return
    train_vec_dir = os.path.split(options.vectors_path)[0]
    if not os.path.exists(train_vec_dir):
        os.makedirs(train_vec_dir)


    # construct network
    net = network.construct_network(options)

    Kstep = 2

    # train info
    logger.info('Train info:')
    logger.info('\t train_model = {}'.format(options.model))
    logger.info('\t total embedding nodes = {}'.format(net.get_nodes_size()))
    logger.info('\t total edges = {}'.format(net.get_edges_size()))
    logger.info('\t embedding size = {}'.format(options.embedding_size))
    logger.info('\t Kstep = {}'.format(Kstep))
    logger.info('\t vectors_path = {}'.format(options.vectors_path))

    fr_vec = open(os.path.join(train_vec_dir, 'embedding.info'), 'w')
    fr_vec.write('embedding info:\n')
    fr_vec.write('\t train_model = {}\n'.format(options.model))
    fr_vec.write('\t total embedding nodes = {}\n'.format(net.get_nodes_size()))
    fr_vec.write('\t total edges = {}\n'.format(net.get_edges_size()))
    fr_vec.write('\t embedding size = {}\n'.format(options.embedding_size))
    fr_vec.write('\t Kstep = {}\n'.format(Kstep))
    fr_vec.write('\t vectors_path = {}\n'.format(options.vectors_path))
    fr_vec.close()


    # train
    logger.info('training...')
    time_start = time.time()
    grarep = GraRep(net.get_nodes_size(), net.edges, options.embedding_size, Kstep)
    vecs = grarep.train()
    save_word2vec_format(options.vectors_path, vecs, net._idx_nodes)
    logger.info('train completed in {}s'.format(time.time() - time_start))
    return










