#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""
import sys
import os
import time
import logging

import skipgram
import TF_rwne
import TF_line
import TF_sdne
import TF_dngr
import grarep
import TF_gcn
import utils


logger = logging.getLogger("NRL")


# train vectors
def train_vectors(options, sens = None):
    if not utils.check_rebuild(options.vectors_path, descrip='embedding vectors', always_rebuild=options.always_rebuild):
        return

    if options.model == 'DeepWalk' or options.model == 'Node2Vec':
        # if options.using_tensorflow:
        #     TF_skipgram.train_vectors(options, sens=sens)
        # else:
        skipgram.train_vectors(options, sens= sens)
    elif options.model == 'LINE':
        TF_line.train_vectors(options)
    elif options.model == 'RWNE':
        TF_rwne.train_vectors(options)
    elif options.model == 'SDNE':
        TF_sdne.train_vectors(options)
    elif options.model == 'DNGR':
        TF_dngr.train_vectors(options)
    elif options.model == 'GraRep':
        grarep.train_vectors(options)
    elif options.model == 'GCN':
        TF_gcn.train_vectors(options)
    else:
        logger.error("Unknown model for embedding: '%s'. "% options.model+
                     "Valid models: 'DeepWalk', 'Node2Vec', 'LINE', 'RWNE'.")
        sys.exit()







