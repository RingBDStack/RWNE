#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import tensorflow as tf



def distance(v1,v2,metric = "euclid"):
    """
    calculate two vectors' distance.
    :param metric: "euclid", "cosine".
    :return:
    """
    dis = None
    if metric == "euclid":
        dis = np.sqrt(np.sum(np.power(v1-v2, 2)))
    elif metric == "cosine":
        dis = np.sum(np.multiply(v1,v2)) / np.sqrt(np.sum(np.power(v1,2)) * np.sum(np.power(v2,2)))
    return dis

def sigmod(v):
    with tf.Session():
        return tf.sigmoid(v).eval()


def distance_probablity(syn0, syn1, metric = "euclid"):
    with tf.Graph().as_default(), tf.device('/gpu:0'):
        syn0 = tf.Constant(syn0,dtype=tf.float32)
        syn1 = tf.Constant(syn1,dtype=tf.float32)
        prob_tensor = tf.sigmoid(tf.matmul(syn0,syn1,transpose_b=True))
        if metric == "euclid":
            syn0 - syn0

def evaluate_syn0_syn1(sg_model_path, metric="cosine"):
    model = word2vec.Word2Vec.load(sg_model_path)
    # probablity
    probablity = sigmod(np.matmul(model.wv.syn0,np.transpose(model.syn1neg)))
    data = set()
    for i in range(np.size(probablity,axis=0)):
        for j in range(np.size(probablity,axis=0)):
            x = round(probablity[i,j],3)
            y = round(distance(model.wv.syn0[i],model.wv.syn0[j], metric = metric),3)
            data.add((x,y))
    X = []
    Y = []
    for item in data:
        X.append(item[0])
        Y.append(item[1])
    fig = plt.figure()
    plt.scatter(X,Y)
    plt.xlabel("joint probablity")
    plt.ylabel(metric+" distance")
    name = os.path.join( os.path.split(sg_model_path)[0], metric+"_probablity")
    plt.savefig(name + ".pdf", format='pdf', bbox_inches='tight')
    plt.savefig(name + ".png", format='png', bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    # evaluate_syn0_syn1(sys.argv[1])
    # evaluate_syn0_syn1("./data/blogcatalog.model",metric="cosine")
    path = "./data/blogcatalog.model"
    metric = "cosine"
    arg = sys.argv
    if len(arg)>1:
        path = arg[1]
        if len(arg) > 2:
            metric = arg[2]
    evaluate_syn0_syn1(path,metric)


