#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 

remote：
    https://www.jianshu.com/p/9424fbc76c2c

"""

import numpy as np
from collections import defaultdict
import tensorflow as tf
import time
import os
import pickle
import utils
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import random
import utils




#
# with tf.Session() as sess:
    # tensor_a = tf.constant(np.array([[v,v,v,v,v] for v in range(20)]))
    # res = tf.nn.embedding_lookup(tensor_a, [[0], [1], [2]])
    # res = sess.run(res)
    # print(res)
    # print(np.shape(res))
    # res = tf.reduce_sum(tensor_a, 1)
    # res = sess.run(res)
    # print(res)
    # labels_matrix = tf.reshape(tf.cast(tf.constant(np.array([0,1,2,3])), dtype=tf.int64), [1, -1])
    # sampled_ids, _, _ = tf.nn.uniform_candidate_sampler(
    #     true_classes=labels_matrix,
    #     num_true=4,
    #     num_sampled=5,
    #     unique=True,
    #     range_max=10,
    #     seed=utils.get_random_seed())
    # res = sess.run(sampled_ids)
    # print(res)
    # tf.nn.nce_loss()

    # tensor_embed = tf.constant([[[v,v,v,v,v] for v in range(4)],[[v,v,v,v,v] for v in range(4,8)]],dtype=tf.float32) # 2* 4 *5
    # embed = sess.run(tensor_embed)
    # print("embed=\n{}\nwith shape={}".format(embed, np.shape(embed)))
    #
    # tensor_nce_w_true = tf.constant([[v,v,v,v,v] for v in range(100,201,100)],dtype=tf.float32) # 2*5
    # tensor_nce_b_true = tf.constant([1,2],dtype=tf.float32) # 2
    # nce_w_true = sess.run(tensor_nce_w_true)
    # nce_b_true = sess.run(tensor_nce_b_true)
    # print("nce_w_true=\n{}\nwith shape={}".format(nce_w_true, np.shape(nce_w_true)))
    # print("nce_b_true=\n{}\nwith shape={}".format(nce_b_true, np.shape(nce_b_true)))
    # tensor_nce_w_true = tf.expand_dims(tensor_nce_w_true, -1) # 2*5*1
    # tensor_nce_b_true = tf.expand_dims(tensor_nce_b_true, -1) # 2*1
    # nce_w_true = sess.run(tensor_nce_w_true)
    # nce_b_true = sess.run(tensor_nce_b_true)
    # print("nce_w_true_expand=\n{}\nwith shape={}".format(nce_w_true, np.shape(nce_w_true)))
    # print("nce_b_true_expand=\n{}\nwith shape={}".format(nce_b_true, np.shape(nce_b_true)))
    # tensor_true_logits = tf.matmul(tensor_embed, tensor_nce_w_true) # 2*4*1
    # true_logits = sess.run(tensor_true_logits) #
    # print("true_logits=\n{}\nwith shape={}".format(true_logits, np.shape(true_logits)))
    # tensor_true_logits = tf.squeeze(tensor_true_logits, -1) # 2*4
    # true_logits = sess.run(tensor_true_logits)  #
    # print("true_logits_squeeze=\n{}\nwith shape={}".format(true_logits, np.shape(true_logits)))
    # tensor_true_logits = tensor_true_logits + tensor_nce_b_true # 2*4
    # true_logits = sess.run(tensor_true_logits)  #
    # print("true_logits_add_b=\n{}\nwith shape={}".format(true_logits, np.shape(true_logits)))
    # tensor_true_xent = tf.nn.sigmoid_cross_entropy_with_logits( labels=tf.ones_like(tensor_true_logits), logits=tensor_true_logits) # 2 *4
    # true_xent = sess.run(tensor_true_xent)  #
    # print("true_xent=\n{}\nwith shape={}".format(true_xent, np.shape(true_xent)))

    # tensor_nce_w_neg = tf.constant([[[v,v,v,v,v] for v in range(100,301,100)],[[v,v,v,v,v] for v in range(400,601,100)]],dtype=tf.float32) # 2* 3 *5
    # tensor_nce_b_neg = tf.constant([[1,2,3], [11,22,33]],dtype=tf.float32)  # 2 * 3
    # nce_w_neg = sess.run(tensor_nce_w_neg)
    # nce_b_neg = sess.run(tensor_nce_b_neg)
    # print("nce_w_neg=\n{}\nwith shape={}".format(nce_w_neg, np.shape(nce_w_neg)))
    # print("nce_b_neg=\n{}\nwith shape={}".format(nce_b_neg, np.shape(nce_b_neg)))
    # tensor_nce_w_neg = tf.transpose(tensor_nce_w_neg, perm=[0, 2, 1])  # 2*5*3
    # tensor_nce_b_neg = tf.expand_dims(tensor_nce_b_neg, -2)  # 2 *1* 3
    # nce_w_neg = sess.run(tensor_nce_w_neg)
    # nce_b_neg = sess.run(tensor_nce_b_neg)
    # print("nce_w_neg_transpose=\n{}\nwith shape={}".format(nce_w_neg, np.shape(nce_w_neg)))
    # print("nce_b_neg_expand=\n{}\nwith shape={}".format(nce_b_neg, np.shape(nce_b_neg)))
    # tensor_neg_logits = tf.matmul(tensor_embed, tensor_nce_w_neg)  # 2*4*3
    # neg_logits = sess.run(tensor_neg_logits)  #
    # print("neg_logits=\n{}\nwith shape={}".format(neg_logits, np.shape(neg_logits)))
    # tensor_neg_logits = tensor_neg_logits + tensor_nce_b_neg  # 2*4*3
    # neg_logits = sess.run(tensor_neg_logits)  #
    # print("neg_logits_add_b=\n{}\nwith shape={}".format(neg_logits, np.shape(neg_logits)))
    # tensor_neg_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(tensor_neg_logits),
    #                                                            logits=tensor_neg_logits)  # 2*4*3
    # neg_xent = sess.run(tensor_neg_xent)  #
    # print("neg_xent=\n{}\nwith shape={}".format(neg_xent, np.shape(neg_xent)))







# mtime_before = os.stat('eval_classify.py').st_mtime
# print(mtime_before)
# time.sleep(10)
# mtime = os.stat('eval_classify.py').st_mtime
# print(mtime)
# print(mtime==mtime_before)
# fr = open("xxxxx.md",'r')
# fr.write("1 2 3 4 \n 5 6 7 8")
# fr.readline()
# fr.close()
# os.utime('xxxxx.md',None)
# print(time.time() - os.stat('xxxxx.md').st_ctime) # create
# print(time.time() - os.stat('xxxxx.md').st_mtime) # modify
# print(time.time() - os.stat('xxxxx.md').st_atime) # access




# nodeID_list = [3, 1, 0, 2]
# similarity = np.array([
#     [1,   0,   0.8, 0.5],
#     [0,   1,   0.5, 0.7],
#     [0.8, 0.5, 1,   0.2],
#     [0.5, 0.7, 0.2, 1]
# ])
# sortedInd = np.argsort(similarity.reshape(-1))
# sortedInd = sortedInd[::-1]
# count_hitting = 0
# count_predict = 0
# precisionK = []
# for ind in sortedInd:
#     x, y = divmod(ind, len(nodeID_list))
#     nodeID_s = nodeID_list[x]
#     nodeID_t = nodeID_list[y]
#     print("{},{}: {}".format(nodeID_s, nodeID_t, similarity[x,y]))

#
# alias_edges = {}
# # normalized_probs = [0.1,0.2,0.3,0.4]
# # alias_edges[(0, 0)] = utils.alias_setup(normalized_probs)
# # normalized_probs = [0.1,0.5,0.4]
# # alias_edges[(1, 1)] = utils.alias_setup(normalized_probs)
# # pickle.dump(alias_edges, open("data/test.pickle","wb"), -1)
#
# alias_edges = pickle.load(open("data/test.pickle","rb"))
#
# print(alias_edges)
#
# print("end!")

# a = 100000000
# aaaaa = list(range(a))
# time_start = time.time()
# b = np.random.choice(aaaaa, size=5, replace=False)
# print(b)
# print(time.time()-time_start)
# time_start = time.time()
# b = random.sample(aaaaa, k=5)
# print(b)
# print(time.time()-time_start)
# time_start = time.time()
# b = random.sample(range(a), k=5)
# print(b)
# print(time.time()-time_start)

# data = np.arange(100000000)
# num_examples = len(data)
# # print(data)
# time_start = time.time()
# perm = np.arange(num_examples)
# np.random.shuffle(perm)
# data = data[perm]
# # print(data)
# print(time.time()-time_start)
# time_start = time.time()
# perm = np.arange(num_examples)
# random.shuffle(perm)
# data = data[perm]
# # print(data)
# print(time.time()-time_start)



# def cos_sim(vector_a, vector_b):
#     """
#     计算两个向量之间的余弦相似度
#     :param vector_a: 向量 a
#     :param vector_b: 向量 b
#     :return: sim
#     """
#     vector_a = np.mat(vector_a)
#     vector_b = np.mat(vector_b)
#     num = float(vector_a * vector_b.T)
#     denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
#     cos = num / denom
#     # sim = 0.5 + 0.5 * cos
#     return cos
#
# X = [432,2,3,432,5]
# Y = [-543,564,3,-32,0]
# cosine = cos_sim(X, Y)
# print(cosine)
# tensor_X = tf.constant(X,dtype=tf.float32)
# tensor_Y = tf.constant(Y,dtype=tf.float32)
# tensor_cosine = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(tensor_X,dim=0), tf.nn.l2_normalize(tensor_Y,dim=0)))
# with tf.Session() as sess:
#     print(sess.run(tensor_cosine))


a = np.array([1,2],dtype=int)
b = np.array([11,22],dtype=int)
c = np.add(a,b,dtype=float)
print(c)









