#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 

"""

from scipy.io import loadmat
from scipy.sparse import issparse
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

DataPath = '../data/PPI'


def process_small_mat(datapath):
    data = loadmat(datapath, appendmat=True)
    print(data.keys())
    network = data["network"].toarray()
    group = data["group"].toarray()

    print(np.shape(network))
    print(np.shape(group))

    fr_adjlist = open(datapath + '.adjlist', 'w')
    for id in range(0, np.size(network, axis=0)):
        fr_adjlist.write('{}'.format(id))
        for nid in np.where(network[id] == 1)[0]:
            fr_adjlist.write('\t{}'.format(nid))
        fr_adjlist.write('\n')
    fr_adjlist.close()

    fr_edgelist = open(datapath + '.edgelist', 'w')
    rows, columns = np.where(network == 1)
    for i in range(len(rows)):
        fr_edgelist.write('{}\t{}\n'.format(rows[i], columns[i]))
    fr_edgelist.close()

    fr_label = open(datapath + '.labels', 'w')
    for id in range(0, np.size(group, axis=0)):
        fr_label.write('{}'.format(id))
        for l in np.where(group[id] == 1)[0]:
            fr_label.write('\t{}'.format(l))
        fr_label.write('\n')
    fr_label.close()


def process_spare_mat(datapath):
    data = loadmat(datapath, appendmat=True)
    print(data.keys())

    network = data["network"]
    group = data["group"]

    print(np.shape(network))
    print(np.shape(group))

    if not issparse(network):
        raise Exception("Dense matrices not yet supported.")

    fr_adjlist = open(datapath + '.adjlist', 'w')
    fr_edgelist = open(datapath + '.edgelist', 'w')
    ids_adjlist = {}
    cx = network.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        if i not in ids_adjlist.keys():
            ids_adjlist[i] = set([j])
        else:
            ids_adjlist[i].add(j)
        fr_edgelist.write('{}\t{}\n'.format(i, j))
    fr_edgelist.close()
    for id, adjlist in sorted(ids_adjlist.items(), key=lambda item: item[0]):
        fr_adjlist.write('{}'.format(id))
        for nid in sorted(adjlist):
            fr_adjlist.write('\t{}'.format(nid))
        fr_adjlist.write('\n')
    fr_adjlist.close()

    fr_label = open(datapath + '.labels', 'w')
    ids_labelslist = {}
    cx = group.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        if i not in ids_labelslist.keys():
            ids_labelslist[i] = set([j])
        else:
            ids_labelslist[i].add(j)
    for id, labelslist in sorted(ids_labelslist.items(), key=lambda item: item[0]):
        fr_label.write('{}'.format(id))
        for nid in sorted(labelslist):
            fr_label.write('\t{}'.format(nid))
        fr_label.write('\n')
    fr_label.close()


### process_Pubmed
def process_Pubmed_cites(datapath="Pubmed-Diabetes.DIRECTED.cites.tab"):
    linelist = open(datapath).readlines()
    linelist = linelist[2:]
    line_dict = {}
    for line in linelist:
        ID = int(line.split("\t")[0].strip())
        if ID in line_dict.keys():
            print(ID)
        else:
            line_dict[ID] = line
    with open(datapath + ".sorted", "w") as fr:
        for key in sorted(line_dict.keys()):
            fr.write(line_dict[key])


def process_Pubmed_paper(datapath="Pubmed-Diabetes.NODE.paper.tab"):
    fr = open(datapath)
    linelist = fr.readline()
    linelist = fr.readline()
    linelist = linelist.strip().split("\t")
    word_list = []
    for line in linelist:
        items = [v.strip() for v in line.split(":")]
        if items[0] == "numeric":
            word_list.append(items[1])
    for idx, word in enumerate(word_list):
        print("{}\t{}".format(idx, word))


def process_Pubmed():
    linelist = open("Pubmed-Diabetes.DIRECTED.cites.tab").readlines()
    linelist = linelist[2:]
    edge_set = set()
    for line in linelist:
        nodes = line.strip().split("\t")
        source = nodes[1].split(":")[1].strip()
        target = nodes[3].split(":")[1].strip()
        if (source, target) in edge_set:
            print((source, target))
        else:
            edge_set.add((source, target))
    with open("pubmed.edgelist", "w") as fr:
        for v in edge_set:
            fr.write("{}\t{}\n".format(v[0], v[1]))

    linelist = open("Pubmed-Diabetes.NODE.paper.tab").readlines()
    word_list = []
    for line in linelist[1].strip().split("\t"):
        items = [v.strip() for v in line.split(":")]
        if items[0] == "numeric":
            word_list.append(items[1])
    features_dim = len(word_list)
    print("features_dim={}\n".format(features_dim))
    word_idx = {}
    for idx, word in enumerate(word_list):
        word_idx[word] = idx

    node_feat = set()
    fr_feat = open("pubmed.features", "w")
    fr_label = open("pubmed.labels", "w")
    for line in linelist[2:]:
        terms = line.strip().split("\t")
        ID = terms[0].strip()
        node_feat.add(ID)
        label = int(terms[1].split("=")[1].strip()) - 1
        feature = ["0"] * features_dim
        for word_term in terms[2:-1]:
            word = word_term.split("=")[0].strip()
            value = word_term.split("=")[1].strip()
            feature[word_idx[word]] = value
        fr_feat.write(ID)
        for v in feature:
            fr_feat.write("\t" + v)
        fr_feat.write("\n")
        fr_label.write(ID + "\t" + str(label) + "\n")
    fr_feat.close()
    fr_label.close()

    for v in edge_set:
        if v[0] not in node_feat:
            print(v[0])
        if v[1] not in node_feat:
            print(v[1])


if __name__ == '__main__':
    process_Pubmed()
