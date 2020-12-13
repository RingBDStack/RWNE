#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""

import scipy.io as scio
import numpy as np

datapath = '../data/blogcatalog'
data = scio.loadmat(datapath, appendmat = True)
print(data.keys())
network = data["network"].toarray()
group = data["group"].toarray()

print(np.shape(network))
print(np.shape(group))

fr_adjlist = open(datapath+'.adjlist', 'w')
for id in range(0,np.size(network,axis=0)):
    fr_adjlist.write('{}'.format(id))
    for nid in np.where(network[id]==1)[0]:
        fr_adjlist.write('\t{}'.format(nid))
    fr_adjlist.write('\n')
fr_adjlist.close()

fr_edgelist = open(datapath+'.edgelist', 'w')
rows, columns = np.where(network==1)
for i in range(len(rows)):
    fr_edgelist.write('{}\t{}\n'.format(rows[i],columns[i]))
fr_edgelist.close()

fr_label = open(datapath+'.labels', 'w')
for id in range(0,np.size(group,axis=0)):
    fr_label.write('{}'.format(id))
    for l in np.where(group[id]==1)[0]:
        fr_label.write('\t{}'.format(l))
    fr_label.write('\n')
fr_label.close()
