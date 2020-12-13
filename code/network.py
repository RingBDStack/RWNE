#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""
import logging
import sys
import os
import time
from collections import Iterable
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import utils

logger = logging.getLogger("NRL")


class Graph(object):
    """
    an graph/network.
    each node in the graph/network is represented as an interger ID, and the node ID start from 0.
    each edge in the graph/network is directed, which means an undirected edge (u,v) will have two node pairs <u,v> and <v,u>.
    """
    def __init__(self, isdirected = False, isweighted = False, self_looped = False):
        self._isdirected = isdirected
        self._isweighted = isweighted
        self._self_looped = self_looped
        self._nodes_adjlist = dict()  # all saved in ID.
        self._nodes_id = {}
        self._id_nodes = []
        # self._idx_nodes = []
        # self._degrees = []

    @property
    def isdirected(self):
        return self._isdirected
    @property
    def nodes(self):
        return self._nodes_adjlist.keys() # also: return range(len(self._nodes_adjlist))
    @property
    def edges(self):
        edges_list = []
        edges_distribution = [] # future work for weighted network.
        for source_node, adj_list in self._nodes_adjlist.items():
            for target_node in adj_list:
                edges_list.append([source_node,target_node])  # (source, target)
        return edges_list
    @property
    def degrees(self):
        return [self.get_degrees(v) for v in self.nodes]
    def get_nodes_size(self):
        return len(self._nodes_adjlist.keys())
    def get_edges_size(self):
        if self._isdirected:
            return sum([self.get_degrees(v) for v in self._nodes_adjlist.keys()])
        else: return sum([self.get_degrees(v) for v in self._nodes_adjlist.keys()])//2
    def get_degrees(self,nodes):
        if isinstance(nodes, Iterable):
            return {v: len(self._nodes_adjlist[v]) for v in nodes}
        else:
            return len(self._nodes_adjlist[nodes])
    def get_adjlist(self,nodes):
        if isinstance(nodes, Iterable):
            return {v: self._nodes_adjlist[v] for v in nodes}
        else:
            return self._nodes_adjlist[nodes]
    def make_consistent(self, remove_isolated = True):
        logger.info('make_consistent ...')
        time_start = time.time()
        # self.remove_self_loops()
        if remove_isolated:
            self.remove_isolated_nodes()
        sorted_nodes_adjlist = {}
        self._nodes_id = {}
        self._id_nodes = sorted(self._nodes_adjlist.keys())
        for id, node in enumerate(self._id_nodes):
            self._nodes_id[node] = id
        for k, v in self._nodes_adjlist.items():
            sorted_nodes_adjlist[self._nodes_id[k]] = sorted([self._nodes_id[node] for node in v])
        self._nodes_adjlist = sorted_nodes_adjlist
        logger.info('make consistent completed in {}s'.format(time.time() - time_start))
    def remove_self_loops(self):
        removed = 0
        time_start = time.time()
        for node in self._nodes_adjlist.keys():
            if node in self._nodes_adjlist[node]:
                self._nodes_adjlist[node].remove(node)
                removed += 1
        logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (time.time() - time_start)))
    def remove_isolated_nodes(self):
        isolated_nodes = set()
        time_start = time.time()
        nodeID_degrees = self.get_degrees(self.nodes)
        if self._isdirected:
            target_nodes = set()
            for tar_list in self._nodes_adjlist.values():
                for tar in tar_list:
                    target_nodes.add(tar)
            for v, deg in nodeID_degrees.items():
                if deg == 0 and v not in target_nodes:
                    isolated_nodes.add(v)
        else:
            for v, deg in nodeID_degrees.items():
                if deg == 0:
                    isolated_nodes.add(v)
        for v in isolated_nodes:
            self._nodes_adjlist.pop(v)
        logger.info('remove_isolated_nodes: removed {} isolated_nodes in {}s'.format(len(isolated_nodes), (time.time() - time_start)))
    def has_edge(self, start, end):
        if end in self._nodes_adjlist[start]:
            return True
        if not self._isdirected and start in self._nodes_adjlist[end]:
            return True
        return False
    def add_single_node(self, nodeID):
        if nodeID not in self._nodes_adjlist:
            self._nodes_adjlist[nodeID] = []
    def _add_single_edge_directed(self, source_nodeID, target_nodeID):
        if target_nodeID not in self._nodes_adjlist[source_nodeID]:
            self._nodes_adjlist[source_nodeID].append(target_nodeID)
    def add_single_edge(self, source_nodeID, target_nodeID):
        self.add_single_node(source_nodeID)
        if not self._self_looped and source_nodeID == target_nodeID: #
            return
        self.add_single_node(target_nodeID)
        self._add_single_edge_directed(source_nodeID, target_nodeID)
        if not self._isdirected:
            self._add_single_edge_directed(target_nodeID, source_nodeID)
    def print_net_info(self, file_path='./tmp/net/net.info', edges_file=None):
        if not os.path.exists(os.path.split(file_path)[0]):
            os.makedirs(os.path.split(file_path)[0])
        fr = open(file_path, 'w')
        fr.write('constructed network from {}\n'.format(edges_file))
        fr.write('network info::\n')
        fr.write('\t isdirected: {}\n'.format(str(self._isdirected)))
        fr.write('\t isweighted: {}\n'.format(str(self._isweighted)))
        fr.write('\t self_looped: {}\n'.format(str(self._self_looped)))
        fr.write('\t nodes size: {}\n'.format(self.get_nodes_size()))
        fr.write('\t edges size: {}\n'.format(self.get_edges_size()))

        # logger.info('constructed network form {}'.format(edges_file))
        logger.info('network info::')
        logger.info('\t isdirected: {}'.format(str(self._isdirected)))
        logger.info('\t isweighted: {}'.format(str(self._isweighted)))
        logger.info('\t self_looped: {}'.format(str(self._self_looped)))
        logger.info('\t nodes size: {}'.format(self.get_nodes_size()))
        logger.info('\t edges size: {}'.format(self.get_edges_size()))

        nodeID_degrees = self.get_degrees(self.nodes)
        degrees = np.array(list(nodeID_degrees.values()),dtype=np.int32)
        degree_max = np.max(degrees)
        degree_mean = np.mean(degrees)
        degree_median = np.median(degrees)
        degree_min = np.min(degrees)
        degree_zero_count = np.sum(degrees==0)
        fr.write('\t max degree: {}\n'.format(degree_max))
        fr.write('\t mean degree: {}\n'.format(degree_mean))
        fr.write('\t median degree: {}\n'.format(degree_median))
        fr.write('\t min degree: {}\n'.format(degree_min))
        fr.write('\t zero degree counts: {}\n'.format(degree_zero_count))
        fr.write('-'*20+'\n'+'-'*20+'\n'+'nodes degrees details:'+'\n')
        for id in self.nodes:
            fr.write("{}\t{}\n".format(id, nodeID_degrees[id]))
        fr.close()
        logger.info('\t max degree: {}'.format(degree_max))
        logger.info('\t mean degree: {}'.format(degree_mean))
        logger.info('\t median degree: {}'.format(degree_median))
        logger.info('\t min degree: {}'.format(degree_min))
        logger.info('\t zero degree counts: {}'.format(degree_zero_count))

    def split_by_edges(self, train_ratio = 0, keep_static_nodes = True, keep_consistent_nodes = False):
        """
        split the network to two parts: one has train_ratio edges, one has 1-train_ratio edges.
        :param train_ratio:
        :param keep_static_nodes: whether the splited two parts keep the same node set as the original network.
        :param keep_consistent_nodes: whether making the splited nodes consistent by re-sorting the nodesID.
        :return: train_netwrok: with train_ratio edges, eval_netwrok: with 1-train_ratio edges.
        """
        logger.info('Net split: spliting edges to train_network and eval_network ...')
        logger.info("\t\t train_ratio = {}, keep_static_nodes = {}, keep_consistent_nodes = {}".format(train_ratio, keep_static_nodes, keep_consistent_nodes))
        logger.info("\t\t origin_edges_size = {}".format(self.get_edges_size()))
        time_start = time.time()
        edges_list = self.edges
        if not self._isdirected:
            edges_set = set()
            for source, target in edges_list:
                if (source, target) not in edges_set and (target, source) not in edges_set:
                    edges_set.add((source, target))
            edges_list = list(edges_set)

        train_edges_list, test_edges_list = train_test_split(edges_list,
                                                            test_size=1.0 - train_ratio,
                                                            random_state=utils.get_random_seed(),
                                                            shuffle=True)
        # perm = np.arange(len(edges_list))
        # random.shuffle(perm)
        # edges_list_t = [edges_list[i] for i in perm]
        # edges_list = edges_list_t
        # # split for train:
        # train_edges_size = int(np.ceil(len(edges_list)*train_ratio))
        # assert train_edges_size <= len(edges_list), "error, {} > {}".format(train_edges_size, len(edges_list))
        #
        # train network:
        train_net = Graph(isdirected=self._isdirected, isweighted=self._isweighted, self_looped=self._self_looped)
        # for source, target in edges_list[0:train_edges_size]:
        for source, target in train_edges_list:
            train_net.add_single_edge(source, target)
        if keep_static_nodes:
            for v in self.nodes:
                train_net.add_single_node(v)
        elif keep_consistent_nodes:
            train_net.make_consistent()
        logger.info("\t\t train_edges_size = {}".format(train_net.get_edges_size())) 
        
        # eval network:
        eval_net = Graph(isdirected=self._isdirected, isweighted=self._isweighted, self_looped=self._self_looped) 
        # for source, target in edges_list[train_edges_size:]:
        for source, target in test_edges_list:
            eval_net.add_single_edge(source, target)
        if keep_static_nodes:
            for v in self.nodes:
                eval_net.add_single_node(v)
        elif keep_consistent_nodes:
            eval_net.make_consistent()
        logger.info("\t\t eval_edges_size = {}".format(eval_net.get_edges_size()))
        logger.info('Net split: split edges completed in {}s'.format(time.time() - time_start))
        return train_net, eval_net
    def sample_by_nodes(self, sampled_num, rule = "random", keep_consistent_nodes = False):
        """
        sample some nodes to construct a sub-network.
        :param sampled_num:
        :param keep_consistent_nodes: whether making the sampled nodes consistent by re-sorting the nodesID.
        :param rule: sampling rule. random: randomly sample all sampled_nodes;
                                            extend: randomly sample one root-node and then extend to sampled_nodes.
        :return: a sub-network with sampled_nodes and corresponding edges.
        """
        logger.info('Net sampling: sample nodes to construct a sub-network ...')
        logger.info("\t\t sampled_nodes = {}, sample_rule = {}, keep_consistent_nodes = {}".format(sampled_num, rule, keep_consistent_nodes))
        logger.info("\t\t origin_node_size = {}".format(self.get_nodes_size()))
        assert sampled_num <= self.get_nodes_size(), "error, {} > {}".format(sampled_num, self.get_nodes_size())

        time_start = time.time()
        origin_nodes_list = list(self.nodes)

        if rule == "random":
            sampled_nodes_set = set(shuffle(origin_nodes_list, random_state=utils.get_random_seed())[0:sampled_num])
            # random.shuffle(origin_nodes_list)
            # sampled_nodes_set = set(origin_nodes_list[0:sampled_num])
        elif rule == "extend":
            sampled_nodes_set = set()
            extend_nodes_list = []
            origin_nodes_set = set(origin_nodes_list)
            while len(sampled_nodes_set) < sampled_num:
                if len(extend_nodes_list) == 0:
                    origin_nodes_set = origin_nodes_set - sampled_nodes_set
                    root = random.choice(shuffle(list(origin_nodes_set), random_state=utils.get_random_seed()))
                    sampled_nodes_set.add(root)
                    extend_nodes_list.append(root)
                    if len(sampled_nodes_set) >= sampled_num:
                        break
                root = extend_nodes_list.pop(0)
                for v in self._nodes_adjlist[root]:
                    if v not in sampled_nodes_set:
                        sampled_nodes_set.add(v)
                        extend_nodes_list.append(v)
                        if len(sampled_nodes_set) >= sampled_num:
                            break
        else:
            logger.error("Unknown sampling rule: '%s'.  Valid rules: 'random', 'extend'." % rule)

        sampled_net = Graph(isdirected=self._isdirected, isweighted=self._isweighted, self_looped=self._self_looped)

        for node in sampled_nodes_set:
            sampled_net.add_single_node(node)
            for v in self._nodes_adjlist[node]:
                if v in sampled_nodes_set:
                    sampled_net.add_single_edge(node, v)
        if keep_consistent_nodes:
            sampled_net.make_consistent()

        logger.info("\t\t sampled_net edges_size = {}".format(sampled_net.get_edges_size()))
        logger.info('Net sampling: sample nodes completed in {}s'.format(time.time() - time_start))
        return sampled_net
    def save_network(self, save_dir, save_prefixname, save_format):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logger.info('Net saved: saving network to {}'.format(os.path.join(save_dir, save_prefixname+"."+save_format)))
        time_start = time.time()
        fr = open(os.path.join(save_dir, save_prefixname+"."+save_format),'w')
        if save_format == "edgelist":
            for node, adj_list in sorted(self._nodes_adjlist.items(), key=lambda item: item[0]):
                if len(adj_list) == 0:
                    fr.write("{}\n".format(node))
                else:
                    for v in adj_list:
                        fr.write("{}\t{}\n".format(node, v))
        elif save_format == "adjlist":
            for node, adj_list in sorted(self._nodes_adjlist.items(), key=lambda item: item[0]):
                fr.write("{}".format(node))
                for v in adj_list:
                    fr.write("\t{}".format(v))
                fr.write("\n")
        else:
            logger.error("Unknown data format: '%s'.  Valid formats: 'adjlist', 'edgelist'" % save_format)
            sys.exit()
        fr.close()
        if self._id_nodes:
            with open(os.path.join(save_dir, "id_node.info"),'w') as fr:
                for id, node in enumerate(self._id_nodes):
                    fr.write("{}\t{}\n".format(id, node))
        self.print_net_info(file_path=os.path.join(save_dir, "net.info"), edges_file=os.path.join(save_dir, save_prefixname+"."+save_format))
        logger.info('Net saved: saving edges completed in {}s'.format(time.time() - time_start))
            
            
        
    

            
def construct_graph_from_adjlist(adjlist_file, info_path = None, isdirected = False, isweighted = False, self_looped = False):
    """ construct a network form adjlist files.
        the format of each line:
                v1 n1 n2 n3 ... nk
        which means [n1 n2 n3 ... nk] is v1's adjacency list.
    """
    net = Graph(isdirected = isdirected, isweighted = isweighted, self_looped = self_looped)
    logger.info('Net construct: loading edges to net...')
    time_start = time.time()
    # if not isinstance(adjlist_files, Iterable):
    #     adjlist_files = [adjlist_files]
    filepath = adjlist_file
    if not os.path.exists(filepath):
        logger.error('\t file \'%s\' not exists!' % filepath)
        sys.exit()
    else:
        logger.info('\t reading from file \'%s\'' % filepath)
        for line in open(filepath):
            if line:
                adjlist = [int(x) for x in line.strip().split('\t')]
                if len(adjlist) > 1:
                    for node_ID in adjlist[1:]:
                        net.add_single_edge(adjlist[0], node_ID)
                elif len(adjlist) == 1:
                    net.add_single_node(adjlist[0])
    logger.info('Net construct: load edges completed in {}s'.format(time.time() - time_start))
    # net.make_consistent()
    if info_path:
        net.print_net_info(edges_file=adjlist_file,file_path=info_path)
    return net


def construct_graph_from_edgelist(edgelist_file, info_path=None, isdirected = False, isweighted = False, self_looped = False):
    """ construct a network form edgelist files.
        the format of each line:
                v1 v2
        which means v1 points to v2.
    """
    net = Graph(isdirected = isdirected, isweighted = isweighted, self_looped = self_looped)
    logger.info('Net construct: loading edges to net...')
    time_start = time.time()
    # if not isinstance(edgelist_files, Iterable):
    #     edgelist_files = [edgelist_files]
    filepath = edgelist_file
    if not os.path.exists(filepath):
        logger.error('\t file \'%s\' not exists!' % filepath)
        sys.exit()
    else:
        logger.info('\t reading from file \'%s\'' % filepath)
        for line in open(filepath):
            if line:
                adjlist = [int(x) for x in line.strip().split('\t')]
                if len(adjlist) > 1:
                    net.add_single_edge(adjlist[0],adjlist[1])
                    if len(adjlist) > 2:
                        logger.warning('more than two tokens in edgelist line \'{}\''.format(line))
                elif len(adjlist) == 1:
                    net.add_single_node(adjlist[0])
    logger.info('Net construct: load edges completed in {}s'.format(time.time() - time_start))
    # net.make_consistent()
    if info_path:
        net.print_net_info(edges_file=edgelist_file, file_path=info_path)
    return net



def construct_network(options = None, data_path = None, data_format = None, net_info_path = None, isdirected = None, print_net_info = True):
    if data_path == None:
        data_path = options.data_path
    if data_format == None:
        data_format = options.data_format    
    if print_net_info:
        if net_info_path == None:
            net_info_path = options.net_info_path
    else:
        net_info_path = None
    if isdirected == None:
        isdirected = options.isdirected
        
    if data_format == "adjlist":
        net = construct_graph_from_adjlist(adjlist_file = data_path,
                                                   info_path = net_info_path,
                                                   isdirected = isdirected)
    elif data_format == "edgelist":
        net = construct_graph_from_edgelist(edgelist_file = data_path,
                                                    info_path = net_info_path,
                                                    isdirected = isdirected)
    else:
        logger.error("Unknown data format: '%s'.  Valid formats: 'adjlist', 'edgelist'" % data_format)
        sys.exit()
    return net