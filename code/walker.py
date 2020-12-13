#! /usr/bin/env python  
# -*- coding:utf-8 -*-  
#====#====#====#====  
# __author__ = "He Yu"   
# Version: 1.0.0  
#====#====#====#====

import logging
import sys
import os
import gc
import time
import random
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import pickle

import network
import utils


logger = logging.getLogger("NRL")

# global sharing variable
walker = None


class Walker(object):
    """random walker on the network."""
    def __init__(self, net, random_walker = 'uniform', walk_length = 100, walk_restart = 0,
                 distortion_power = 1., neg_sampled = 5,
                 p = 1., q = 1.):
        self._net = net
        self._walk_nodes_size = self._net.get_nodes_size()
        self._walk_nodes = range(self._walk_nodes_size)
        self._random_walker = random_walker
        self._walk_length = walk_length
        self._walk_restart = walk_restart
        self._p = p
        self._q = q
        self._nodes_degrees = None
        self._normed_degrees = None
        self._distortion_power = distortion_power
        self._neg_sampled = neg_sampled

        if self._random_walker == 'uniform': # Deepwalk
            self.random_walk = self._uniform_random_walk
        elif self._random_walker == 'restart':
            self.random_walk = self._restart_random_walk
        elif self._random_walker == 'bias': # Node2vec
            self.random_walk = self._bias_random_walk
        elif self._random_walker == 'line':
            self._edges_list = self._net.edges
            self.random_walk = self._edge_sampling  # equivalent to random-walk on nodes in actual
        else: self.random_walk = None

    @property
    def walk_nodes_size(self):
        return self._walk_nodes_size
    @property
    def walk_nodes(self):
        return self._net._nodes_adjlist.keys()
    @property
    def walk_nodes_normed_degrees(self):
        if self._normed_degrees == None:
            nodes_degrees = self.walk_nodes_degrees
            logger.info("norm the degrees vector by distortion_power {}".format(self._distortion_power))
            self._normed_degrees = np.power(nodes_degrees, self._distortion_power)
            self._normed_degrees = self._normed_degrees / np.sum(self._normed_degrees)
            if np.sum(self._normed_degrees) != 1.:
                self._normed_degrees = self._normed_degrees / np.sum(self._normed_degrees)
            if np.sum(self._normed_degrees) != 1.:
                self._normed_degrees = self._normed_degrees / np.sum(self._normed_degrees)
        return self._normed_degrees
    @property
    def walk_nodes_degrees(self):
        if self._nodes_degrees == None:
            self._nodes_degrees = np.array([self._net.get_degrees(v) for v in self._walk_nodes])
        return self._nodes_degrees
    @property
    def walk_edges_size(self):
        return len(self._edges_list)

    def _uniform_random_walk(self, start_node = None):
        """truncated uniform random walk used in deepwalk model."""
        if start_node == None:
            # Sampling is uniform w.r.t V, and not w.r.t E
            start_node = random.choice(self._walk_nodes)
        # if walk_length == None:
        #     walk_length = self._walk_length
        path = [start_node]
        while len(path) < self._walk_length:
            #if random.random() < self._walk_restart:
            #    path.append(start_node)
            #    continue
            cur = path[-1]
            adj_list = self._net._nodes_adjlist[cur]
            if len(adj_list) > 0:
                path.append(random.choice(adj_list)) # Generate a uniform random sample
            else:
                # logger.warning('no type-corresponding node found, walk discontinued, generate a path less than specified length.')
                # break
                # logger.warning('no type-corresponding node found, walk restarted.')
                path.append(start_node)

        return [str(node) for node in path]

    def _restart_random_walk(self, restart_node = None, walk_times = 0):
        """random walk with restart."""
        # if restart_node == None:
        #     # Sampling is uniform w.r.t V, and not w.r.t E
        #     restart_node = random.choice(self._walk_nodes)
        # target = restart_node
        context_list = []
        except_set = set()
        except_set.add(restart_node)
        for _ in range(walk_times):
            start_node = restart_node
            context = []
            while len(context) < self._walk_length:
                if random.random() < self._walk_restart:
                    start_node = restart_node
                adj_list = self._net._nodes_adjlist[start_node]
                if len(adj_list) > 0:
                    start_node = random.choice(adj_list)  # Generate a uniform random sample
                else:
                    start_node = restart_node
                    # logger.warning('no type-corresponding node found, walk restarted.')
                    # continue
                # if start_node != restart_node:
                context.append(start_node) # context
                except_set.add(start_node)
            context_list.extend(context)
        neg_nodes = utils.neg_sample(self._walk_nodes, except_set, num = self._neg_sampled,
                                     alias_table = self._alias_nodesdegrees)
        # np.asarray(context_list)
        return restart_node, context_list, neg_nodes # input(center_node), targets(context_nodes), neg_targets(neg_nodes)

    def _bias_random_walk(self, start_node = None):
        """bias random walk used in node2vec model."""
        if start_node == None:
            # Sampling is uniform w.r.t V, and not w.r.t E
            start_node = random.choice(self._walk_nodes)
        # if walk_length == None:
        #     walk_length = self._walk_length
        alias_edges = self._alias_edges
        path = [start_node]
        while len(path) < self._walk_length:
            cur = path[-1]
            adj_list = self._net._nodes_adjlist[cur]
            if len(adj_list) > 0:
                if len(path) == 1:
                    path.append(random.choice(adj_list))  # Generate a uniform random sample
                else:
                    prev = path[-2]
                    next = adj_list[utils.alias_draw(alias_edges[(prev, cur)][0],
                                               alias_edges[(prev, cur)][1])]
                    path.append(next)
            else:
                # logger.warning('no type-corresponding node found, walk discontinued, '
                #                'generate a path less than specified length.')
                # break
                path.append(start_node)
        return [str(node) for node in path]
    def preprocess_nodesdegrees(self, net_info_path = "./net.info"):
        if self._distortion_power == 0:
            self._alias_nodesdegrees=None
        else:
            time_start = time.time()
            net_dir = os.path.split(net_info_path)[0]
            alias_nodesdegrees_path = os.path.join(net_dir, "alias_nodesdegrees_power_{}.pickle".format(self._distortion_power))

            if os.path.exists(alias_nodesdegrees_path):
                logger.info("reading nodesdegrees from {}".format(alias_nodesdegrees_path))
                self._alias_nodesdegrees = pickle.load(open(alias_nodesdegrees_path, "rb"))
                logger.info('nodesdegrees readed in {}s'.format(time.time() - time_start))
                return

            if not os.path.exists(net_dir):
                os.makedirs(net_dir)

            logger.info("preprocessing nodesdegrees with distortion_power = {} ...".format(self._distortion_power))
            nodes_degrees = [self._net.get_degrees(v) for v in self._walk_nodes]
            normed_degrees = np.power(nodes_degrees, self._distortion_power)
            normed_degrees = normed_degrees / np.sum(normed_degrees)
            if np.sum(normed_degrees) != 1.:
                normed_degrees = normed_degrees / np.sum(normed_degrees)
            if np.sum(normed_degrees) != 1.:
                normed_degrees = normed_degrees / np.sum(normed_degrees)
            self._alias_nodesdegrees = utils.alias_setup(normed_degrees) # J, q
            pickle.dump(self._alias_nodesdegrees, open(alias_nodesdegrees_path, "wb"), -1)
            logger.info('nodesdegrees processed in {}s'.format(time.time() - time_start))
    def _preprocess_transition_probs(self, nodes):
        '''
        Preprocessing of second-order transition probabilities for guiding the bias random walks.
        process a probabilities list for each edge in a sparse network, for memory saving.
        '''

        # alias_nodes = {} # not used in unweighted network.
        # for node in range(self._walk_nodes_size):
        #     unnormalized_probs = [1 for nbr in self._net._nodes_adjlist[node]]
        #     norm_const = sum(unnormalized_probs)
        #     normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        #     norm_const = sum(normalized_probs)
        #     if norm_const != 1.:
        #         normalized_probs = [float(u_prob) / norm_const for u_prob in normalized_probs]
        #     alias_nodes[node] = utils.alias_setup(normalized_probs)

        # Get the alias edge setup lists for a given edge.
        alias_edges = {}
        # for src, adj_list in self._net._nodes_adjlist.items():
        for src in nodes:
            adj_list = self._net._nodes_adjlist[src]
            for dst in adj_list:
                unnormalized_probs = []
                for dst_nbr in self._net._nodes_adjlist[dst]:
                    if dst_nbr == src:
                        unnormalized_probs.append( 1.0 / self._p)
                    elif self._net.has_edge(src, dst_nbr):
                        unnormalized_probs.append(1.0)
                    else:
                        unnormalized_probs.append(1.0 / self._q)
                norm_const = sum(unnormalized_probs)
                normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
                norm_const = sum(normalized_probs)
                if norm_const != 1.:
                    normalized_probs = [float(u_prob) / norm_const for u_prob in normalized_probs]
                alias_edges[(src, dst)] = utils.alias_setup(normalized_probs)

        # self._alias_nodes = alias_nodes
        # self._alias_edges = alias_edges
        return alias_edges
    def preprocess_transition_probs(self, net_info_path = "./net.info", max_num_workers=1):
        time_start = time.time()

        net_dir = os.path.split(net_info_path)[0]
        alias_edges_path = os.path.join(net_dir, "alias_edges_p_{}_q_{}.pickle".format(self._p, self._q))

        if os.path.exists(alias_edges_path):
            logger.info("reading second-order transition probabilities from {}".format(alias_edges_path))
            self._alias_edges = pickle.load(open(alias_edges_path, "rb"))
            logger.info('second-order transition probabilities readed in {}s'.format(time.time() - time_start))
            return

        if not os.path.exists(net_dir):
            os.makedirs(net_dir)

        logger.info("preprocessing of second-order transition probabilities for guiding the bias random walks.")
        if max_num_workers <= 1:
            self._alias_edges = self._preprocess_transition_probs(self._walk_nodes)
        else:
            # speed up by using multi-process
            logger.info("\t allocating {} walk_nodes to {} workers ...".format(self._walk_nodes_size, max_num_workers))
            if self._walk_nodes_size <= max_num_workers:
                times_per_worker = [1 for _ in self._walk_nodes]
            else:
                div, mod = divmod(self._walk_nodes_size, max_num_workers)
                times_per_worker = [div for _ in range(max_num_workers)]
                for idx in range(mod):
                    times_per_worker[idx] = times_per_worker[idx] + 1
            assert sum(times_per_worker) == self._walk_nodes_size, 'workers allocating failed: %d != %d' % (
                sum(times_per_worker), self._walk_nodes_size)

            logger.info("using {} processes for preprocessing 2nd transition probabilities:".format(len(times_per_worker)))
            counter = 0
            nodes_list = []
            for c in times_per_worker:
                nodes_list.append(range(counter,counter+c))
                logger.info("\t process-{}: preprocessing nodes {}~{}".format(len(nodes_list)-1, counter, counter+c-1))
                counter = counter + c
            counter = 0
            for v in nodes_list:
                counter += len(v)
            assert counter == self._walk_nodes_size, 'workers allocating failed: %d != %d' % (counter, self._walk_nodes_size)
            alias_edges = []
            with ProcessPoolExecutor(max_workers=max_num_workers) as executor:
                for ret in executor.map(self._preprocess_transition_probs, nodes_list):
                    alias_edges.append(ret)
            self._alias_edges = {}
            for item in alias_edges:
                self._alias_edges.update(item)

        pickle.dump(self._alias_edges, open(alias_edges_path, "wb"), -1)
        logger.info('transition probabilities processed in {}s'.format(time.time() - time_start))

    def _edge_sampling(self, batch_size):
        edges = random.sample(self._edges_list, k=batch_size) # np.random.choice(self._edges_list,size=batch_size)
        data = []
        labels = []
        for source, target in edges:
            data.append(source)
            labels.append(target)
        return np.asarray(data), np.asarray(labels)


# walk to memory
def _construct_walk_corpus(walk_times):
    global walker
    logger.info('\t new walk process starts to walk %d times' % walk_times)
    walks = []
    nodes = list(walker.walk_nodes)
    for cnt in range(walk_times):
        random.shuffle(nodes)
        for node in nodes:
            path_instance = walker.random_walk(node)
            if path_instance is not None and len(path_instance) > 1:  # ???????????????????
                walks.append(path_instance)
    return walks

def _construct_walk_corpus_no_multiprocess(walk_times):
    logger.info('Corpus bulid: walking to memory (without using multi-process)...')
    time_start = time.time()
    walks = _construct_walk_corpus(walk_times)
    logger.info('Corpus bulid: walk completed in {}s'.format(time.time() - time_start))
    return walks

def _construct_walk_corpus_multiprocess(walk_times, max_num_workers=cpu_count()):
    """ Use multi-process scheduling"""
    # allocate walk times to workers
    if walk_times <= max_num_workers:
        times_per_worker = [1 for _ in range(walk_times)]
    else:
        div, mod = divmod(walk_times, max_num_workers)
        times_per_worker = [div for _ in range(max_num_workers)]
        for idx in range(mod):
            times_per_worker[idx] = times_per_worker[idx] + 1
    assert sum(times_per_worker) == walk_times, 'workers allocating failed: %d != %d' % (
        sum(times_per_worker), walk_times)

    sens = []
    args_list = []
    for index in range(len(times_per_worker)):
        args_list.append(times_per_worker[index])
    logger.info('Corpus bulid: walking to memory (using %d workers for multi-process)...' % len(times_per_worker))
    time_start = time.time()
    with ProcessPoolExecutor(max_workers=max_num_workers) as executor:
    # # the walker for node2vec is so large that we can not use multi-process, so we use multi-thread instead.
    # with ThreadPoolExecutor(max_workers=max_num_workers) as executor:
        for walks in executor.map(_construct_walk_corpus, args_list):
            sens.extend(walks)
    logger.info('Corpus bulid: walk completed in {}s'.format(time.time() - time_start))
    return sens

def build_walk_corpus_to_memory(walk_times, max_num_workers=cpu_count()):
    if max_num_workers <= 1 or walk_times <= 1:
        if max_num_workers > 1:
            logger.warning('Corpus bulid: walk times too small, using single-process instead...')
        return _construct_walk_corpus_no_multiprocess(walk_times)
    else:
        return _construct_walk_corpus_multiprocess(walk_times, max_num_workers=max_num_workers)


# store corpus
def store_walk_corpus(filebase, walk_sens, always_rebuild = False):
    if not utils.check_rebuild(filebase, descrip='walk corpus', always_rebuild=always_rebuild):
        return
    logger.info('Corpus store: storing...')
    time_start = time.time()
    with open(filebase, 'w') as fout:
        for sen in walk_sens:
            for v in sen:
                fout.write(u"{} ".format(str(v)))
            fout.write('\n')
    logger.info('Corpus store: store completed in {}s'.format(time.time() - time_start))
    return


# walk to files
def _construct_walk_corpus_iter(walk_times, walk_process_id):
    global walker
    nodes = list(walker.walk_nodes)
    last_time = time.time()
    for cnt in range(walk_times):
        start_time = time.time()
        logger.info('\t !process-%s walking %d/%d, interval %.4fs' % (walk_process_id, cnt, walk_times, start_time - last_time))
        last_time = start_time
        random.shuffle(nodes)
        for node in nodes:
            path_instance = walker.random_walk(node)
            if path_instance is not None and len(path_instance) > 1:  # ???????????????????
                yield path_instance

def _construct_walk_corpus_and_write_singprocess(args):
    filebase, walk_times = args
    walk_process_id = filebase.split('.')[-1]
    logger.info('\t new walk process-%s starts to walk %d times' % (walk_process_id, walk_times))
    time_start = time.time()
    with open(filebase, 'w') as fout:
        for walks in _construct_walk_corpus_iter(walk_times,walk_process_id):
            for v in walks:
                fout.write(u"{} ".format(str(v)))
            fout.write('\n')
    logger.info('\t process-%s walk ended, generated a new file \'%s\', it took %.4fs' % (
        walk_process_id, filebase, time.time() - time_start))
    return filebase

def _construct_walk_corpus_and_write_multiprocess(filebase,walk_times,headflag_of_index_file = '',
                                                  max_num_workers=cpu_count()):
    """ Walk to files.
        this method is designed for a very large scale network which is too large to walk to memory.
    """
    # allocate walk times to workers
    if walk_times <= max_num_workers:
        times_per_worker = [1 for _ in range(walk_times)]
    else:
        div, mod = divmod(walk_times, max_num_workers)
        times_per_worker = [div for _ in range(max_num_workers)]
        for idx in range(mod):
            times_per_worker[idx] = times_per_worker[idx] + 1
    assert sum(times_per_worker) == walk_times, 'workers allocating failed: %d != %d' % (
    sum(times_per_worker), walk_times)

    files_list = ["{}.{}".format(filebase, str(x)) for x in range(len(times_per_worker))]
    f = open(filebase, 'w')
    f.write('{}\n'.format(headflag_of_index_file))
    f.write('DESCRIPTION: allocate %d workers to concurrently walk %d times.\n' % (len(times_per_worker), walk_times))
    f.write('DESCRIPTION: generate %d files to save walk corpus:\n' % (len(times_per_worker)))
    for item in files_list:
        f.write('FILE: {}\n'.format(item))
    f.close()

    files = []
    args_list = []
    for index in range(len(times_per_worker)):
        args_list.append((files_list[index], times_per_worker[index]))

    logger.info('Corpus bulid: walking to files (using %d workers for multi-process)...' % len(times_per_worker))
    time_start = time.time()
    with ProcessPoolExecutor(max_workers=max_num_workers) as executor:
    # # the walker for node2vec is so large that we can not use multi-process, so we use multi-thread instead.
    # with ThreadPoolExecutor(max_workers=max_num_workers) as executor:
        for file_ in executor.map(_construct_walk_corpus_and_write_singprocess, args_list):
            files.append(file_)
    assert len(files) == len(files_list), 'ProcessPoolExecutor occured error, %d!=%d' % (len(files), len(files_list))

    logger.info('Corpus bulid: walk completed in {}s'.format(time.time() - time_start))
    return files

def build_walk_corpus_to_files(filebase, walk_times, headflag_of_index_file = '',
                               max_num_workers=cpu_count(), always_rebuild=False):
    if not utils.check_rebuild(filebase, descrip='walk corpus', always_rebuild=always_rebuild):
        return

    if max_num_workers <= 1 or walk_times <= 1:
        if max_num_workers > 1:
            logger.warning('Corpus bulid: walk times too small, using single-process instead...')
        files = []
        logger.info('Corpus bulid: walking to files (without using multi-process)...')
        time_start = time.time()
        files.append(_construct_walk_corpus_and_write_singprocess((filebase, walk_times)))
        logger.info('Corpus bulid: walk completed in {}s'.format(time.time() - time_start))
        return files
    else:
        return _construct_walk_corpus_and_write_multiprocess(filebase,walk_times,
                                                             headflag_of_index_file = headflag_of_index_file,
                                                             max_num_workers=max_num_workers)


# ========= Walks Corpus ===========#
class WalksCorpus(object):
    """
    Walks Corpus, load from files.
    Note: this class is designed to privode training corpus in form of a sentence iterator to reduce memeory.
    """
    def __init__(self, files_list):
        self.files_list = files_list
    def __iter__(self):
        for file in self.files_list:
            if (not os.path.exists(file)) or (not os.path.isfile(file)):
                continue
            with open(file, 'r') as f:
                for line in f:
                    yield line.strip().split()

def load_walks_corpus(files_list):
    logger.info('Corpus load: loading corpus to memory...')
    time_start = time.time()
    sens = []
    for file in files_list:
        if (not os.path.exists(file)) or (not os.path.isfile(file)):
            continue
        with open(file, 'r') as f:
            for line in f:
                sens.append(line.strip().split())
    logger.info('Corpus load: loading completed in {}s'.format(time.time() - time_start))
    return sens



# walk sentences
def build_walk_corpus(options, net = None):
    global walker

    # check walk info  and record
    if not utils.check_rebuild(options.corpus_store_path, descrip='walk corpus',
                              always_rebuild=options.always_rebuild):
        return
    if options.model == "DeepWalk":
        random_walker = "uniform"
    elif options.model == "Node2Vec":
        random_walker = "bias"
    else:
        logger.error("Unknown model or it cann't build walk corpus: '%s'." % options.model)
        sys.exit()
    if net == None:
        net  = network.construct_network(options)

    logger.info('Corpus bulid: walk info:')
    logger.info('\t random_walker = {}'.format(random_walker))
    logger.info('\t walk times = {}'.format(options.walk_times))
    logger.info('\t walk length = {}'.format(options.walk_length))
    if random_walker == "uniform":
        logger.info('\t walk restart = {}'.format(options.walk_restart))
    elif random_walker == "bias":
        logger.info('\t return_parameter (p) = {}'.format(options.p))
        logger.info('\t in-out_parameter (q) = {}'.format(options.q))
    logger.info('\t max walk workers = {}'.format(options.walk_workers))
    logger.info('\t walk to memory = {}'.format(str(options.walk_to_memory)))
    if options.walk_to_memory:
        logger.info('\t donot store corpus = {}'.format(str(options.not_store_corpus)))
        if not options.not_store_corpus:
            logger.info('\t corpus store path = {}'.format(options.corpus_store_path))
    else:
        logger.info('\t corpus store path = {}'.format(options.corpus_store_path))

    fr_walks = open(os.path.join(os.path.split(options.corpus_store_path)[0], 'walks.info'), 'w')
    fr_walks.write('Corpus walk info:\n')
    fr_walks.write('\t random_walker = {}\n'.format(random_walker))
    fr_walks.write('\t walk times = {}\n'.format(options.walk_times))
    fr_walks.write('\t walk length = {}\n'.format(options.walk_length))
    if random_walker == "uniform":
        fr_walks.write('\t walk restart = {}\n'.format(options.walk_restart))
    elif random_walker == "bias":
        fr_walks.write('\t return_parameter (p) = {}\n'.format(options.p))
        fr_walks.write('\t in-out_parameter (q) = {}\n'.format(options.q))
    fr_walks.write('\t max walk workers = {}\n'.format(options.walk_workers))
    fr_walks.write('\t walk to memory = {}\n'.format(str(options.walk_to_memory)))
    if options.walk_to_memory:
        fr_walks.write('\t donot store corpus = {}\n'.format(str(options.not_store_corpus)))
        if not options.not_store_corpus:
            fr_walks.write('\t corpus store path = {}\n'.format(options.corpus_store_path))
    else:
        fr_walks.write('\t corpus store path = {}\n'.format(options.corpus_store_path))
    fr_walks.close()

    walker = Walker(net, random_walker = random_walker, walk_length = options.walk_length,
                 p=options.p, q=options.q)
    if random_walker == "bias":
        # walker.preprocess_transition_probs(options.walk_workers)
        walker.preprocess_transition_probs(net_info_path = options.net_info_path)

    walk_corpus = None
    if options.walk_to_memory:
        walk_corpus = build_walk_corpus_to_memory(options.walk_times, max_num_workers=options.walk_workers)
        if not options.not_store_corpus:
            store_walk_corpus(options.corpus_store_path, walk_corpus, always_rebuild=options.always_rebuild)
    else:
        # walk to files
        walk_files = build_walk_corpus_to_files(options.corpus_store_path, options.walk_times,
                                                headflag_of_index_file=options.headflag_of_index_file,
                                                max_num_workers=options.walk_workers,
                                                always_rebuild=options.always_rebuild)
        if "train" in options.task:
            if options.load_from_memory:
                walk_corpus = load_walks_corpus(walk_files)
            else:
                walk_corpus = WalksCorpus(walk_files)
    del walker
    gc.collect()
    return walk_corpus













