#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.5.3
 
"""

import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import numpy as np
import tensorflow as tf
import utils
import network
from walker import Walker


logger = logging.getLogger("NRL")


##
class RWRGenerator(object):
    def __init__(self, walker, batch_size, walk_times, max_walk_workers = 1):
        self._walker = walker
        self._nodes_size = self._walker.walk_nodes_size
        self._max_walk_workers = max_walk_workers
        self._batch_size = batch_size
        self._walk_times = walk_times
        assert self._batch_size<=self._nodes_size,"error, batch_size({}) > nodes_size({}), ".format(self._batch_size,self._nodes_size)

        # # sample strategy
        # if strategy == "shuffle":
        #     self.get_restart_nodes = self._get_restart_nodes_without_replacement
        # elif strategy == "uniform":
        #     self.get_restart_nodes = self._get_restart_nodes_uniform
        # elif strategy == "unigram":
        #     self.get_restart_nodes = self._get_restart_nodes_weight
        # else:
        #     logger.error("'{}': undefined sampling strategy for restart-random-walk!".format(strategy))
        self.get_restart_nodes = self._get_restart_nodes_without_replacement

        self.walker_nodes = np.arange(self._nodes_size)
        # self.cur_start_nodes = np.arange(self._nodes_size)
        # self.cur_hop = np.zeros([self._nodes_size],dtype=np.int32)
        np.random.shuffle(self.walker_nodes)
        self.head = 0

        # walk workers
        if self._batch_size <= self._max_walk_workers:
            self._times_per_worker = [1 for _ in range(self._batch_size)]
        else:
            div, mod = divmod(self._batch_size, self._max_walk_workers)
            self._times_per_worker = [div for _ in range(self._max_walk_workers)]
            for idx in range(mod):
                self._times_per_worker[idx] = self._times_per_worker[idx] + 1
        assert sum(self._times_per_worker) == self._batch_size, 'workers allocating failed: %d != %d' % (
            sum(self._times_per_worker), self._batch_size)

    def _get_restart_nodes_without_replacement(self,num):
        end = self.head + num
        if end > self._nodes_size:
            np.random.shuffle(self.walker_nodes)
            self.head = 0
            end = num
        nodes = self.walker_nodes[self.head:end]
        self.head = end
        return nodes
    # def _get_restart_nodes_uniform(self,num):
    #     return np.random.choice(self.walker_nodes,num,replace=False)
    # def _get_restart_nodes_weight(self,num):
    #     return np.random.choice(self.walker_nodes,num,replace=False,p=self._walker.walk_nodes_normed_degrees)

    def _next_batch(self,restart_nodes):
        inputs = []
        labels = []
        neg_labels = []
        for restart_node in restart_nodes:
            center_node, context_nodes, neg_nodes = self._walker.random_walk(restart_node = restart_node, walk_times = self._walk_times)
            inputs.append(center_node)
            labels.append(context_nodes)
            neg_labels.append(neg_nodes)
        # return np.asarray(inputs), np.asarray(labels), np.asarray(neg_labels)
        return inputs, labels, neg_labels # inputs(center_nodes), labels(context_nodes), neg_labels(neg_nodes)
    def next_batch(self):
        restart_nodes = self.get_restart_nodes(self._batch_size)
        if len(self._times_per_worker) == 1:
            inputs, labels, neg_labels = self._next_batch(restart_nodes)
            return inputs, labels, neg_labels
        else:
            args_list = []
            counter = 0
            for index in range(len(self._times_per_worker)):
                end = counter + self._times_per_worker[index]
                args_list.append(restart_nodes[counter:end])
                counter = end
            rets = []
            with ThreadPoolExecutor(max_workers=self._max_walk_workers) as executor:
                for ret in executor.map(self._next_batch, args_list):
                    rets.append(ret)
            inputs, labels, neg_labels = rets[0]
            for item in rets[1:]:
                inputs.extend(item[0])
                labels.extend(item[1])
                neg_labels.extend(item[2])
            return inputs, labels, neg_labels


## model
class SGNS(object):
    """A customized version of Word2Vec(SkipGram-NegativeSampling) model."""
    def __init__(self, vocab_size, embedding_size = 128, batch_size =1, batch_single_size = 8, local_weight = 0, global_weight = 1.):
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._batch_size = batch_size
        self._batch_single_size = batch_single_size
        self._local_weight = local_weight
        self._global_weight = global_weight

        init_width = 0.5 / self._embedding_size
        self._embedding = tf.Variable(
            tf.random_uniform([self._vocab_size, self._embedding_size], -init_width, init_width),
            name='embedding')
        if self._global_weight > 0:
            self._nce_weight = tf.Variable(
                tf.truncated_normal([self._vocab_size, self._embedding_size],
                                    stddev=1.0 / np.sqrt(self._embedding_size)),
                name='nce_weight')
            self._nce_biases = tf.Variable(tf.zeros([self._vocab_size]),
                                           name='nce_biases')
    @property
    def vectors(self):
        return self._embedding
    @property
    def context_weights(self):
        return self._nce_weight
    @property
    def context_biases(self):
        return self._nce_biases

    def inference(self, inputs, labels, neg_labels):
        """
        P(context(v)|v)
        # inputs(center_nodes), labels(context_nodes), neg_labels(neg_nodes)
        :param inputs: center_nodes, shape=[options.batch_size]
        :param labels:  context_nodes, shape=[options.batch_size, batch_single_size]
        :param neg_labels:  neg_nodes, shape=[options.batch_size, options.negative]
        :return:
        """
        # embedding for true labels: [batch_size, batch_single_size, emb_dim]
        embed_true = tf.nn.embedding_lookup(self._embedding, labels)
        # embedding for negative labels: [batch_size, num_sampled, emb_dim]
        embed_neg = tf.nn.embedding_lookup(self._embedding, neg_labels)

        if self._global_weight > 0:
            # Weights: [batch_size, emb_dim]
            nce_w = tf.nn.embedding_lookup(self._nce_weight, inputs)
            nce_w = tf.expand_dims(nce_w, -1)  # [batch_size, emb_dim, 1]
            # Biases: [batch_size]
            nce_b = tf.nn.embedding_lookup(self._nce_biases, inputs)
            nce_b = tf.expand_dims(nce_b, -1)  # [batch_size,1]
            # True logits: [batch_size, batch_single_size]
            true_logits = tf.squeeze(tf.matmul(embed_true, nce_w), -1) + nce_b
            # negative logits: [batch_size, num_sampled]
            neg_logits = tf.squeeze(tf.matmul(embed_neg, nce_w), -1) + nce_b
            # Build the graph for the global_loss by using negative sampling (NCE loss)
            self.global_loss(true_logits, neg_logits)

        if self._local_weight > 0:
            # embedding normalized: [batch_size, batch_single_size, emb_dim]
            embed_true = tf.nn.l2_normalize(embed_true, axis=2)
            # embedding normalized: [batch_size, batch_single_size, emb_dim]
            embed_neg = tf.nn.l2_normalize(embed_neg, axis=2)
            # embedding: [batch_size, emb_dim]
            embed = tf.nn.embedding_lookup(self._embedding, inputs)
            embed = tf.nn.l2_normalize(embed, axis=1)
            embed = tf.expand_dims(embed, -1)  # [batch_size, emb_dim, 1]
            # True logits: [batch_size, batch_single_size]
            true_logits = tf.squeeze(tf.matmul(embed_true, embed), -1)
            # negative logits: [batch_size, num_sampled]
            neg_logits = tf.squeeze(tf.matmul(embed_neg, embed), -1)
            # Build the graph for the local_loss by using cosine similarity.
            self.local_loss(true_logits, neg_logits)

    def global_loss(self, true_logits, neg_logits):
        """Build the graph for the global_loss by using negative sampling (NCE loss)."""
        # cross-entropy(logits, labels)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(neg_logits), logits=neg_logits)

        # NCE-loss is the sum of the true and noise (sampled words) contributions, averaged over the batch.
        nce_loss_tensor = tf.reduce_mean(true_xent) + tf.reduce_sum(sampled_xent) / self._batch_size
        tf.summary.scalar("global_loss", nce_loss_tensor)
        tf.add_to_collection('losses', nce_loss_tensor * self._global_weight)

    def local_loss(self, true_logits, neg_logits):
        """Build the graph for the local_loss by using cosine similarity."""
        # -log loss: -log(x)
        # cross-entropy(logits, labels)
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        # cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
        true_logits = 0.5 + 0.5 * true_logits
        neg_logits = 0.5 + 0.5 * neg_logits
        true_xent = -tf.log(tf.clip_by_value(true_logits, 1e-10, 1.0))
        sampled_xent = -tf.log(tf.clip_by_value(1 - neg_logits, 1e-10, 1.0))
        # one positive + one negative
        log_loss_tensor = tf.reduce_mean(true_xent) + tf.reduce_mean(sampled_xent)
        tf.summary.scalar("local_loss", log_loss_tensor)
        tf.add_to_collection('losses', log_loss_tensor * self._local_weight)

    def optimize(self, loss, global_step, lr):
        """Build the graph to optimize the loss function."""
        tf.summary.scalar('learning_rate', lr)

        # Compute gradients
        # opt = tf.train.MomentumOptimizer(lr, option.MOMENTUM)
        # opt = tf.train.AdamOptimizer(lr)
        # opt = tf.train.RMSPropOptimizer(lr) # ???????????
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.minimize(loss,
                                      global_step=global_step,
                                      gate_gradients=optimizer.GATE_NONE)
        return train_op

    def train(self, batch_inputs, batch_labels, batch_neg_labels, global_step, learning_rate):
        self.inference(batch_inputs, batch_labels, batch_neg_labels)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar("total_loss", loss)
        train_op = self.optimize(loss, global_step, learning_rate)
        return train_op, loss



def _train_thread_body(rwrgenerator, inputs, labels, neg_labels, session, train_op, global_step, learning_rate, LR):
    while True:
        batch_inputs, batch_labels, batch_neg_labels = rwrgenerator.next_batch()
        feed_dict = {inputs:batch_inputs, labels: batch_labels, neg_labels:batch_neg_labels, learning_rate: LR.learning_rate}
        _, cur_step = session.run([train_op, global_step], feed_dict=feed_dict)
        if cur_step >= LR.iter_steps:
            break

## train
def train(walker, lr_file, ckpt_dir, checkpoint, options):
    vocab_size = walker.walk_nodes_size
    num_steps_per_epoch = int(vocab_size * options.train_workers / options.batch_size) # a rough formula of epoch in RWR.???????????
    iter_epochs = options.iter_epoches
    iter_steps = round(iter_epochs * num_steps_per_epoch) # iter_epoches should be big enough to converge.
    decay_epochs = options.decay_epochs
    decay_steps = round(decay_epochs * num_steps_per_epoch)
    ckpt_steps = round(options.ckpt_epochs * num_steps_per_epoch)
    initial_learning_rate = options.learning_rate
    decay_rate = options.decay_rate
    
    LR = utils.LearningRateGenerator(initial_learning_rate = initial_learning_rate, initial_steps = 0,
                                     decay_rate = decay_rate, decay_steps = decay_steps, iter_steps = iter_steps)

    with tf.Graph().as_default(), tf.device('/gpu:0' if options.using_gpu else '/cpu:0'):

        global_step = tf.Variable(0, trainable=False, name="global_step")
        batch_single_size = options.walk_times*options.walk_length # !
        # inputs(center_nodes), labels(context_nodes), neg_labels(neg_nodes)
        inputs = tf.placeholder(tf.int32, shape=[options.batch_size], name='inputs')  # center_nodes
        labels = tf.placeholder(tf.int32, shape=[options.batch_size, batch_single_size], name='labels') # context_nodes
        neg_labels = tf.placeholder(tf.int32, shape=[options.batch_size, options.negative], name='neg_labels') # neg_nodes
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        model = SGNS(vocab_size=vocab_size, embedding_size=options.embedding_size,
                     batch_size=options.batch_size, batch_single_size = batch_single_size,
                     local_weight=options.local_weight, global_weight = options.global_weight)

        train_op, loss = model.train(inputs, labels, neg_labels, global_step, learning_rate)

        # Create a saver.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=6)

        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init_op = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU implementations.
        config = tf.ConfigProto(
            allow_soft_placement=options.allow_soft_placement,
            log_device_placement=options.log_device_placement)
        config.gpu_options.per_process_gpu_memory_fraction = options.gpu_memory_fraction
        config.gpu_options.allow_growth = options.allow_growth
        # config.gpu_options.visible_device_list = visible_device_list

        with tf.Session(config=config) as sess:
            # first_step = 0
            if checkpoint == '0': # new train
                sess.run(init_op)

            elif checkpoint == '-1':  # choose the latest one
                ckpt = tf.train.get_checkpoint_state(ckpt_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    # new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # global_step_for_restore = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # first_step = int(global_step_for_restore) + 1
                else:
                    logger.warning('No checkpoint file found')
                    return
            else:
                if os.path.exists(os.path.join(ckpt_dir, 'model.ckpt-' + checkpoint + '.index')):
                    # new_saver = tf.train.import_meta_graph(
                    #     os.path.join(ckpt_dir, 'model.ckpt-' + checkpoint + '.meta'))
                    saver.restore(sess,
                                  os.path.join(ckpt_dir, 'model.ckpt-' + checkpoint))
                    # first_step = int(checkpoint) + 1
                else:
                    logger.warning('checkpoint {} not found'.format(checkpoint))
                    return

            summary_writer = tf.summary.FileWriter(ckpt_dir, sess.graph)

            ## train
            executor_workers = options.train_workers-1
            if executor_workers > 0:
                futures = set()
                executor = ThreadPoolExecutor(max_workers=executor_workers)
                for _ in range(executor_workers):
                    future = executor.submit(_train_thread_body,
                                             RWRGenerator(walker, options.batch_size, options.walk_times, options.walk_workers),
                                    inputs, labels, neg_labels, sess, train_op, global_step, learning_rate, LR)
                    logger.info("open a new training thread: %s"%future)
                    futures.add(future)
            last_loss_time = time.time() - options.loss_interval
            last_summary_time = time.time() - options.summary_interval
            last_decay_time = last_checkpoint_time = time.time()
            last_decay_step = last_summary_step = last_checkpoint_step = 0
            rwrgenerator = RWRGenerator(walker, options.batch_size, options.walk_times, options.walk_workers)
            while True:
                start_time = time.time()
                batch_inputs, batch_labels, batch_neg_labels = rwrgenerator.next_batch()
                feed_dict = {inputs: batch_inputs, labels: batch_labels, neg_labels: batch_neg_labels,
                             learning_rate: LR.learning_rate}
                _, loss_value, cur_step = sess.run([train_op, loss, global_step], feed_dict=feed_dict)
                now = time.time()

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                epoch, epoch_step = divmod(cur_step,num_steps_per_epoch)

                if now - last_loss_time >= options.loss_interval:
                    format_str = '%s: step=%d(%d/%d), lr=%.6f, loss=%.6f, duration/step=%.4fs'
                    logger.info(format_str % ( time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                                               cur_step, epoch_step, epoch, LR.learning_rate, loss_value, now - start_time))
                    last_loss_time = time.time()
                if now - last_summary_time >= options.summary_interval or cur_step - last_summary_step >= options.summary_steps or cur_step >= iter_steps:
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, cur_step)
                    last_summary_time = time.time()
                    last_summary_step = cur_step
                ckpted = False
                # Save the model checkpoint periodically. (named 'model.ckpt-global_step.meta')
                if now - last_checkpoint_time >= options.ckpt_interval or cur_step - last_checkpoint_step >= ckpt_steps or cur_step >= iter_steps:
                    vecs, global_step_value = sess.run([model.vectors, global_step], feed_dict=feed_dict)
                    # vecs,weights,biases = sess.run([model.vectors,model.context_weights,model.context_biases],
                    #                              feed_dict=feed_dict) 
                    checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
                    utils.save_word2vec_format_and_ckpt(options.vectors_path, vecs, checkpoint_path, sess, saver, global_step_value)
                    # save_word2vec_format(vectors_path+".contexts", weights, walker.idx_nodes)
                    # save_word2vec_format(vectors_path+".context_biases", np.reshape(biases,[-1,1]), walker.idx_nodes)
                    last_checkpoint_time = time.time()
                    last_checkpoint_step = global_step_value
                    ckpted = True
                # update learning rate
                if ckpted or now - last_decay_time >= options.decay_interval or (decay_steps > 0 and cur_step - last_decay_step >= decay_steps):
                    lr_info = np.loadtxt(lr_file, dtype=float)
                    if np.abs(lr_info[1]-decay_epochs) > 1e-6:
                        decay_epochs = lr_info[1]
                        decay_steps = round(decay_epochs * num_steps_per_epoch)
                    if np.abs(lr_info[2]-decay_rate) > 1e-6:
                        decay_rate = lr_info[2]
                    if np.abs(lr_info[3]-iter_epochs) > 1e-6:
                        iter_epochs = lr_info[3]
                        iter_steps = round(iter_epochs * num_steps_per_epoch)
                    if np.abs(lr_info[0] - initial_learning_rate) > 1e-6:
                        initial_learning_rate = lr_info[0]
                        LR.reset(initial_learning_rate=initial_learning_rate, initial_steps=cur_step,
                                 decay_rate=decay_rate, decay_steps=decay_steps, iter_steps = iter_steps)
                    else:
                        LR.exponential_decay(cur_step,
                                             decay_rate=decay_rate, decay_steps=decay_steps, iter_steps=iter_steps)
                    last_decay_time = time.time()
                    last_decay_step = cur_step
                if cur_step >= LR.iter_steps:
                    break

            summary_writer.close()
            if executor_workers > 0:
                logger.info("waiting the training threads finished:")
                try:
                    for future in as_completed(futures):
                        logger.info(future)
                except KeyboardInterrupt:
                    print("stopped by hand.")


def train_vectors(options):

    # options.global_weight = 1. - options.local_weight
    # options.global_weight = 1
    # options.local_weight = 1

    # check vectors and ckpt
    checkpoint = '0'
    train_vec_dir = os.path.split(options.vectors_path)[0]
    ckpt_dir = os.path.join(train_vec_dir, 'ckpt')
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        cur_step= ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        logger.info("model and vectors already exists, checkpoint step = {}".format(cur_step))
        checkpoint = input("please input 0 to start a new train, or input a choosed ckpt to restore (-1 for latest ckpt)")
    if checkpoint == '0':
        if ckpt:
            tf.gfile.DeleteRecursively(ckpt_dir)
        logger.info('start a new embedding train using tensorflow ...')
    elif checkpoint == '-1':
        logger.info('restore a embedding train using tensorflow from latest ckpt ...')
    else:
        logger.info('restore a embedding train using tensorflow from ckpt-%s ...'%checkpoint)
    if not os.path.exists(train_vec_dir):
        os.makedirs(train_vec_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # construct network
    net = network.construct_network(options)

    lr_file = os.path.join(train_vec_dir, "lr.info")
    np.savetxt(lr_file,
               np.asarray([options.learning_rate, options.decay_epochs,options.decay_rate,options.iter_epoches],
                          dtype=np.float32),
               fmt="%.6f")

    random_walker = "restart"

    # train info
    logger.info('Train info:')
    logger.info('\t train_model = {}'.format(options.model))
    logger.info('\t random_walker = {}'.format(random_walker))
    logger.info('\t walk_workers = {}'.format(options.walk_workers))
    logger.info('\t train_workers = {}\n'.format(options.train_workers))
    logger.info('\t walk_restart = {}'.format(options.walk_restart))
    logger.info('\t walk_times = {}'.format(options.walk_times))
    logger.info('\t walk_length = {}'.format(options.walk_length))
    logger.info('\t batch_size = {}'.format(options.batch_size))
    logger.info('\t local_weight = {}'.format(options.local_weight))
    logger.info('\t global_weight = {}\n'.format(options.global_weight))
    logger.info('\t total embedding nodes = {}'.format(net.get_nodes_size()))
    logger.info('\t total edges = {}'.format(np.size(np.array(net.edges, dtype=np.int32), axis=0)))
    logger.info('\t embedding_size = {}'.format(options.embedding_size))
    logger.info('\t negative = {}'.format(options.negative))
    logger.info('\t distortion_power = {}'.format(options.distortion_power))
    logger.info('\t iter_epoches = {}'.format(options.iter_epoches))
    logger.info('\t init_learning_rate = {}'.format(options.learning_rate))
    logger.info('\t decay_epochs = {}'.format(options.decay_epochs))
    logger.info('\t decay_interval = {}'.format(options.decay_interval))
    logger.info('\t decay_rate = {}'.format(options.decay_rate))
    logger.info('\t loss_interval = {}s'.format(options.loss_interval))
    logger.info('\t summary_steps = {}'.format(options.summary_steps))
    logger.info('\t summary_interval = {}s'.format(options.summary_interval))
    logger.info('\t ckpt_epochs = {}'.format(options.ckpt_epochs))
    logger.info('\t ckpt_interval = {}s\n'.format(options.ckpt_interval))
    logger.info('\t using_gpu = {}'.format(options.using_gpu))
    logger.info('\t visible_device_list = {}'.format(options.visible_device_list))
    logger.info('\t log_device_placement = {}'.format(options.log_device_placement))
    logger.info('\t allow_soft_placement = {}'.format(options.allow_soft_placement))
    logger.info('\t gpu_memory_fraction = {}'.format(options.gpu_memory_fraction))
    logger.info('\t gpu_memory_allow_growth = {}'.format(options.allow_growth))

    logger.info('\t ckpt_dir = {}'.format(ckpt_dir))
    logger.info('\t vectors_path = {}'.format(options.vectors_path))
    logger.info('\t learning_rate_path = {}'.format(lr_file))

    fr_vec = open(os.path.join(train_vec_dir, 'embedding.info'), 'w')
    fr_vec.write('embedding info:\n')
    fr_vec.write('\t train_model = {}\n'.format(options.model))
    fr_vec.write('\t random_walker = {}\n'.format(random_walker))
    fr_vec.write('\t walk_workers = {}\n'.format(options.walk_workers))
    fr_vec.write('\t train_workers = {}\n\n'.format(options.train_workers))
    fr_vec.write('\t walk_restart = {}\n'.format(options.walk_restart))
    fr_vec.write('\t walk_times = {}\n'.format(options.walk_times))
    fr_vec.write('\t walk_length = {}\n'.format(options.walk_length))
    fr_vec.write('\t batch_size = {}\n'.format(options.batch_size))
    fr_vec.write('\t local_weight = {}\n'.format(options.local_weight))
    fr_vec.write('\t global_weight = {}\n'.format(options.global_weight))
    fr_vec.write('\t total embedding nodes = {}\n'.format(net.get_nodes_size()))
    fr_vec.write('\t total edges = {}\n'.format(np.size(np.array(net.edges, dtype=np.int32), axis=0)))
    fr_vec.write('\t embedding size = {}\n'.format(options.embedding_size))
    fr_vec.write('\t negative = {}\n'.format(options.negative))
    fr_vec.write('\t distortion_power = {}\n\n'.format(options.distortion_power))
    fr_vec.write('\t iter_epoches = {}\n'.format(options.iter_epoches))
    fr_vec.write('\t init_learning_rate = {}\n'.format(options.learning_rate))
    fr_vec.write('\t decay_epochs = {}\n'.format(options.decay_epochs))
    fr_vec.write('\t decay_interval = {}\n'.format(options.decay_interval))
    fr_vec.write('\t decay_rate = {}\n'.format(options.decay_rate))
    fr_vec.write('\t loss_interval = {}s\n'.format(options.loss_interval))
    fr_vec.write('\t summary_steps = {}\n'.format(options.summary_steps))
    fr_vec.write('\t summary_interval = {}s\n'.format(options.summary_interval))
    fr_vec.write('\t ckpt_epochs = {}\n'.format(options.ckpt_epochs))
    fr_vec.write('\t ckpt_interval = {}s\n\n'.format(options.ckpt_interval))
    fr_vec.write('\t using_gpu = {}\n'.format(options.using_gpu))
    fr_vec.write('\t visible_device_list = {}\n'.format(options.visible_device_list))
    fr_vec.write('\t log_device_placement = {}\n'.format(options.log_device_placement))
    fr_vec.write('\t allow_soft_placement = {}\n'.format(options.allow_soft_placement))
    fr_vec.write('\t gpu_memory_fraction = {}\n'.format(options.gpu_memory_fraction))
    fr_vec.write('\t gpu_memory_allow_growth = {}\n'.format(options.allow_growth))

    fr_vec.write('\t ckpt_dir = {}\n'.format(ckpt_dir))
    fr_vec.write('\t vectors_path = {}\n'.format(options.vectors_path))
    fr_vec.write('\t learning_rate_path = {}\n'.format(lr_file))

    fr_vec.close()

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if options.using_gpu:
        visible_devices = str(options.visible_device_list[0])
        for dev in options.visible_device_list[1:]:
            visible_devices = visible_devices + ',%s' % dev
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # set log_level for gpu:
    console_log_level = options.log.upper()
    if console_log_level == "CRITICAL":
        gpu_log = '3'
    elif console_log_level == "ERROR":
        gpu_log = '2'
    elif console_log_level == "WARNING":
        gpu_log = '1'
    else:
        gpu_log = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = gpu_log

    walker = Walker(net, random_walker=random_walker, walk_length=options.walk_length,
                    walk_restart=options.walk_restart, distortion_power = options.distortion_power,
                    neg_sampled=options.negative)
    walker.preprocess_nodesdegrees(net_info_path = options.net_info_path)

    # train
    logger.info('training...')
    time_start = time.time()
    train(walker = walker, lr_file = lr_file, ckpt_dir = ckpt_dir, checkpoint = checkpoint,
          options = options)
    logger.info('train completed in {}s'.format(time.time() - time_start))
    return









