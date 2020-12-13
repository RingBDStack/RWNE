#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""

import os
import numpy as np
import tensorflow as tf
import time
import logging
import utils
import network
# import scipy.sparse as sp
# from scipy.sparse import dok_matrix

import TF_utils

logger = logging.getLogger("NRL")



class DataSet(object):
    def __init__(self, net, using_label = False, feature_type = "random", shuffled=True,
                 label_path = None, label_size = None, feature_path = None, feature_size = None):

        self._nodes_size = net.get_nodes_size()
        self._using_label = using_label
        self._feature_type = feature_type
        self._shuffled = shuffled
        self._label_size = label_size
        self._feature_size = feature_size

        # generate adjacency matrix
        adj_matrix = np.zeros((self._nodes_size, self._nodes_size), dtype = np.int32)
        for x, y in net.edges:
            adj_matrix[x, y] = 1
            # adj_matrix[y, x] = 1
        # generate normalized laplacian matrix
        self._laplacian_matrix = self.preprocess_adj(adj_matrix)

        # generate labels (multi-hot encoding)
        if self._using_label:
            self._nodes_labels = np.zeros((self._nodes_size, self._label_size), dtype= np.int32)
            id_list, labels_list = utils.get_labeled_data(label_path)
            assert len(id_list) == self._nodes_size, "error: not all nodes is labeled, %d != %d"%(len(id_list),self._nodes_size)
            for idx in range(len(id_list)):
                self._nodes_labels[id_list[idx], labels_list[idx]] = 1
        else:
            self._adj_matrix = adj_matrix # adj_matrix will be the target label

        # generate features (additional attribute features is future work)
        if self._feature_type == "attribute":
            # for future work
            self._nodes_features = utils.get_features(feature_path)
            assert self._nodes_features.shape[0] == self._nodes_size, "error: %d != %d"%(self._nodes_features.shape[0], self._nodes_size)
            assert self._nodes_features.shape[1] == self._feature_size, "error: %d != %d"%(self._nodes_features.shape[1], self._feature_size)
        elif self._feature_type == "random":
            self._nodes_features = np.random.uniform(size = [self._nodes_size, self._feature_size])
        elif self._feature_type == "degree":
            assert self._feature_size == 1, "error: %d != 1" % self._feature_size
            self._nodes_features = np.zeros((self._nodes_size, self._feature_size), dtype= np.float32)
            for idx in range(self._nodes_size):
                self._nodes_features[idx][0] = net.get_degrees(idx)
        elif self._feature_type == "adjacency":
            assert self._feature_size == self._nodes_size, "error: %d != %d" % (self._feature_size, self._nodes_size)
            self._nodes_features = adj_matrix
        else:
            logger.error("error! invalid feature_type: {}".format(self._feature_type))

        self._nodes_order = np.arange(self._nodes_size)
        if self._shuffled:
            np.random.shuffle(self._nodes_order)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size, keep_strict_batching=True):
        if batch_size == self._nodes_size:
            return self.get_full()
        if keep_strict_batching:
            assert batch_size <= self._nodes_size, "error: %d > %d"%(batch_size, self._nodes_size)
        if self._index_in_epoch >= self._nodes_size:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffled:
                np.random.shuffle(self._nodes_order)
            # Start next epoch
            self._index_in_epoch = 0

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._nodes_size:
            if keep_strict_batching:
                # Finished epoch
                self._epochs_completed += 1
                # Shuffle the data
                if self._shuffled:
                    np.random.shuffle(self._nodes_order)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                self._index_in_epoch = self._nodes_size
        end = self._index_in_epoch
        index = self._nodes_order[start:end]
        return self.get_batch(index)

    def get_batch(self, index):
        batch_features = self._nodes_features[index]
        batch_laplacian = self._laplacian_matrix[index][:, index]
        if self._using_label:
            batch_labels = self._nodes_labels[index]
            return batch_features, batch_laplacian, batch_labels
        else:
            batch_adj = self._adj_matrix[index][:, index]
            return batch_features, batch_laplacian, batch_adj

    def get_full(self):
        if self._using_label:
            return self._nodes_features, self._laplacian_matrix, self._nodes_labels
        else:
            return self._nodes_features, self._laplacian_matrix, self._adj_matrix

    def preprocess_adj(self,adj):
        adj = np.add(adj, np.eye(adj.shape[0],dtype=np.float32), dtype=np.float32)
        rowsum = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return np.matmul(np.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)




# gcn model
class GCN(object):
    def __init__(self, dropout = 0.5, feature_size = 128, embedding_size = 128,
                 hidden_size_list = [], label_size = 0, using_label = True,
                 weight_decay=5e-4, act_func = tf.nn.relu):
        """
        :param dropout: Dropout rate (1 - keep probability)
        :param weight_decay: Weight for L2 loss on embedding matrix.
        """
        self._dropout = dropout
        self._embedding_size = embedding_size
        self._feature_size = feature_size
        self._label_size = label_size
        self._using_label = using_label
        self._weight_decay = weight_decay
        self._act_func = act_func
        self._hidden_size_list = hidden_size_list
        self._layers = []

        var_dim_list = [self._feature_size] + hidden_size_list + [self._embedding_size]
        for i in range(len(var_dim_list)-1):
            self._layers.append(GraphConvolution(input_dim=var_dim_list[i],
                                                 output_dim=var_dim_list[i+1],
                                                 dropout=self._dropout,
                                                 act=self._act_func,
                                                 layername="conv_layer_" + str(i),
                                                 weight_decay = weight_decay))
        if self._using_label:
            layername = "fully_layer"
            logger.info("constructing {}: {} -> {}".format(layername, self._embedding_size, self._label_size))
            with tf.variable_scope(layername):
                init_width = np.sqrt(6.0 / (self._embedding_size + self._label_size))
                self.full_weight = TF_utils.variable_with_weight_decay(name="weight",
                                                    initial_value=tf.random_uniform([self._embedding_size, self._label_size], minval=-init_width, maxval=init_width, dtype=tf.float32),
                                                    dtype=tf.float32,
                                                    wd=weight_decay)
                self.full_bias = TF_utils.variable_with_weight_decay(name="bias",
                                                    initial_value=tf.zeros([self._label_size], dtype=tf.float32),
                                                    dtype=tf.float32,
                                                    wd=weight_decay)
        else:
            logger.info("no fully_layer, unsupervised GCN!")

    @property
    def vectors(self):
        return self._embedding

    def inference(self, batch_input, batch_laplacian):
        """
        :param batch_input: features, shape=[nodes_size or batch_size, feature_size]
        :param batch_laplacian: laplacian matrix, shape=[nodes_size or batch_size, nodes_size or batch_size]
        :return:
        """
        X = batch_input
        for i in self._layers:
            X = i.apply(X, laplacian_mat=batch_laplacian)
        self._embedding = X # [nodes_size, embedding_size]

        if self._using_label:
            logits = tf.matmul(X, self.full_weight) + self.full_bias
        else:
            logits = tf.matmul(X, X, transpose_b=True)
        return logits

    def loss(self, logits, labels):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        tf.summary.scalar("sigmoid_loss", loss)
        tf.add_to_collection('losses', loss)

    def optimize(self, loss, global_step, lr):
        """Build the graph to optimize the loss function."""
        tf.summary.scalar('learning_rate', lr)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.minimize(loss,
                                      global_step=global_step,
                                      gate_gradients=optimizer.GATE_NONE)
        return train_op

    def train(self, batch_input, batch_laplacian, batch_labels, global_step, learning_rate):
        self.loss(self.inference(batch_input, batch_laplacian), batch_labels)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar("total_loss", loss)
        train_op = self.optimize(loss, global_step, learning_rate)
        return train_op, loss

class GraphConvolution(object):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, layername='conv_layer', weight_decay = 0):
        logger.info("constructing {}: {} -> {}".format(layername, input_dim, output_dim))
        self._dropout = dropout
        self._act = act
        with tf.variable_scope(layername):
            init_width = np.sqrt(6.0 / (input_dim + output_dim))
            self.weight = TF_utils.variable_with_weight_decay(name="weight",
                                                initial_value = tf.random_uniform([input_dim, output_dim], minval=-init_width, maxval=init_width, dtype=tf.float32),
                                                dtype=tf.float32,
                                                wd=weight_decay)
            self.bias = TF_utils.variable_with_weight_decay(name="bias",
                                                initial_value = tf.zeros([output_dim], dtype=tf.float32),
                                                dtype=tf.float32,
                                                wd=weight_decay)

    def apply(self, inputs, laplacian_mat):
        x = inputs
        if self._dropout >= 0.0001:
            x = tf.nn.dropout(x, 1 - self._dropout)
        x = tf.matmul(laplacian_mat, x)
        x = tf.matmul(x, self.weight)
        x = tf.add(x, self.bias)
        x = self._act(x)
        return x



def train(dataset, lr_file, ckpt_dir, checkpoint, options):
    nodes_size = dataset._nodes_size
    num_steps_per_epoch = int(nodes_size / options.batch_size)
    iter_epochs = options.iter_epoches
    iter_steps = round(iter_epochs * num_steps_per_epoch)  # iter_epoches should be big enough to converge.
    decay_epochs = options.decay_epochs
    decay_steps = round(decay_epochs * num_steps_per_epoch)
    ckpt_steps = round(options.ckpt_epochs * num_steps_per_epoch)
    initial_learning_rate = options.learning_rate
    decay_rate = options.decay_rate

    LR = utils.LearningRateGenerator(initial_learning_rate=initial_learning_rate, initial_steps=0,
                                     decay_rate=decay_rate, decay_steps=decay_steps, iter_steps=iter_steps)

    with tf.Graph().as_default(), tf.device('/gpu:0' if options.using_gpu else '/cpu:0'):

        global_step = tf.Variable(0, trainable=False, name="global_step")
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        inputs = tf.placeholder(tf.float32, shape=[None, options.feature_size], name='inputs')
        laplacian = tf.placeholder(tf.float32, [None, None], name="laplacian_matrix")
        if options.using_label:
            labels = tf.placeholder(tf.int32, shape=[None, options.label_size], name='labels')
        else:
            labels = tf.placeholder(tf.int32, shape=[None, None], name='adjacency')

        model = GCN(dropout=options.dropout, feature_size=options.feature_size, using_label=options.using_label,
                    embedding_size=options.embedding_size, hidden_size_list=options.hidden_size_list,
                    label_size=options.label_size, weight_decay=options.weight_decay)
        train_op, loss = model.train(inputs, laplacian, labels, global_step, learning_rate)

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

            last_loss_time = time.time() - options.loss_interval
            last_summary_time = time.time() - options.summary_interval
            last_decay_time = last_checkpoint_time = time.time()
            last_decay_step = last_summary_step = last_checkpoint_step = 0
            while True:
                start_time = time.time()
                batch_features, batch_adj, batch_labels = dataset.next_batch(options.batch_size)
                feed_dict = {inputs: batch_features, laplacian:batch_adj, labels: batch_labels, learning_rate: LR.learning_rate}
                _, loss_value, cur_step = sess.run([train_op, loss, global_step], feed_dict=feed_dict)
                now = time.time()

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                epoch, epoch_step = divmod(cur_step, num_steps_per_epoch)

                if now - last_loss_time >= options.loss_interval:
                    format_str = '%s: step=%d(%d/%d), lr=%.6f, loss=%.6f, duration/step=%.4fs'
                    logger.info(format_str % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                                              cur_step, epoch_step, epoch, LR.learning_rate, loss_value,
                                              now - start_time))
                    last_loss_time = time.time()
                if now - last_summary_time >= options.summary_interval or cur_step - last_summary_step >= options.summary_steps or cur_step >= iter_steps:
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, cur_step)
                    last_summary_time = time.time()
                    last_summary_step = cur_step
                ckpted = False
                # Save the model checkpoint periodically. (named 'model.ckpt-global_step.meta')
                if now - last_checkpoint_time >= options.ckpt_interval or cur_step - last_checkpoint_step >= ckpt_steps or cur_step >= iter_steps:
                    if options.batch_size == nodes_size:
                        batch_features, batch_adj, batch_labels = dataset.get_full()
                        feed_dict = {inputs: batch_features, laplacian: batch_adj, labels: batch_labels,
                                     learning_rate: LR.learning_rate}
                        vecs = sess.run(model.vectors, feed_dict=feed_dict)
                    else:
                        vecs = []
                        start = 0
                        while start < nodes_size:
                            end = min(nodes_size, start + options.batch_size)
                            index = np.arange(start, end)
                            start = end
                            batch_features, batch_adj, batch_labels = dataset.get_batch(index)
                            feed_dict = {inputs: batch_features, laplacian: batch_adj, labels: batch_labels,
                                         learning_rate: LR.learning_rate}
                            batch_embeddings = sess.run(model.vectors, feed_dict=feed_dict)
                            vecs.append(batch_embeddings)
                        vecs = np.concatenate(vecs, axis=0)
                    checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
                    utils.save_word2vec_format_and_ckpt(options.vectors_path, vecs, checkpoint_path, sess, saver, cur_step)
                    last_checkpoint_time = time.time()
                    last_checkpoint_step = cur_step
                    ckpted = True
                # update learning rate
                if ckpted or now - last_decay_time >= options.decay_interval or (
                        decay_steps > 0 and cur_step - last_decay_step >= decay_steps):
                    lr_info = np.loadtxt(lr_file, dtype=float)
                    if np.abs(lr_info[1] - decay_epochs) > 1e-6:
                        decay_epochs = lr_info[1]
                        decay_steps = round(decay_epochs * num_steps_per_epoch)
                    if np.abs(lr_info[2] - decay_rate) > 1e-6:
                        decay_rate = lr_info[2]
                    if np.abs(lr_info[3] - iter_epochs) > 1e-6:
                        iter_epochs = lr_info[3]
                        iter_steps = round(iter_epochs * num_steps_per_epoch)
                    if np.abs(lr_info[0] - initial_learning_rate) > 1e-6:
                        initial_learning_rate = lr_info[0]
                        LR.reset(initial_learning_rate=initial_learning_rate, initial_steps=cur_step,
                                 decay_rate=decay_rate, decay_steps=decay_steps, iter_steps=iter_steps)
                    else:
                        LR.exponential_decay(cur_step,
                                             decay_rate=decay_rate, decay_steps=decay_steps, iter_steps=iter_steps)
                    last_decay_time = time.time()
                    last_decay_step = cur_step
                if cur_step >= LR.iter_steps:
                    break
            summary_writer.close()


def train_vectors(options):
    # check vectors and ckpt
    checkpoint = '0'
    train_vec_dir = os.path.split(options.vectors_path)[0]
    ckpt_dir = os.path.join(train_vec_dir, 'ckpt')
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        cur_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        logger.info("model and vectors already exists, checkpoint step = {}".format(cur_step))
        checkpoint = input(
            "please input 0 to start a new train, or input a choosed ckpt to restore (-1 for latest ckpt)")
    if checkpoint == '0':
        if ckpt:
            tf.gfile.DeleteRecursively(ckpt_dir)
        logger.info('start a new embedding train using tensorflow ...')
    elif checkpoint == '-1':
        logger.info('restore a embedding train using tensorflow from latest ckpt...')
    else:
        logger.info('restore a embedding train using tensorflow from ckpt-%s...' % checkpoint)
    if not os.path.exists(train_vec_dir):
        os.makedirs(train_vec_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # construct network
    net = network.construct_network(options)

    lr_file = os.path.join(train_vec_dir, "lr.info")
    np.savetxt(lr_file,
               np.asarray([options.learning_rate, options.decay_epochs, options.decay_rate, options.iter_epoches],
                          dtype=np.float32),
               fmt="%.6f")

    if options.batch_size == 0:
        options.batch_size = net.get_nodes_size()
    if options.feature_type == "adjacency":
        options.feature_size = net.get_nodes_size()
    elif options.feature_type == "degree":
        options.feature_size = 1

    # train info
    logger.info('Train info:')
    logger.info('\t train_model = {}'.format(options.model))
    logger.info('\t total embedding nodes = {}'.format(net.get_nodes_size()))
    logger.info('\t total edges = {}'.format(np.size(np.array(net.edges, dtype=np.int32), axis=0)))
    logger.info('\t embedding size = {}'.format(options.embedding_size))
    logger.info('\t batch_size = {}'.format(options.batch_size))
    logger.info('\t feature_size = {}'.format(options.feature_size))
    logger.info('\t feature_type = {}'.format(options.feature_type))
    logger.info('\t feature_path = {}'.format(options.feature_path))
    logger.info('\t using_label = {}'.format(options.using_label))
    logger.info('\t label_path = {}'.format(options.label_path))
    logger.info('\t label_size = {}'.format(options.label_size))
    logger.info('\t hidden_size_list = {}'.format(options.hidden_size_list))
    logger.info('\t weight_decay = {}'.format(options.weight_decay))
    logger.info('\t dropout = {}'.format(options.dropout))
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
    fr_vec.write('\t total embedding nodes = {}\n'.format(net.get_nodes_size()))
    fr_vec.write('\t total edges = {}\n'.format(np.size(np.array(net.edges, dtype=np.int32), axis=0)))
    fr_vec.write('\t embedding size = {}\n'.format(options.embedding_size))
    fr_vec.write('\t batch_size = {}\n'.format(options.batch_size))
    fr_vec.write('\t feature_size = {}\n'.format(options.feature_size))
    fr_vec.write('\t feature_type = {}\n'.format(options.feature_type))
    fr_vec.write('\t feature_path = {}\n'.format(options.feature_path))
    fr_vec.write('\t using_label = {}\n'.format(options.using_label))
    fr_vec.write('\t label_path = {}\n'.format(options.label_path))
    fr_vec.write('\t label_size = {}\n'.format(options.label_size))
    fr_vec.write('\t hidden_size_list = {}\n'.format(options.hidden_size_list))
    fr_vec.write('\t weight_decay = {}\n'.format(options.weight_decay))
    fr_vec.write('\t dropout = {}\n'.format(options.dropout))
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

    dataset = DataSet(net=net, using_label=options.using_label, feature_type=options.feature_type,
                      label_path=options.label_path, label_size=options.label_size,
                      feature_path = options.feature_path, feature_size=options.feature_size)

    # train
    logger.info('training...')
    time_start = time.time()
    train(dataset=dataset, lr_file=lr_file, ckpt_dir=ckpt_dir, checkpoint=checkpoint, options=options)
    logger.info('train completed in {}s'.format(time.time() - time_start))
    return


