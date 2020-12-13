#! /usr/bin/env python  
# -*- coding:utf-8 -*-  
# dngr = Deep Neural network for learning Gragh Representations.
"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""
import os
import sys
import gc
import numpy as np
import tensorflow as tf
import time
import logging
import utils
import network



logger = logging.getLogger("NRL")


def get_adj_matrix(nodes_size, edges_list):
    adj_matrix = np.zeros([nodes_size,nodes_size],dtype=np.int32)
    for x, y in edges_list:
        adj_matrix[x, y] = 1
        adj_matrix[y, x] = 1
    return adj_matrix

def scale_sim_mat(mat):
    # Scale Matrix by row
    mat  = mat - np.diag(np.diag(mat))
    D_inv = np.diag(np.reciprocal(np.sum(mat,axis=0)))
    mat = np.dot(D_inv,  mat)

    return mat

def random_surfing(adj_matrix, max_step, alpha):
    # Random Surfing
    nm_nodes = len(adj_matrix)
    adj_matrix = scale_sim_mat(adj_matrix)
    P0 = np.eye(nm_nodes, dtype='float32')
    M = np.zeros((nm_nodes,nm_nodes),dtype='float32')
    P = np.eye(nm_nodes, dtype='float32')
    for i in range(0,max_step):
        P = alpha*np.dot(P,adj_matrix) + (1-alpha)*P0
        M = M + P

    return M

def PPMI_matrix(M):

    M = scale_sim_mat(M)
    nm_nodes = len(M)

    col_s = np.sum(M, axis=0).reshape(1,nm_nodes)
    row_s = np.sum(M, axis=1).reshape(nm_nodes,1)
    D = np.sum(col_s)
    rowcol_s = np.dot(row_s,col_s)
    PPMI = np.log(np.divide(D*M,rowcol_s))
    PPMI[np.isnan(PPMI)] = 0.0
    PPMI[np.isinf(PPMI)] = 0.0
    PPMI[np.isneginf(PPMI)] = 0.0
    PPMI[PPMI<0] = 0.0

    return PPMI


class AdditiveGuassianNoiseAutoencoder(object):
    def __init__(self, nodes_size, struct, embedding_size = 128, scale = 0.1):
        self._nodes_size = nodes_size
        self._embedding_size = embedding_size
        self._scale = scale
        struct.insert(0, nodes_size)
        struct.append(embedding_size)
        self._layers = len(struct)
        self._weights = {}
        self._bias = {}

        # encoder
        for i in range(self._layers - 1):
            name = "encoder" + str(i)
            self._weights[name] = tf.Variable(tf.random_normal([struct[i], struct[i + 1]]), name=name)
            self._bias[name] = tf.Variable(tf.zeros([struct[i + 1]]), name=name)
        # decoder
        struct.reverse()
        for i in range(self._layers - 1):
            name = "decoder" + str(i)
            self._weights[name] = tf.Variable(tf.random_normal([struct[i], struct[i + 1]]), name=name)
            self._bias[name] = tf.Variable(tf.zeros([struct[i + 1]]), name=name)

    def inference(self, batch_input):
        """
        make_compute_graph
        :param batch_input: [batch_size, N] ????????????????
        :return:
        """
        def encoder(X):
            for i in range(self._layers - 1):
                name = "encoder" + str(i)
                X = tf.nn.relu(tf.matmul(X, self._weights[name]) + self._bias[name])
            return X
        def decoder(X):
            for i in range(self._layers - 2):
                name = "decoder" + str(i)
                X = tf.nn.relu(tf.matmul(X, self._weights[name]) + self._bias[name])
            name = "decoder" + str(self._layers - 2)
            X = tf.nn.sigmoid(tf.matmul(X, self._weights[name]) + self._bias[name])
            return X
        # add noise
        batch_input = batch_input + self._scale * tf.random_normal((self._nodes_size,))
        batch_embeddings = encoder(batch_input)
        batch_reconstruct = decoder(batch_embeddings)
        return batch_embeddings, batch_reconstruct

    def loss(self, batch_input, batch_reconstruct):
        # mse loss
        return 0.5 * tf.reduce_sum(tf.pow(tf.subtract(batch_reconstruct, batch_input), 2.0))

    def optimize(self, loss, global_step, lr):
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

    def train(self, batch_input, global_step, learning_rate):
        batch_embeddings, batch_reconstruct = self.inference(batch_input)
        loss = self.loss(batch_input, batch_reconstruct)
        train_op = self.optimize(loss, global_step, learning_rate)
        return train_op, loss, batch_embeddings

## train
def train(dataset, vectors_path, lr_file, ckpt_dir, checkpoint, embedding_size,
          nodes_size, struct, gause_scale, idx_nodes,
          iter_epochs, batch_size, initial_learning_rate, decay_epochs, decay_interval, decay_rate,
          allow_soft_placement, log_device_placement, gpu_memory_fraction, using_gpu, allow_growth,
          loss_interval, summary_steps, summary_interval, ckpt_epochs, ckpt_interval):
    num_steps_per_epoch = int(nodes_size / batch_size) #
    iter_steps = round(iter_epochs * num_steps_per_epoch) # iter_epochs should be big enough to converge.
    decay_steps = round(decay_epochs * num_steps_per_epoch)
    ckpt_steps = round(ckpt_epochs * num_steps_per_epoch)

    LR = utils.LearningRateGenerator(initial_learning_rate = initial_learning_rate, initial_steps = 0,
                                     decay_rate = decay_rate, decay_steps = decay_steps, iter_steps = iter_steps)

    with tf.Graph().as_default(), tf.device('/gpu:0' if using_gpu else '/cpu:0'):

        global_step = tf.Variable(0, trainable=False, name="global_step")
        inputs = tf.placeholder(tf.float32, [None, nodes_size])
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        model = AdditiveGuassianNoiseAutoencoder(nodes_size, struct, embedding_size, gause_scale)

        train_op, loss, embeddings = model.train(inputs, global_step, learning_rate)

        # Create a saver.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)

        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init_op = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU implementations.
        config = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        config.gpu_options.allow_growth = allow_growth
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
            last_loss_time = time.time() - loss_interval
            last_summary_time = time.time() - summary_interval
            last_decay_time = last_checkpoint_time = time.time()
            last_decay_step = last_summary_step = last_checkpoint_step = 0
            while True:
                start_time = time.time()
                batch_input, batch_adj = dataset.next_batch(batch_size, keep_strict_batching = True)
                feed_dict = {inputs: batch_input,
                             learning_rate: LR.learning_rate}

                _, loss_value, cur_step = sess.run([train_op, loss, global_step], feed_dict=feed_dict)
                now = time.time()

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                epoch, epoch_step = divmod(cur_step,num_steps_per_epoch)

                if now - last_loss_time >= loss_interval:
                    format_str = '%s: step=%d(%d/%d), lr=%.6f, loss=%.6f, duration/step=%.4fs'
                    logger.info(format_str % ( time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                                               cur_step, epoch_step, epoch, LR.learning_rate, loss_value, now - start_time))
                    last_loss_time = time.time()
                if now - last_summary_time >= summary_interval or cur_step - last_summary_step >= summary_steps or cur_step >= iter_steps:
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, cur_step)
                    last_summary_time = time.time()
                    last_summary_step = cur_step
                ckpted = False
                # Save the model checkpoint periodically. (named 'model.ckpt-global_step.meta')
                if now - last_checkpoint_time >= ckpt_interval or cur_step - last_checkpoint_step >= ckpt_steps or cur_step >= iter_steps:
                    checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=cur_step)
                    vecs = []
                    start = 0
                    while start<nodes_size:
                        end = min(nodes_size, start + batch_size)
                        index = np.arange(start,end)
                        start = end
                        batch_input, batch_adj = dataset.get_batch(index)
                        feed_dict = {inputs: batch_input,
                                     learning_rate: LR.learning_rate}
                        batch_embeddings = sess.run(embeddings, feed_dict=feed_dict)
                        vecs.append(batch_embeddings)
                    vecs = np.concatenate(vecs, axis=0)
                    utils.save_word2vec_format(vectors_path, vecs, idx_nodes)
                    last_checkpoint_time = time.time()
                    last_checkpoint_step = cur_step
                    ckpted = True
                # update learning rate
                if ckpted or now - last_decay_time >= decay_interval or (decay_steps > 0 and cur_step - last_decay_step >= decay_steps):
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



def train_vectors(options):
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
        logger.info('restore a embedding train using tensorflow from latest ckpt...')
    else:
        logger.info('restore a embedding train using tensorflow from ckpt-%s...'%checkpoint)
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

    GaussianNoiseScale = 0.2
    max_step = 10
    alpha = options.walk_restart
    adj_matrix = get_adj_matrix(net.get_nodes_size(), net.edges)
    adj_matrix = random_surfing(adj_matrix, max_step, alpha)
    adj_matrix = PPMI_matrix(adj_matrix)

    dataset = utils.DataSet(adj_matrix, shuffled = not options.unshuffled)

    # train info
    logger.info('Train info:')
    logger.info('\t train_model = {}'.format(options.model))
    logger.info('\t total embedding nodes = {}'.format(net.get_nodes_size()))
    logger.info('\t total edges = {}'.format(net.get_edges_size()))
    logger.info('\t embedding size = {}'.format(options.embedding_size))
    logger.info('\t struct = {}'.format(options.struct))
    logger.info('\t alpha = {}'.format(alpha))
    logger.info('\t max_step = {}'.format(max_step))
    logger.info('\t GaussianNoiseScale = {}'.format(GaussianNoiseScale))
    logger.info('\t shuffled in training = {}'.format(not options.unshuffled))
    logger.info('\t batch_size = {}'.format(options.batch_size))
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
    logger.info('\t gpu_memory_allow_growth = {}\n'.format(options.allow_growth))

    logger.info('\t ckpt_dir = {}'.format(ckpt_dir))
    logger.info('\t vectors_path = {}'.format(options.vectors_path))
    logger.info('\t learning_rate_path = {}'.format(lr_file))

    fr_vec = open(os.path.join(train_vec_dir, 'embedding.info'), 'w')
    fr_vec.write('embedding info:\n')
    fr_vec.write('\t train_model = {}\n'.format(options.model))
    fr_vec.write('\t total embedding nodes = {}\n'.format(net.get_nodes_size()))
    fr_vec.write('\t total edges = {}\n'.format(net.get_edges_size()))
    fr_vec.write('\t embedding size = {}\n'.format(options.embedding_size))
    fr_vec.write('\t struct = {}\n'.format(options.struct))
    fr_vec.write('\t alpha = {}\n'.format(alpha))
    fr_vec.write('\t max_step = {}\n'.format(max_step))
    fr_vec.write('\t GaussianNoiseScale = {}\n'.format(GaussianNoiseScale))
    fr_vec.write('\t shuffled in training = {}\n'.format(not options.unshuffled))
    fr_vec.write('\t sparse_dot = {}\n'.format(options.sparse_dot))
    fr_vec.write('\t batch_size = {}\n'.format(options.batch_size))
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
    fr_vec.write('\t gpu_memory_allow_growth = {}\n\n'.format(options.allow_growth))

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

    # train
    logger.info('training...')
    time_start = time.time()
    train(dataset = dataset, vectors_path = options.vectors_path, lr_file = lr_file,
          ckpt_dir = ckpt_dir, checkpoint = checkpoint,
          embedding_size = options.embedding_size, struct = options.struct,
          nodes_size=net.get_nodes_size(), gause_scale = GaussianNoiseScale,
          batch_size = options.batch_size, idx_nodes = net._idx_nodes,
          initial_learning_rate = options.learning_rate, decay_epochs = options.decay_epochs,
          decay_rate = options.decay_rate, iter_epochs = options.iter_epoches,
          allow_soft_placement = options.allow_soft_placement, log_device_placement = options.log_device_placement,
          gpu_memory_fraction = options.gpu_memory_fraction, using_gpu = options.using_gpu,
          allow_growth = options.allow_growth, loss_interval = options.loss_interval, summary_steps = options.summary_steps,
          ckpt_interval = options.ckpt_interval, ckpt_epochs = options.ckpt_epochs, summary_interval = options.summary_interval,
          decay_interval=options.decay_interval)
    logger.info('train completed in {}s'.format(time.time() - time_start))
    return

