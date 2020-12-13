#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.5.0 

mostly referenced from https://github.com/suanrong/SDNE


"""
import os
import sys
import gc
import numpy as np
import tensorflow as tf
import time
import logging
from scipy.sparse import dok_matrix
import utils
import network


logger = logging.getLogger("NRL")


class rbm:
    def __init__(self, inputsize, outputsize, batchsize=16, learning_rate=0.005, config=None):
        """
        :param inputsize: int
        :param outputsize: int
        :param batchsize: int
        :param learning_rate: float
        """
        # v是visiblelayer的意思
        # h是hiddenlayer的意思，指的是输出层。
        self._learning_rate = learning_rate
        self._batchsize = batchsize
        if config is None:
            self.sess = tf.Session()
        else:
            self.sess = tf.Session(config=config)
        stddev = 1.0 / np.sqrt(inputsize)
        # 为什么初始化的时候权重要设置为这个
        self.W = tf.Variable(tf.random_normal([inputsize, outputsize], stddev=stddev), name="Wii")
        self.bias = {"vb": tf.Variable(tf.zeros(inputsize), name="visible_bias"),
                     "hb": tf.Variable(tf.zeros(outputsize), name="hidden_bias")
                     }

        self.v = tf.placeholder(tf.float32, [None, inputsize], name="visiable_input")
        logger.info("rbm\t\tvsize = " + str(inputsize))
        # 输入向量v
        self.sess.run(tf.global_variables_initializer())
        self.buildModel()
        pass

    def buildModel(self):
        self.h = self.sample(tf.sigmoid(tf.matmul(self.v, self.W) + self.bias['hb']))
        # gibbs_sample
        v_sample = self.sample(tf.sigmoid(tf.matmul(self.h, tf.transpose(self.W)) + self.bias['vb']))
        h_sample = self.sample(tf.sigmoid(tf.matmul(v_sample, self.W) + self.bias['hb']))
        lr = self._learning_rate / tf.to_float(self._batchsize)
        W_adder = self.W.assign_add(
            lr * (tf.matmul(tf.transpose(self.v), self.h) - tf.matmul(tf.transpose(v_sample), h_sample)))
        bv_adder = self.bias['vb'].assign_add(lr * tf.reduce_mean(self.v - v_sample, 0))
        bh_adder = self.bias['hb'].assign_add(lr * tf.reduce_mean(self.h - h_sample, 0))
        self.upt = [W_adder, bv_adder, bh_adder]
        self.error = tf.reduce_sum(tf.pow(self.v - v_sample, 2))

    def fit(self, data):
        _, ret = self.sess.run((self.upt, self.error), feed_dict={self.v: data})
        return ret

    def sample(self, probs):
        # 随机向上或向下取整？没太看懂
        # 这里好像是说随机成01的概率值。
        return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

    def getWb(self):
        return self.sess.run([self.W, self.bias['vb'], self.bias['hb']])

    def getH(self, data):
        return self.sess.run(self.h, feed_dict={self.v: data})


class DataSet(object):
    def __init__(self, nodes_size, edges_list, shuffled = True):
        self._nodes_size = nodes_size
        self._adj_matrix = dok_matrix((nodes_size, nodes_size), np.int_)
        for x, y in edges_list:
            self._adj_matrix[x, y] = 1
            # self._adj_matrix[y, x] = 1
        self._adj_matrix = self._adj_matrix.tocsr()
        self._order = np.arange(self._nodes_size)
        self._shuffled = shuffled
        self._epochs_completed = 0
        self._index_in_epoch = 0
        if shuffled:
            np.random.shuffle(self._order)

    @property
    def nodes_size(self):
        return self._nodes_size

    def next_batch(self, batch_size, keep_strict_batching = True):
        if keep_strict_batching:
            assert batch_size <= self._nodes_size

        if self._index_in_epoch >= self._nodes_size:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffled:
                np.random.shuffle(self._order)
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
                    np.random.shuffle(self._order)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                self._index_in_epoch = self._nodes_size
        end = self._index_in_epoch
        index = self._order[start:end]
        batch_input, batch_adj = self.get_batch(index)

        return batch_input, batch_adj

    def get_batch(self, index):
        batch_input = self._adj_matrix[index].toarray()
        batch_adj = self._adj_matrix[index].toarray()[:][:, index]
        return batch_input, batch_adj


class SDNE(object):
    def __init__(self, nodes_size, struct, embedding_size=128, alpha=100, beta=10, gamma=1, reg=1,
                 sparse_dot=False, active_function = tf.nn.sigmoid):
        """
        SDNE model.
        :param nodes_size: N, the nodes size of graph.
        :param struct: [l1,l2,l3,...,lk], the struct(layer size) of the encoder/decoder.
        :param embedding_size: d, note that a encoder is N-l1-l2-...-lk-d.
        """
        self._nodes_size = nodes_size
        self._embedding_size = embedding_size
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._reg = reg
        self._sparse_dot = sparse_dot
        struct.insert(0, nodes_size)
        struct.append(embedding_size)
        self._struct = struct
        self._layers = len(struct)
        self._weights = {}
        self._bias = {}
        self._active_function = active_function
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
        struct.reverse()

    def inference(self, batch_input):
        """
        make_compute_graph
        :param batch_input: [batch_size, N] ????????????????
        :return:
        """

        def encoder(X):
            for i in range(self._layers - 1):
                name = "encoder" + str(i)
                X = self._active_function(tf.matmul(X, self._weights[name]) + self._bias[name])
            return X

        def encoder_sp(X):
            for i in range(self._layers - 1):
                name = "encoder" + str(i)
                if i == 0:
                    X = self._active_function(tf.sparse_tensor_dense_matmul(X, self._weights[name]) + self._bias[name])
                else:
                    X = self._active_function(tf.matmul(X, self._weights[name]) + self._bias[name])
            return X

        def decoder(X):
            for i in range(self._layers - 1):
                name = "decoder" + str(i)
                X = self._active_function(tf.matmul(X, self._weights[name]) + self._bias[name])
            return X

        if self._sparse_dot:
            batch_embeddings = encoder_sp(batch_input)
        else:
            batch_embeddings = encoder(batch_input)
        batch_reconstruct = decoder(batch_embeddings)
        return batch_embeddings, batch_reconstruct

    def loss(self, batch_input, batch_reconstruct, batch_embeddings, adjacent_matrix):
        def get_2nd_loss(X, newX, beta):
            B = X * (beta - 1) + 1
            return tf.reduce_sum(tf.pow((newX - X) * B, 2))

        def get_1st_loss(H, adj_mini_batch):
            D = tf.diag(tf.reduce_sum(adj_mini_batch, 1))
            L = D - adj_mini_batch  ## L is laplacian matrix
            return 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(H), L), H))

        def get_reg_loss(weight, biases):
            ret = tf.add_n([tf.nn.l2_loss(w) for w in weight.values()])
            ret = ret + tf.add_n([tf.nn.l2_loss(b) for b in biases.values()])
            return ret

        # Loss function
        if self._gamma > 0:
            loss_2nd = get_2nd_loss(batch_input, batch_reconstruct, self._beta)
            loss_2nd = self._gamma * loss_2nd
            tf.summary.scalar('loss_2nd', loss_2nd)
            tf.add_to_collection('losses', loss_2nd)

        if self._alpha > 0:
            loss_1st = get_1st_loss(batch_embeddings, adjacent_matrix)
            loss_1st = self._alpha * loss_1st
            tf.summary.scalar('loss_1st', loss_1st)
            tf.add_to_collection('losses', loss_1st)
        if self._reg > 0:
            loss_reg = get_reg_loss(self._weights, self._bias)
            loss_reg = self._reg * loss_reg
            tf.summary.scalar('loss_reg', loss_reg)
            tf.add_to_collection('losses', loss_reg)

        # the loss func is  // gamma * L2 + alpha * L1 + reg * regularTerm //
        # loss_total = loss_2nd + loss_1st + loss_reg

        loss_total = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar('loss_total', loss_total)

        return loss_total

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

    def train(self, batch_input, batch_adj, global_step, learning_rate):
        batch_embeddings, batch_reconstruct = self.inference(batch_input)
        loss = self.loss(batch_input, batch_reconstruct, batch_embeddings, batch_adj)
        train_op = self.optimize(loss, global_step, learning_rate)
        return train_op, loss, batch_embeddings


## train
def train(dataset, vectors_path, lr_file, ckpt_dir, checkpoint, embedding_size,
          struct, alpha, beta, gamma, reg, sparse_dot,
          iter_epochs, batch_size, initial_learning_rate, decay_epochs, decay_interval, decay_rate,
          allow_soft_placement, log_device_placement, gpu_memory_fraction, using_gpu, allow_growth,
          loss_interval, summary_steps, summary_interval, ckpt_epochs, ckpt_interval,
          dbn_initial, dbn_epochs, dbn_batchsize, dbn_learning_rate, active_function = "sigmoid"):
    actv_func = {'sigmoid':tf.sigmoid,
                 'tanh':tf.tanh,
                 'relu':tf.nn.relu,
                 'leaky_relu':tf.nn.leaky_relu
                 }[active_function]
    nodes_size = dataset.nodes_size
    num_steps_per_epoch = int(nodes_size / batch_size) #
    iter_steps = round(iter_epochs * num_steps_per_epoch) # iter_epochs should be big enough to converge.
    decay_steps = round(decay_epochs * num_steps_per_epoch)
    ckpt_steps = round(ckpt_epochs * num_steps_per_epoch)

    LR = utils.LearningRateGenerator(initial_learning_rate = initial_learning_rate, initial_steps = 0,
                                     decay_rate = decay_rate, decay_steps = decay_steps, iter_steps = iter_steps)

    with tf.Graph().as_default(), tf.device('/gpu:0' if using_gpu else '/cpu:0'):

        global_step = tf.Variable(0, trainable=False, name="global_step")
        adj_matrix = tf.placeholder(tf.float32, [None, None])
        if sparse_dot:
            inputs_sp_indices = tf.placeholder(tf.int64)
            inputs_sp_ids_val = tf.placeholder(tf.float32)
            inputs_sp_shape = tf.placeholder(tf.int64)
            inputs = tf.SparseTensor(inputs_sp_indices, inputs_sp_ids_val, inputs_sp_shape)
        else:
            inputs = tf.placeholder(tf.float32, [None, nodes_size])
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        model = SDNE(nodes_size=nodes_size, struct=struct, embedding_size=embedding_size,
                     alpha=alpha, beta=beta, gamma=gamma, reg=reg, sparse_dot=sparse_dot,
                     active_function=actv_func)

        train_op, loss, embeddings = model.train(inputs, adj_matrix, global_step, learning_rate)

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
                if dbn_initial:
                    time_start = time.time()
                    logger.info("DBN initial start ...")
                    RBMs = []
                    for i in range(len(model._struct) - 1):
                        RBM = rbm(model._struct[i], model._struct[i + 1],
                                  batchsize=dbn_batchsize,
                                  learning_rate=dbn_learning_rate,
                                  config=config)
                        logger.info("create rbm {}-{}".format(model._struct[i], model._struct[i + 1]))
                        RBMs.append(RBM)
                        for epoch in range(dbn_epochs):
                            error = 0
                            for batch in range(0, nodes_size, batch_size):
                                # 这句没动
                                # 它是遍历了全局的node？
                                mini_batch, _ = dataset.next_batch(batch_size)
                                for k in range(len(RBMs) - 1):
                                    mini_batch = RBMs[k].getH(mini_batch)
                                error += RBM.fit(mini_batch)
                            logger.info("rbm_" + str(len(RBMs)) + " epochs:" + str(epoch) + " error: " + str(error))

                        W, bv, bh = RBM.getWb()
                        name = "encoder" + str(i)

                        def assign(a, b, sessss):
                            op = a.assign(b)
                            sessss.run(op)

                        assign(model._weights[name], W, sess)
                        assign(model._bias[name], bh, sess)

                        name = "decoder" + str(len(model._struct)- i - 2)
                        assign(model._weights[name], W.transpose(), sess)
                        assign(model._bias[name], bv, sess)
                    logger.info("dbn_init finished in {}s.".format(time.time() - time_start))

                vecs = []
                start = 0
                while start < nodes_size:
                    end = min(nodes_size, start + batch_size)
                    index = np.arange(start, end)
                    start = end
                    batch_input, batch_adj = dataset.get_batch(index)
                    if sparse_dot:
                        batch_input_ind = np.vstack(np.where(batch_input)).astype(np.int64).T
                        batch_input_shape = np.array(batch_input.shape).astype(np.int64)
                        batch_input_val = batch_input[np.where(batch_input)]
                        feed_dict = {inputs_sp_indices: batch_input_ind,
                                     inputs_sp_shape: batch_input_shape,
                                     inputs_sp_ids_val: batch_input_val,
                                     adj_matrix: batch_adj,
                                     learning_rate: LR.learning_rate}
                    else:
                        feed_dict = {inputs: batch_input,
                                     adj_matrix: batch_adj,
                                     learning_rate: LR.learning_rate}
                    batch_embeddings = sess.run(embeddings, feed_dict=feed_dict)
                    vecs.append(batch_embeddings)
                vecs = np.concatenate(vecs, axis=0)
                checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
                utils.save_word2vec_format_and_ckpt(vectors_path, vecs, checkpoint_path, sess, saver, 0)

            elif checkpoint == '-1':  # load the latest one
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
                if sparse_dot:
                    batch_input_ind = np.vstack(np.where(batch_input)).astype(np.int64).T
                    batch_input_shape = np.array(batch_input.shape).astype(np.int64)
                    batch_input_val = batch_input[np.where(batch_input)]
                    feed_dict = {inputs_sp_indices: batch_input_ind,
                                 inputs_sp_shape: batch_input_shape,
                                 inputs_sp_ids_val: batch_input_val,
                                 adj_matrix: batch_adj,
                                 learning_rate: LR.learning_rate}
                else:
                    feed_dict = {inputs: batch_input,
                                 adj_matrix: batch_adj,
                                 learning_rate: LR.learning_rate}

                _, loss_value, cur_step = sess.run([train_op, loss, global_step], feed_dict=feed_dict)
                now = time.time()

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                epoch, epoch_step = divmod(cur_step, num_steps_per_epoch)

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
                    vecs = []
                    start = 0
                    while start < nodes_size:
                        end = min(nodes_size, start + batch_size)
                        index = np.arange(start, end)
                        start = end
                        batch_input, batch_adj = dataset.get_batch(index)
                        if sparse_dot:
                            batch_input_ind = np.vstack(np.where(batch_input)).astype(np.int64).T
                            batch_input_shape = np.array(batch_input.shape).astype(np.int64)
                            batch_input_val = batch_input[np.where(batch_input)]
                            feed_dict = {inputs_sp_indices: batch_input_ind,
                                         inputs_sp_shape: batch_input_shape,
                                         inputs_sp_ids_val: batch_input_val,
                                         adj_matrix: batch_adj,
                                         learning_rate: LR.learning_rate}
                        else:
                            feed_dict = {inputs: batch_input,
                                         adj_matrix: batch_adj,
                                         learning_rate: LR.learning_rate}
                        batch_embeddings = sess.run(embeddings, feed_dict=feed_dict)
                        vecs.append(batch_embeddings)
                    vecs = np.concatenate(vecs, axis=0)
                    checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
                    utils.save_word2vec_format_and_ckpt(vectors_path, vecs, checkpoint_path, sess, saver, cur_step)
                    last_checkpoint_time = time.time()
                    last_checkpoint_step = cur_step
                    ckpted = True
                # update learning rate
                if ckpted or now - last_decay_time >= decay_interval or (decay_steps > 0 and cur_step - last_decay_step >= decay_steps):
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


def train_vectors(options):
    # check vectors and ckpt
    checkpoint = '0'
    train_vec_dir = os.path.split(options.vectors_path)[0]
    ckpt_dir = os.path.join(train_vec_dir, 'ckpt')
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        cur_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        logger.info("model and vectors already exists, checkpoint step = {}".format(cur_step))
        checkpoint = input("please input 0 to start a new train, or input a choosed ckpt to restore (-1 for latest ckpt)")
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

    dataset = DataSet(nodes_size = net.get_nodes_size(), edges_list = net.edges, shuffled=not options.unshuffled)

    # train info
    logger.info('Train info:')
    logger.info('\t train_model = {}'.format(options.model))
    logger.info('\t total embedding nodes = {}'.format(net.get_nodes_size()))
    logger.info('\t total edges = {}'.format(net.get_edges_size()))
    logger.info('\t embedding size = {}'.format(options.embedding_size))
    logger.info('\t struct = {}'.format(options.struct))
    logger.info('\t alpha = {}'.format(options.alpha))
    logger.info('\t beta = {}'.format(options.beta))
    logger.info('\t gamma = {}'.format(options.gamma))
    logger.info('\t reg = {}\n'.format(options.reg))
    logger.info('\t shuffled in training = {}'.format(not options.unshuffled))
    logger.info('\t sparse_dot = {}'.format(options.sparse_dot))
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
    logger.info('\t dbn_initial = {}'.format(options.dbn_initial))
    logger.info('\t dbn_epochs = {}'.format(options.dbn_epochs))
    logger.info('\t dbn_batchsize = {}'.format(options.dbn_batchsize))
    logger.info('\t dbn_learning_rate = {}'.format(options.dbn_learning_rate))
    logger.info('\t active_function = {}\n'.format(options.active_function))
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
    fr_vec.write('\t alpha = {}\n'.format(options.alpha))
    fr_vec.write('\t beta = {}\n'.format(options.beta))
    fr_vec.write('\t gamma = {}\n'.format(options.gamma))
    fr_vec.write('\t reg = {}\n\n'.format(options.reg))
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
    fr_vec.write('\t dbn_initial = {}s\n'.format(options.dbn_initial))
    fr_vec.write('\t dbn_epochs = {}s\n'.format(options.dbn_epochs))
    fr_vec.write('\t dbn_batchsize = {}s\n'.format(options.dbn_batchsize))
    fr_vec.write('\t dbn_learning_rate = {}s\n'.format(options.dbn_learning_rate))
    fr_vec.write('\t active_function = {}\n'.format(options.active_function))
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
    train(dataset=dataset, vectors_path=options.vectors_path, lr_file=lr_file,
          ckpt_dir=ckpt_dir, checkpoint=checkpoint,
          embedding_size=options.embedding_size, struct=options.struct, alpha=options.alpha,
          beta=options.beta, gamma=options.gamma, reg=options.reg, sparse_dot=options.sparse_dot,
          batch_size=options.batch_size,
          initial_learning_rate=options.learning_rate, decay_epochs=options.decay_epochs,
          decay_rate=options.decay_rate, iter_epochs=options.iter_epoches,
          allow_soft_placement=options.allow_soft_placement, log_device_placement=options.log_device_placement,
          gpu_memory_fraction=options.gpu_memory_fraction, using_gpu=options.using_gpu,
          allow_growth=options.allow_growth, loss_interval=options.loss_interval, summary_steps=options.summary_steps,
          ckpt_interval=options.ckpt_interval, ckpt_epochs=options.ckpt_epochs,
          summary_interval=options.summary_interval,
          decay_interval=options.decay_interval, dbn_initial=options.dbn_initial, dbn_epochs=options.dbn_epochs,
          dbn_batchsize=options.dbn_batchsize, dbn_learning_rate=options.dbn_learning_rate,
          active_function = options.active_function)
    logger.info('train completed in {}s'.format(time.time() - time_start))
    return








