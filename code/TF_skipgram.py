#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0
ref:
    https://github.com/tensorflow/models/tree/master/tutorials/embedding
    https://www.tensorflow.org/tutorials/word2vec
"""


""" Multi-threaded word2vec mini-batched skip-gram model. """

import os
import sys
import gc
import time
import logging
from collections import defaultdict
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import utils
from utils import DataSet
import walker


# os.environ['CUDA_VISIBLE_DEVICES']='0,1'  #

logger = logging.getLogger("NRL")




def get_vocabs_from_sentences(sentences):
    """Do an initial scan of all words appearing in sentences."""
    # logger.info("collecting all nodes and their counts")
    # total_words = 0
    # total_sens = 0
    vocab = defaultdict(int)
    for sentence in sentences:
        # total_sens += 1
        for word in sentence:
            node = int(word)  # ???
            vocab[node] += 1
            # total_words += 1
    # logger.info( "collected %i nodes from a corpus of %i raw words and %i sentences",
    #              len(vocab), total_words, total_sens)
    return vocab

def get_vocabs_from_files(args):
    files, worker_id, work_times = args
    assert len(files)==work_times, "error, %d != %d" % (len(files),work_times)
    logger.info('worker-{} starts to scan {} files: {}'.format(worker_id, work_times, files))
    time_start = time.time()
    vocab = get_vocabs_from_sentences(walker.WalksCorpus(files))
    logger.info('worker-{} ended in {}s'.format(worker_id, time.time() - time_start))
    return vocab

def get_examples_from_sentences(sentences, vocab2idx, window_size):
    # logger.info("generating traing data and labels")
    data = []
    labels = []
    for sentence in sentences:
        word_ids = [vocab2idx[int(word)] for word in sentence]
        for pos in range(0,len(word_ids)):
            target_word = word_ids[pos]
            left_pos = max(pos - window_size, 0)
            right_pos = min(pos + window_size+1, len(word_ids))
            for context_word in word_ids[left_pos:right_pos]:
                # if context_word == target_word:
                #     continue
                data.append(context_word)
                labels.append(target_word)
    return np.asarray(data), np.asarray(labels)

def get_examples_from_files(args):
    files, worker_id, work_times, vocab2idx, window_size = args
    assert len(files)==work_times, "error, %d != %d" % (len(files),work_times)
    logger.info('worker-{} starts to scan {} files: {}'.format(worker_id, work_times, files))
    time_start = time.time()
    data, labels = get_examples_from_sentences(walker.WalksCorpus(files),vocab2idx, window_size)
    logger.info('worker-{} ended in {}s'.format(worker_id, time.time() - time_start))
    return data, labels

def scan_files_using_multiprocess(files, max_num_workers=cpu_count(),func=get_vocabs_from_files,args=()):
    if max_num_workers <= 1 or len(files) <= 1:
        if max_num_workers > 1:
            logger.info('corpus files less than 2, using single-process instead...')
        times_per_worker = [len(files)]
    else:
        if len(files) <= max_num_workers:
            times_per_worker = [1 for _ in range(len(files))]
        else:
            div, mod = divmod(len(files), max_num_workers)
            times_per_worker = [div for _ in range(max_num_workers)]
            for idx in range(mod):
                times_per_worker[idx] = times_per_worker[idx] + 1
    assert sum(times_per_worker) == len(files), 'workers allocating failed: %d != %d' % (
        sum(times_per_worker), len(files))

    counter = 0
    files_list = []
    for c in times_per_worker:
        files_list.append(files[counter:counter+c])
        counter = counter + c

    args_list = []
    for index in range(len(times_per_worker)):
        args_list.append((files_list[index], index, times_per_worker[index])+args)

    logger.info('scan corpus files (using %d workers for multi-process)...' % len(times_per_worker))
    time_start = time.time()
    rets = []
    with ProcessPoolExecutor(max_workers=max_num_workers) as executor:
        for ret in executor.map(func, args_list):
            rets.append(ret)
    assert len(rets) == len(files_list), 'ProcessPoolExecutor occured error, %d!=%d' % (
    len(rets), len(files_list))
    logger.info('scan corpus files completed in {}s'.format(time.time() - time_start))
    return rets

def scan_vocabs(vocabs, save_file):
    logger.info("collecting all nodes and their counts")
    total_vocab = defaultdict(int)
    total_words = 0
    for vocab in vocabs:
        for k,v in vocab.items():
            total_vocab[k] += v
            total_words += v
    logger.info("collected %i nodes from a corpus of %i raw words", len(total_vocab), total_words)

    vocab2idx = {}
    idx2vocab = []
    sample_frequencies = []
    fr = open(save_file,'w')
    for k, v in sorted(total_vocab.items(), key=lambda item:item[0]):
        # q = np.power(v,power)
        fr.write("{} {} {}\n".format(len(idx2vocab), k, v))
        vocab2idx[k] = len(idx2vocab)
        idx2vocab.append(k)
        sample_frequencies.append(v)
    fr.close()
    return vocab2idx, idx2vocab, sample_frequencies

def generate_train_data(corpus_store_path, headflag_of_index_file, train_workers, window_size,
                        idx_vocab_freq_file, sens = None, always_rebuild = False):
    corpusfiles_list = []
    if sens == None:
        # check index file
        with open(corpus_store_path, 'r') as f:
            headline = f.readline().strip()
            if headline == headflag_of_index_file:
                logger.info('generate training examples from corpus files: ')
                for line in f:
                    line = line.strip()
                    if line[0:5] == 'FILE:':
                        if os.path.exists(line[6:]):
                            logger.info('corpus file: {}'.format(line[6:]))
                            corpusfiles_list.append(line[6:])
                        else:
                            logger.warning('cannot find corpus file: {}, skiped.'.format(line[6:]))
            else:
                corpusfiles_list.append(corpus_store_path)
                logger.info('generate training examples from file: {}...'.format(corpusfiles_list))
    else:
        logger.info('generate training examples from memory sentences...')
    # generate train data
    if utils.check_rebuild(idx_vocab_freq_file, descrip='vocab and frequencies', always_rebuild=always_rebuild):
        logger.info('get vocabs ...')
        time_start = time.time()
        if sens == None:
            vocabs = scan_files_using_multiprocess(corpusfiles_list,
                                                   max_num_workers=train_workers,
                                                   func=get_vocabs_from_files)
        else:
            vocabs = [get_vocabs_from_sentences(sens)]
        logger.info('get vocabs completed in {}s'.format(time.time() - time_start))
        logger.info('get frequencies ...')
        time_start = time.time()
        vocab2idx, idx2vocab, nodes_frequencies = scan_vocabs(vocabs, idx_vocab_freq_file)
        logger.info('get frequencies completed in {}s'.format(time.time() - time_start))
    else:
        logger.info("get vocab and frequencies from file: {}".format(idx_vocab_freq_file))
        time_start = time.time()
        vocab2idx = {}
        idx2vocab = []
        nodes_frequencies = []
        # count = 0
        for line in open(idx_vocab_freq_file):
            linelist = line.strip().split(' ')
            idx = int(linelist[0])
            node = int(linelist[1])
            freq = float(linelist[2])
            vocab2idx[node] = idx
            # assert count == idx, "error, %d != %d" %(count,idx)
            # count += 1
            idx2vocab.append(idx)
            nodes_frequencies.append(freq)
        logger.info('get vocab and frequencies completed in {}s'.format(time.time()-time_start))

    logger.info('get training examples ...')
    time_start = time.time()
    if sens == None:
        rets = scan_files_using_multiprocess(corpusfiles_list,
                                               max_num_workers= train_workers,
                                               func=get_examples_from_files,
                                               args=(vocab2idx, window_size))
    else:
        rets = [get_examples_from_sentences(sens,vocab2idx, window_size)]
    logger.info('get training examples completed in {}s'.format(time.time() - time_start))
    data = np.concatenate([item[0] for item in rets])
    labels = np.concatenate([item[1] for item in rets])
    # logger.info('total nodes: {}, total examples: {}'.format(len(nodes_frequencies), len(data)))
    # dataset = DataSet(data=data, labels=labels, shuffled= not options.unshuffled)
    return data, labels, idx2vocab, nodes_frequencies


## model
class Word2Vec(object):
    """Word2Vec model (Skipgram)."""
    def __init__(self, vocab_size, embedding_size = 128, vocab_unigrams = None, neg_sampled = 6,
                 distortion_power = 0.75, batch_size = 16,
                 # initial_learning_rate = 0.01, decay_steps = 0, decay_rate = 0.1
                 ):

        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._vocab_unigrams = vocab_unigrams
        self._num_sampled = neg_sampled
        self._distortion_power = distortion_power
        self._batch_size = batch_size
        # self._initial_learning_rate = initial_learning_rate
        # self._decay_steps = decay_steps
        # self._decay_rate = decay_rate

        init_width = 0.5 / self._embedding_size
        self._embedding = tf.Variable(
            tf.random_uniform([self._vocab_size, self._embedding_size], -init_width, init_width),
            name='embedding')
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

    def inference(self, batch_input, batch_labels):
        """
        construct the model
        :param input: 1D tensor of idx in vocabulary
        :param batch_labels: 1D tensor if index, positive samples
        :return:
        """

        ## Embedding layer
        # self._embedding: 2D, N*d
        # batch_input: 1D, (n)  # batch_size
        # embed: 2D, n*d
        embed = tf.nn.embedding_lookup(self._embedding, batch_input)

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape( tf.cast(batch_labels, dtype=tf.int64), [-1, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=self._num_sampled,
            unique=True,
            range_max=self._vocab_size,
            distortion=self._distortion_power,
            unigrams=self._vocab_unigrams))

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(self._nce_weight, batch_labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(self._nce_biases, batch_labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(self._nce_weight, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(self._nce_biases, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.multiply(embed, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [self._num_sampled])
        sampled_logits = tf.matmul(embed,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits

    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / self._batch_size
        tf.summary.scalar("NCE loss", nce_loss_tensor)
        self._loss = nce_loss_tensor
        return nce_loss_tensor

    def optimize(self, loss, global_step, lr):
        """Build the graph to optimize the loss function."""

        # Optimizer nodes.
        # Linear learning rate decay.
        # lr = tf.Variable(self._initial_learning_rate, trainable=False, name="lr")
        # # lr = self._initial_learning_rate
        # if self._decay_steps > 0:
        #     # Decay the learning rate exponentially based on the number of steps.
        #     # decayed_learning_rate = initial_learning_rate * decay_rate ^ (global_step / decay_steps)
        #     lr = tf.train.exponential_decay(self._initial_learning_rate,
        #                                     global_step,
        #                                     self._decay_steps,
        #                                     self._decay_rate,
        #                                     staircase=True)

        tf.summary.scalar('learning_rate', lr)

        # Compute gradients
        # opt = tf.train.MomentumOptimizer(lr, graphcnn_option.MOMENTUM)
        # opt = tf.train.AdamOptimizer(lr)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.minimize(loss,
                                      global_step=global_step,
                                      gate_gradients=optimizer.GATE_NONE)
        return train_op

    def train(self, batch_input, batch_labels, global_step, learning_rate):
        true_logits, sampled_logits = self.inference(batch_input, batch_labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        train_op = self.optimize(loss, global_step, learning_rate)
        return train_op, loss

def save_word2vec_format(vector_path, vectors, idx2vocab):
    vocab_size = np.size(vectors, axis=0)
    vector_size = np.size(vectors, axis=1)
    with open(vector_path, 'w') as fr:
        fr.write('%d %d\n' % (vocab_size, vector_size))
        for index in range(vocab_size):
            fr.write('%d ' % idx2vocab[index])
            for one in vectors[index]:
                fr.write('{} '.format(one))
            fr.write('\n')


def _train_thread_body(dataset, batch_size, inputs, labels, session, train_op, iter_steps, global_step,
                       learning_rate, LR):
    while True:
        batch_data, batch_labels = dataset.next_batch(batch_size, keep_strict_batching=True)
        feed_dict = {inputs: batch_data, labels: batch_labels, learning_rate: LR.learning_rate}
        _, cur_step = session.run([train_op, global_step], feed_dict=feed_dict)
        if cur_step >= iter_steps:
            break

## train
def train(dataset, vectors_path, lr_file,
          ckpt_dir, checkpoint, idx2vocab, vocab_unigrams, embedding_size, neg_sampled,
          distortion_power, batch_size, initial_learning_rate, decay_epochs, decay_rate, iter_epochs,
          allow_soft_placement, log_device_placement, gpu_memory_fraction, using_gpu, allow_growth,
          loss_interval, summary_steps, ckpt_interval, ckpt_epochs, summary_interval, decay_interval,
          train_workers):

    num_steps_per_epoch = int(dataset.num_examples / batch_size)
    iter_steps = iter_epochs * num_steps_per_epoch
    decay_steps = int(decay_epochs * num_steps_per_epoch)
    ckpt_steps = int(ckpt_epochs * num_steps_per_epoch)

    LR = utils.LearningRateGenerator(initial_learning_rate = initial_learning_rate, initial_steps = 0,
                                     decay_rate = decay_rate, decay_steps = decay_steps)

    with tf.Graph().as_default(), tf.device('/gpu:0' if using_gpu else '/cpu:0'):

        global_step = tf.Variable(0, trainable=False, name="global_step")

        inputs = tf.placeholder(tf.int32, shape=[batch_size], name='inputs')
        labels = tf.placeholder(tf.int32, shape=[batch_size], name='labels')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        model = Word2Vec(vocab_size = len(idx2vocab), embedding_size = embedding_size, vocab_unigrams = vocab_unigrams,
                         neg_sampled = neg_sampled, distortion_power = distortion_power, batch_size = batch_size)

        train_op, loss = model.train(inputs, labels, global_step, learning_rate)

        # Create a saver.
        saver = tf.train.Saver(var_list=tf.global_variables(),
                               max_to_keep=5)

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
            executor_workers = train_workers-1
            if executor_workers > 0:
                executor = ThreadPoolExecutor(max_workers=executor_workers)
                for _ in range(executor_workers):
                    executor.submit(_train_thread_body,
                                    dataset, batch_size, inputs, labels, sess, train_op, iter_steps, global_step,
                                    learning_rate, LR)

            last_loss_time = time.time() - loss_interval
            last_summary_time = time.time() - summary_interval
            last_decay_time = last_checkpoint_time = time.time()
            last_decay_step = last_summary_step = last_checkpoint_step = 0
            while True:
                start_time = time.time()
                batch_data, batch_labels = dataset.next_batch(batch_size, keep_strict_batching=True)
                feed_dict = {inputs: batch_data, labels: batch_labels, learning_rate: LR.learning_rate}
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
                    # embedding_vectors = sess.run(model.vectors, feed_dict=feed_dict)
                    vecs,weights,biases = sess.run([model.vectors,model.context_weights,model.context_biases],
                                                 feed_dict=feed_dict)
                    save_word2vec_format(vectors_path, vecs, idx2vocab)
                    np.savetxt(vectors_path+".contexts",weights)
                    np.savetxt(vectors_path+".context_biases",biases)
                    last_checkpoint_time = time.time()
                    last_checkpoint_step = cur_step
                    ckpted = True
                # update learning rate
                if ckpted or now - last_decay_time >= decay_interval or cur_step - last_decay_step >= decay_steps:
                    lr_info = np.loadtxt(lr_file, dtype=float)
                    if np.abs(lr_info[1]-decay_epochs) >= 1e-7:
                        decay_epochs = lr_info[1]
                        decay_steps = int(decay_epochs * num_steps_per_epoch)
                    if np.abs(lr_info[2]-decay_rate) >= 1e-7:
                        decay_rate = lr_info[2]
                    if np.abs(lr_info[0]-initial_learning_rate) < 1e-7:
                        LR.exponential_decay(cur_step, decay_rate = decay_rate, decay_steps = decay_steps)
                    else:
                        initial_learning_rate = lr_info[0]
                        LR.reset(initial_learning_rate = initial_learning_rate, initial_steps = cur_step,
                                 decay_rate = decay_rate, decay_steps = decay_steps)
                    last_decay_time = time.time()
                    last_decay_step = cur_step

                if cur_step >= iter_steps:
                    break


def train_vectors(options, sens = None):
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

    # generate training examples
    logger.info("Generate training examples:")
    idx_vocab_freq_file = os.path.join(train_vec_dir, 'vocab.freq')
    logger.info('\t corpus_store_path = {}'.format(options.corpus_store_path))
    logger.info('\t vocab and frequencies file = {}'.format(idx_vocab_freq_file))
    logger.info('\t walk_workers to load dataset = {}'.format(options.walk_workers))
    logger.info('\t window_size = {}'.format(options.window_size))
    data, labels, idx2vocab, nodes_frequencies = generate_train_data(options.corpus_store_path,
                                                                     options.headflag_of_index_file,
                                                                     options.walk_workers,
                                                                     options.window_size,
                                                                     idx_vocab_freq_file,
                                                                     sens=sens,
                                                                     always_rebuild=options.always_rebuild)
    del sens
    gc.collect()
    dataset = DataSet(data=data, labels=labels, shuffled=not options.unshuffled)

    lr_file = os.path.join(train_vec_dir, "lr.info")
    np.savetxt(lr_file, np.asarray([options.learning_rate, options.decay_epochs,options.decay_rate],dtype=np.float32),
               fmt="%.6f")

    # train info
    logger.info('Train info:')
    logger.info('\t total embedding nodes = {}'.format(len(idx2vocab)))
    logger.info('\t total training examples = {}'.format(len(data)))
    logger.info('\t shuffled in training = {}'.format(not options.unshuffled))
    logger.info('\t embedding size = {}'.format(options.embedding_size))
    logger.info('\t window size = {}'.format(options.window_size))
    logger.info('\t negative = {}'.format(options.negative))
    logger.info('\t distortion_power = {}\n'.format(options.distortion_power))
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
    logger.info('\t gpu_memory_allow_growth = {}'.format(options.allow_growth))
    logger.info('\t train_workers = {}\n'.format(options.train_workers))

    logger.info('\t ckpt_dir = {}'.format(ckpt_dir))
    logger.info('\t vectors_path = {}'.format(options.vectors_path))
    logger.info('\t learning_rate_path = {}'.format(lr_file))

    fr_vec = open(os.path.join(train_vec_dir, 'embedding.info'), 'w')
    fr_vec.write('embedding info:\n')
    fr_vec.write('\t corpus_store_path = {}\n'.format(options.corpus_store_path))
    fr_vec.write('\t vocab and frequencies file = {}\n'.format(idx_vocab_freq_file))
    fr_vec.write('\t total embedding nodes = {}\n'.format(len(idx2vocab)))
    fr_vec.write('\t total training examples = {}\n'.format(len(data)))
    fr_vec.write('\t shuffled in training = {}\n'.format(not options.unshuffled))
    fr_vec.write('\t embedding size = {}\n'.format(options.embedding_size))
    fr_vec.write('\t window size = {}\n'.format(options.window_size))
    fr_vec.write('\t negative = {}\n'.format(options.negative))
    fr_vec.write('\t distortion_power = {}\n\n'.format(options.distortion_power))
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
    fr_vec.write('\t gpu_memory_allow_growth = {}\n'.format(options.allow_growth))
    fr_vec.write('\t train_workers = {}\n\n'.format(options.train_workers))

    fr_vec.write('\t ckpt_dir = {}\n'.format(ckpt_dir))
    fr_vec.write('\t vectors_path = {}\n'.format(options.vectors_path))
    fr_vec.write('\t learning_rate_path = {}\n'.format(lr_file))

    fr_vec.close()

    visible_devices = str(options.visible_device_list[0])
    for dev in options.visible_device_list[1:]:
        visible_devices = visible_devices + ',%s' % dev
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices

    # train
    logger.info('training...')
    time_start = time.time()
    train(dataset = dataset, vectors_path = options.vectors_path, lr_file = lr_file,
          ckpt_dir = ckpt_dir, checkpoint = checkpoint, idx2vocab = idx2vocab, vocab_unigrams = nodes_frequencies,
          embedding_size = options.embedding_size, neg_sampled = options.negative,
          distortion_power = options.distortion_power, batch_size = options.batch_size,
          initial_learning_rate = options.learning_rate, decay_epochs = options.decay_epochs,
          decay_rate = options.decay_rate, iter_epochs = options.iter_epoches,
          allow_soft_placement = options.allow_soft_placement, log_device_placement = options.log_device_placement,
          gpu_memory_fraction = options.gpu_memory_fraction, using_gpu = options.using_gpu,
          allow_growth = options.allow_growth, loss_interval = options.loss_interval, summary_steps = options.summary_steps,
          ckpt_interval = options.ckpt_interval, ckpt_epochs = options.ckpt_epochs, summary_interval = options.summary_interval,
          decay_interval=options.decay_interval, train_workers = options.train_workers)
    logger.info('train completed in {}s'.format(time.time() - time_start))
    return











