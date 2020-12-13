

import os
import shutil
import time
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor


import network
import eval_utils
import tensorflow as tf
import utils


logger = logging.getLogger("NRL")
features_matrix = None
net_origin = None
SAMPLE_NODES = None
SAMPLE_RULE = None
METIRC = None
PREC_K = None



def _eval(eval_graph):
    nodeID_list = list(eval_graph.nodes)
    eval_features = features_matrix[nodeID_list]
    similarity = eval_utils.getSimilarity(eval_features, metric=METIRC)
    for i in range(np.size(similarity, axis=0)):
        similarity[i,i] = 0
    precisionK_list = eval_utils.check_precK(nodeID_list, similarity, max_index=PREC_K, eval_graph=eval_graph, except_graph=None)
    MAP = eval_utils.check_MAP(nodeID_list, similarity, eval_graph=eval_graph, except_graph=None)
    return [MAP, precisionK_list]


def _sample_thread_body(repeated_times):
    ret_list = []
    for _ in range(repeated_times):
        logger.info("sampling {} nodes to constructe a sub-nework for evaluation ...".format(SAMPLE_NODES))
        time_start = time.time()
        net_sample = net_origin.sample_by_nodes(SAMPLE_NODES, rule = SAMPLE_RULE)
        logger.info('sampling nodes completed in {}s'.format(time.time() - time_start))
        ret =_eval(net_sample)
        ret_list.append(ret)
    return ret_list



def eval_once(options):
    global features_matrix, net_origin, SAMPLE_NODES, SAMPLE_RULE, METIRC, PREC_K
    if not utils.check_rebuild(options.reconstruction_path, descrip='reconstruction', always_rebuild=options.always_rebuild):
        return

    logger.info('eval case: network reconstruction...')
    logger.info('\t save_path: {}'.format(options.reconstruction_path))
    logger.info('\t data_path: {}'.format(options.data_path))
    logger.info('\t data_format: {}'.format(options.data_format))
    logger.info('\t metrics: MAP and precise@K')
    logger.info('\t max_index for precise@K: {}'.format(options.precK_max_index))
    logger.info('\t similarity_metric: {}'.format(options.similarity_metric))
    logger.info('\t eval_online: {}'.format(options.eval_online))
    logger.info('\t eval_interval: {}s'.format(options.eval_interval))
    logger.info('\t sample_nodes: {}'.format(options.sample_nodes))
    logger.info('\t sample_nodes_rule: {}'.format(options.sample_nodes_rule))
    logger.info('\t repeat {} times'.format(options.repeated_times))
    logger.info('\t eval_workers: {}'.format(options.eval_workers))

    # set the whole network
    logger.info("constructing origin network....")
    net_origin = network.construct_network(data_path = options.data_path, data_format = options.data_format,
                                           print_net_info = False, isdirected = options.isdirected)
    origin_nodes_size = net_origin.get_nodes_size()
    logger.info("origin_nodes_size = {}".format(origin_nodes_size))
    # loading features_matrix(already trained)
    logger.info('\t reading embedding vectors from file {}'.format(options.vectors_path))
    time_start = time.time()
    id_list = list(range(origin_nodes_size)) # must be [0,1,2,3,...]
    SAMPLE_NODES = options.sample_nodes
    SAMPLE_RULE = options.sample_nodes_rule
    METIRC = options.similarity_metric
    PREC_K = options.precK_max_index
    features_matrix = utils.get_vectors(utils.get_KeyedVectors(options.vectors_path), id_list)
    logger.info('\t reading embedding vectors completed in {}s'.format(time.time() - time_start))
    logger.info('total loaded nodes: {}'.format(np.size(features_matrix, axis=0)))
    logger.info('the embedding dimension: {}'.format(np.size(features_matrix, axis=1)))

    fr = open(options.reconstruction_path, 'w')
    fr.write('eval case: network reconstruction ...\n')
    fr.write('\t save_path: {}\n'.format(options.reconstruction_path))
    fr.write('\t data_path: {}\n'.format(options.data_path))
    fr.write('\t data_format: {}\n'.format(options.data_format))
    fr.write('\t metrics: MAP and precise@K\n')
    fr.write('\t max_index for precise@K: {}\n'.format(options.precK_max_index))
    fr.write('\t similarity_metric: {}\n'.format(options.similarity_metric))
    fr.write('\t eval_online: {}\n'.format(options.eval_online))
    fr.write('\t eval_interval: {}s\n'.format(options.eval_interval))
    fr.write('\t sample_nodes: {}\n'.format(options.sample_nodes))
    fr.write('\t sample_nodes_rule: {}\n'.format(options.sample_nodes_rule))
    fr.write('\t repeat {} times\n'.format(options.repeated_times))
    fr.write('\t eval_workers: {}\n'.format(options.eval_workers))
    fr.write("origin_nodes_size = {}\n".format(origin_nodes_size))
    fr.write('total loaded nodes: {}\n'.format(np.size(features_matrix, axis=0)))
    fr.write('the embedding dimension: {}\n'.format(np.size(features_matrix, axis=1)))


    if options.sample_nodes > 0:
        if options.eval_workers > 1 and options.repeated_times > 1:
            # speed up by using multi-process
            logger.info("\t allocating repeat_times to workers ...")
            if options.repeated_times <= options.eval_workers:
                times_per_worker = [1 for _ in range(options.repeated_times)]
            else:
                div, mod = divmod(options.repeated_times, options.eval_workers)
                times_per_worker = [div for _ in range(options.eval_workers)]
                for idx in range(mod):
                    times_per_worker[idx] = times_per_worker[idx] + 1
            assert sum(times_per_worker) == options.repeated_times, 'workers allocating failed: %d != %d' % (
                sum(times_per_worker), options.repeated_times)

            logger.info("\t using {} processes for evaling:".format(len(times_per_worker)))
            for idx, rep_times in enumerate(times_per_worker):
                logger.info("\t process-{}: repeat {} times".format(idx, rep_times))

            ret_list = [] # [[MAP, precisionK_list], ... ]
            with ProcessPoolExecutor(max_workers=options.eval_workers) as executor:
                for ret in executor.map(_sample_thread_body, times_per_worker):
                    ret_list.extend(ret)
            if len(ret_list) != options.repeated_times:
                logger.warning("warning: eval unmatched repeated_times: {} != {}".format(len(ret_list) , options.repeated_times))
        else:
            ret_list = _sample_thread_body(options.repeated_times)
    else:
        # no sampling, no repeat!
        ret_list = [_eval(net_origin)] # [[MAP, precisionK_list]]

    if options.sample_nodes > 0:
        fr.write('expected repeated_times: {}, actual repeated_times: {}, mean results as follows:\n'.format(
            options.repeated_times, len(ret_list)))
    else:
        fr.write('due to the sample nodes = {}, so actual repeated_times = {}, results as follows:\n'.format(
            options.sample_nodes, len(ret_list)))

    mean_MAP = np.mean([ret[0] for ret in ret_list])
    mean_precisionK = np.mean([ret[1] for ret in ret_list], axis=0)

    fr.write('\t\t MAP = {}\n'.format(mean_MAP))
    for k in range(options.precK_max_index):
        if k < len(mean_precisionK):
            fr.write('\t\t precisionK_{} = {}\n'.format(k+1, mean_precisionK[k]))
        else:
            fr.write('\t\t precisionK_{} = None\n'.format(k + 1))
    fr.write('details:\n')
    for repeat in range(len(ret_list)):
        fr.write('\t repeated {}/{}:\n'.format(repeat + 1, len(ret_list)))
        MAP = ret_list[repeat][0]
        precisionK_list = ret_list[repeat][1]
        fr.write('\t\t MAP = {}\n'.format(MAP))
        for k in range(options.precK_max_index):
            if k < len(precisionK_list):
                fr.write('\t\t precisionK_{} = {}\n'.format(k + 1, precisionK_list[k]))
            else:
                fr.write('\t\t precisionK_{} = None\n'.format(k + 1))

    fr.write('\neval case: network reconstruction completed in {}s.'.format(time.time() - time_start))
    fr.close()
    logger.info('eval case: network reconstruction completed in {}s.'.format(time.time() - time_start))

    return

def eval_online(options):
    global features_matrix, net_origin, SAMPLE_NODES, SAMPLE_RULE, METIRC, PREC_K
    reconstruction_dir = os.path.split(options.reconstruction_path)[0]
    if not utils.check_rebuild(reconstruction_dir, descrip='reconstruction', always_rebuild=options.always_rebuild):
        return
    if not os.path.exists(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    logger.info('eval case: network reconstruction...')
    logger.info('\t save_path: {}'.format(options.reconstruction_path))
    logger.info('\t metrics: MAP and precise@K')
    logger.info('\t max_index for precise@K: {}'.format(options.precK_max_index))
    logger.info('\t similarity_metric: {}'.format(options.similarity_metric))
    logger.info('\t eval_online: {}'.format(options.eval_online))
    logger.info('\t eval_interval: {}s'.format(options.eval_interval))
    logger.info('\t sample_nodes: {}'.format(options.sample_nodes))
    logger.info('\t sample_nodes_rule: {}'.format(options.sample_nodes_rule))
    logger.info('\t repeat {} times'.format(options.repeated_times))
    logger.info('\t eval_workers: {}'.format(options.eval_workers))


    # set the whole network
    logger.info("constructing origin network....")
    net_origin = network.construct_network(data_path=options.data_path, data_format=options.data_format,
                                           print_net_info=False, isdirected=options.isdirected)
    origin_nodes_size = net_origin.get_nodes_size()
    logger.info("origin_nodes_size = {}".format(origin_nodes_size))
    id_list = list(range(origin_nodes_size))  # must be [0,1,2,3,...]
    SAMPLE_NODES = options.sample_nodes
    SAMPLE_RULE = options.sample_nodes_rule
    METIRC = options.similarity_metric
    PREC_K = options.precK_max_index

    metric_prec_k_list = [1]
    decimal_number = 10
    while metric_prec_k_list[-1] < options.precK_max_index:
        if decimal_number <= options.precK_max_index:
            metric_prec_k_list.append(decimal_number)
        else:
            break
        if 2*decimal_number <= options.precK_max_index:
            metric_prec_k_list.append(2*decimal_number)
        else:
            break
        if 5*decimal_number <= options.precK_max_index:
            metric_prec_k_list.append(5*decimal_number)
        else:
            break
        decimal_number = decimal_number*10


    if options.sample_nodes > 0:
        if options.eval_workers > 1 and options.repeated_times > 1:
            # speed up by using multi-process
            logger.info("\t allocating repeat_times to workers ...")
            if options.repeated_times <= options.eval_workers:
                times_per_worker = [1 for _ in range(options.repeated_times)]
            else:
                div, mod = divmod(options.repeated_times, options.eval_workers)
                times_per_worker = [div for _ in range(options.eval_workers)]
                for idx in range(mod):
                    times_per_worker[idx] = times_per_worker[idx] + 1
            assert sum(times_per_worker) == options.repeated_times, 'workers allocating failed: %d != %d' % (
                sum(times_per_worker), options.repeated_times)

            logger.info("\t using {} processes for evaling:".format(len(times_per_worker)))
            for idx, rep_times in enumerate(times_per_worker):
                logger.info("\t process-{}: repeat {} times".format(idx, rep_times))

    fr_total = open(options.reconstruction_path, 'w')
    fr_total.write('eval case: network reconstruction ...\n')
    fr_total.write('\t save_path: {}\n'.format(options.reconstruction_path))
    fr_total.write('\t metrics: MAP and precise@K\n')
    fr_total.write('\t max_index for precise@K: {}\n'.format(options.precK_max_index))
    fr_total.write('\t similarity_metric: {}\n'.format(options.similarity_metric))
    fr_total.write('\t eval_online: {}\n'.format(options.eval_online))
    fr_total.write('\t eval_interval: {}s\n'.format(options.eval_interval))
    fr_total.write('\t sample_nodes: {}\n'.format(options.sample_nodes))
    fr_total.write('\t sample_nodes_rule: {}\n'.format(options.sample_nodes_rule))
    fr_total.write('\t repeat {} times\n'.format(options.repeated_times))
    fr_total.write('\t eval_workers: {}\n'.format(options.eval_workers))
    fr_total.write("\t origin_nodes_size = {}\n".format(origin_nodes_size))
    fr_total.write('\t results:\n=============================================================\n')
    fr_total.write('finish_time\tckpt\tMAP\t')
    for v in metric_prec_k_list:
        fr_total.write('\tPr@{}'.format(v))
    fr_total.write("\n")


    last_step = 0
    summary_writer = tf.summary.FileWriter(reconstruction_dir, tf.Graph())
    summary = tf.Summary()
    summary.value.add(tag='MAP', simple_value=0.)
    for v in metric_prec_k_list:
        summary.value.add(tag='Pr_{}'.format(v), simple_value=0.)
    summary_writer.add_summary(summary, last_step)

    best_MAP = 0

    ckpt_dir = os.path.join(os.path.split(options.vectors_path)[0], 'ckpt')
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    while (not (ckpt and ckpt.model_checkpoint_path)):
        logger.info("\t model and vectors not exist, waiting ...")
        time.sleep(options.eval_interval)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    reading = options.vectors_path + ".reading_reconstruction"
    writing = options.vectors_path + ".writing"
    while (options.eval_online):
        while True:
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            cur_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            if cur_step <= last_step or (not os.path.exists(options.vectors_path)) or os.path.exists(writing):
                if os.path.exists(os.path.join(os.path.split(options.vectors_path)[0], "RUN_SUCCESS")):
                    return
                time.sleep(options.eval_interval)
                continue
            # ready for reading
            logger.info("\t declare for reading ...")
            open(reading, "w")  # declare
            time.sleep(30)
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            cur_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            if cur_step <= last_step or (not os.path.exists(options.vectors_path)) or os.path.exists(writing):
                os.remove(reading)  # undeclare
                logger.info("\t confliction! undeclare and waiting ...")
                time.sleep(options.eval_interval)
                continue

            break
        logger.info("\t eval ckpt-{}.......".format(cur_step))
        # loading features_matrix(already trained)
        logger.info('\t reading embedding vectors from file {}'.format(options.vectors_path))
        time_start = time.time()
        features_matrix = utils.get_vectors(utils.get_KeyedVectors(options.vectors_path), id_list)
        os.remove(reading)
        logger.info("\t done for reading ...")
        logger.info('\t reading embedding vectors completed in {}s'.format(time.time() - time_start))
        logger.info('total loaded nodes: {}'.format(np.size(features_matrix, axis=0)))
        logger.info('the embedding dimension: {}'.format(np.size(features_matrix, axis=1)))

        # reconstruction
        fr = open(options.reconstruction_path + '.{}'.format(cur_step), 'w')
        fr.write('eval case: network reconstruction ...\n')
        fr.write('\t metrics: MAP and precise@K\n')
        fr.write('\t max_index for precise@K: {}\n'.format(options.precK_max_index))
        fr.write('\t similarity_metric: {}\n'.format(options.similarity_metric))
        fr.write('\t eval_online: {}\n'.format(options.eval_online))
        fr.write('\t eval_interval: {}s\n'.format(options.eval_interval))
        fr.write('\t sample_nodes: {}\n'.format(options.sample_nodes))
        fr.write('\t sample_nodes_rule: {}\n'.format(options.sample_nodes_rule))
        fr.write('\t repeat {} times\n'.format(options.repeated_times))
        fr.write('\t eval_workers: {}\n'.format(options.eval_workers))
        fr.write("\t origin_nodes_size = {}\n".format(origin_nodes_size))
        fr.write('\t total loaded nodes: {}\n'.format(np.size(features_matrix, axis=0)))
        fr.write('\t the embedding dimension: {}\n'.format(np.size(features_matrix, axis=1)))

        if options.sample_nodes > 0:
            if options.eval_workers > 1 and options.repeated_times > 1:
                # speed up by using multi-process
                ret_list = [] # [[MAP, precisionK_list], ... ]
                with ProcessPoolExecutor(max_workers=options.eval_workers) as executor:
                    for ret in executor.map(_sample_thread_body, times_per_worker):
                        ret_list.extend(ret)
                if len(ret_list) != options.repeated_times:
                    logger.warning("warning: eval unmatched repeated_times: {} != {}".format(len(ret_list) , options.repeated_times))
            else:
                ret_list = _sample_thread_body(options.repeated_times)
        else:
            # no sampling, no repeat!
            ret_list = [_eval(net_origin)] # [[MAP, precisionK_list]]

        fr_total.write('%s ckpt-%-9d: ' % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), cur_step))
        summary = tf.Summary()

        if options.sample_nodes > 0:
            fr.write('expected repeated_times: {}, actual repeated_times: {}, mean results as follows:\n'.format(
                options.repeated_times, len(ret_list)))
        else:
            fr.write('due to the sample nodes = {}, so actual repeated_times = {}, results as follows:\n'.format(
                options.sample_nodes, len(ret_list)))

        mean_MAP = np.mean([ret[0] for ret in ret_list])
        mean_precisionK = np.mean([ret[1] for ret in ret_list], axis=0)

        fr.write('\t\t MAP = {}\n'.format(mean_MAP))
        for k in range(options.precK_max_index):
            if k < len(mean_precisionK):
                fr.write('\t\t precisionK_{} = {}\n'.format(k + 1, mean_precisionK[k]))
            else:
                fr.write('\t\t precisionK_{} = None\n'.format(k + 1))
        fr.write('details:\n')
        for repeat in range(len(ret_list)):
            fr.write('\t repeated {}/{}:\n'.format(repeat + 1, len(ret_list)))
            MAP = ret_list[repeat][0]
            precisionK_list = ret_list[repeat][1]
            fr.write('\t\t MAP = {}\n'.format(MAP))
            for k in range(options.precK_max_index):
                if k < len(precisionK_list):
                    fr.write('\t\t precisionK_{} = {}\n'.format(k + 1, precisionK_list[k]))
                else:
                    fr.write('\t\t precisionK_{} = None\n'.format(k + 1))

        fr.write('\neval case: network reconstruction completed in {}s.'.format(time.time() - time_start))
        fr.close()

        fr_total.write('%.4f' % mean_MAP)
        summary.value.add(tag='MAP', simple_value=mean_MAP)
        for v in metric_prec_k_list:
            fr_total.write('\t%.4f'%mean_precisionK[v-1])
            summary.value.add(tag='Pr_{}'.format(v), simple_value=mean_precisionK[v-1])
        fr_total.write("\n")
        fr_total.flush()
        summary_writer.add_summary(summary, cur_step)
        summary_writer.flush()
        logger.info('eval case: network reconstruction completed in {}s.\n================================='.format(time.time() - time_start))

        # copy ckpt-files according to last mean_Micro_F1 (0.9 ratio).
        if mean_MAP > best_MAP:
            best_MAP = mean_MAP

            ckptIsExists = os.path.exists(os.path.join(ckpt_dir, 'model.ckpt-%d.index' % cur_step))
            if ckptIsExists:
                fr_best = open(os.path.join(reconstruction_dir, 'best_ckpt.info'), 'w')
            else:
                fr_best = open(os.path.join(reconstruction_dir, 'best_ckpt.info'), 'a')
                fr_best.write("Note:the model.ckpt-best is the remainings of last best_ckpt!\n"
                              "the current best_ckpt model is loss, but the result is:\n")
            fr_best.write("best_MAP: {}\n".format(best_MAP))
            fr_best.write("best_ckpt: ckpt-{}\n".format(cur_step))
            fr_best.close()

            if ckptIsExists:
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.data-00000-of-00001' % cur_step)
                targetFile = os.path.join(reconstruction_dir, 'model.ckpt-best.data-00000-of-00001')
                if os.path.exists(targetFile):
                    os.remove(targetFile)
                shutil.copy(sourceFile, targetFile)
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.index' % cur_step)
                targetFile = os.path.join(reconstruction_dir, 'model.ckpt-best.index')
                if os.path.exists(targetFile):
                    os.remove(targetFile)
                shutil.copy(sourceFile, targetFile)
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.meta' % cur_step)
                targetFile = os.path.join(reconstruction_dir, 'model.ckpt-best.meta')
                if os.path.exists(targetFile):
                    os.remove(targetFile)
                shutil.copy(sourceFile, targetFile)

        last_step = cur_step

    fr_total.close()
    summary_writer.close()



def eval(options):
    if(options.eval_online):
        eval_online(options)
    else:
        eval_once(options)