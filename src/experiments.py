#!/usr/bin/env python3

"""
The script also includes various 
command-line arguments to control the experiment settings, 
such as the data directory, model architecture, hyperparameters, and evaluation options.
"""

import copy
import itertools
import numpy as np
import os, sys
import random

import torch

from src.parse_args import parser
from src.parse_args import args
import src.data_utils as data_utils
import src.eval
from src.knowledge_graph import KnowledgeGraph
from src.emb.fact_network import ComplEx, ConvE, DistMult, PTransE, TransE, TuckER
from src.emb.fact_network import get_conve_kg_state_dict, get_complex_kg_state_dict, get_distmult_kg_state_dict, get_tucker_kg_state_dict
from src.emb.emb import EmbeddingBasedMethod
from src.rl.graph_search.pn import GraphSearchPolicy
from src.rl.graph_search.pg import PolicyGradient
from src.rl.graph_search.rs_pg import RewardShapingPolicyGradient
from src.utils.ops import flatten
import argparse
import os

torch.cuda.set_device(args.gpu)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)



def process_data():
    data_dir = args.data_dir
    raw_kb_path = os.path.join(data_dir, 'raw.kb')
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    data_utils.prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, args.test, args.add_reverse_relations)

def initialize_model_directory(args, random_seed=None):
    # add model parameter info to model directory
    model_root_dir = args.model_root_dir
    dataset = os.path.basename(os.path.normpath(args.data_dir))

    reverse_edge_tag = '-RV' if args.add_reversed_training_edges else ''
    entire_graph_tag = '-EG' if args.train_entire_graph else ''
    if args.xavier_initialization:
        initialization_tag = '-xavier'
    elif args.uniform_entity_initialization:
        initialization_tag = '-uniform'
    else:
        initialization_tag = ''

    # Hyperparameter signature
    if args.model in ['rule']:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.baseline,
            args.entity_dim,
            args.relation_dim,
            args.history_num_layers,
            args.learning_rate,
            args.emb_dropout_rate,
            args.ff_dropout_rate,
            args.action_dropout_rate,
            args.bandwidth,
            args.beta
        )
    elif args.model.startswith('point'):
        if args.baseline == 'avg_reward':
            print('* Policy Gradient Baseline: average reward')
        elif args.baseline == 'avg_reward_normalized':
            print('* Policy Gradient Baseline: average reward baseline plus normalization')
        else:
            print('* Policy Gradient Baseline: None')
        if args.action_dropout_anneal_interval < 1000:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.baseline,
                args.entity_dim,
                args.relation_dim,
                args.history_num_layers,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.action_dropout_anneal_factor,
                args.action_dropout_anneal_interval,
                args.bandwidth,
                args.beta,
                args.use_action_selection,
                args.use_state_prediction,
                args.mask_sim_relation,
                args.max_dynamic_action_size,
                args.dynamic_split_bound,
                args.avg_entity_per_relation
            )
            if args.mu != 1.0:
                hyperparam_sig += '-{}'.format(args.mu)
        else:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.baseline,
                args.entity_dim,
                args.relation_dim,
                args.history_num_layers,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.bandwidth,
                args.beta,
                args.use_action_selection,
                args.use_state_prediction,
                args.mask_sim_relation,
                args.max_dynamic_action_size,
                args.dynamic_split_bound,
                args.avg_entity_per_relation
            )
        if args.reward_shaping_threshold > 0:
            hyperparam_sig += '-{}'.format(args.reward_shaping_threshold)
    elif args.model == 'distmult' or args.model == 'TransE' or args.model == 'PTransE':
        hyperparam_sig = '{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.emb_dropout_rate,
            args.label_smoothing_epsilon
        )
    elif args.model == 'complex':
        hyperparam_sig = '{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.emb_dropout_rate,
            args.label_smoothing_epsilon
        )
    elif args.model in ['conve', 'hypere', 'triplee', 'tucker']:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.num_out_channels,
            args.kernel_size,
            args.emb_dropout_rate,
            args.hidden_dropout_rate,
            args.feat_dropout_rate,
            args.label_smoothing_epsilon
        )
    else:
        raise NotImplementedError

    model_sub_dir = '{}-{}{}{}{}-{}'.format(
        dataset,
        args.model,
        reverse_edge_tag,
        entire_graph_tag,
        initialization_tag,
        hyperparam_sig
    )
    if args.model == 'set':
        model_sub_dir += '-{}'.format(args.beam_size)
        model_sub_dir += '-{}'.format(args.num_paths_per_entity)
    if args.relation_only:
        model_sub_dir += '-ro'
    elif args.relation_only_in_path:
        model_sub_dir += '-rpo'
    elif args.type_only:
        model_sub_dir += '-to'

    if args.test:
        model_sub_dir += '-test'

    if random_seed:
        model_sub_dir += '.{}'.format(random_seed)

    model_dir = os.path.join(model_root_dir, model_sub_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print('Model directory created: {}'.format(model_dir))
    else:
        print('Model directory exists: {}'.format(model_dir))

    args.model_dir = model_dir

def construct_model(args):
    """
    Construct NN graph.
    """
    kg = KnowledgeGraph(args)
    if args.model.endswith('.gc'):
        kg.load_fuzzy_facts()

    if args.model in ['point', 'point.gc']:
        pn = GraphSearchPolicy(args)
        lf = PolicyGradient(args, kg, pn)
    elif args.model.startswith('point.rs'):
        fn_model = args.model.split('.')[2]
        fn_args = copy.deepcopy(args)
        fn_args.model = fn_model
        fn_args.relation_only = False
        if fn_model == 'complex':
            fn = ComplEx(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'distmult':
            fn = DistMult(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'conve':
            fn = ConvE(fn_args, kg.num_entities)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'tucker':
            fn = TuckER(fn_args, kg.num_entities)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'PTransE':
            fn = PTransE(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        pn = GraphSearchPolicy(args, fn_kg=fn_kg, fn=fn)
        lf = RewardShapingPolicyGradient(args, kg, pn, fn_kg, fn)
    elif args.model == 'complex':
        fn = ComplEx(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'distmult':
        fn = DistMult(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'tucker':
        fn = TuckER(args, kg.num_entities)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'conve':
        fn = ConvE(args, kg.num_entities)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'TransE':
        fn = TransE(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'PTransE':
        fn = PTransE(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    else:
        raise NotImplementedError
    return lf

def train(lf):
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    if lf.model.startswith('point'):
        train_data = data_utils.load_triples(
        train_path, entity_index_path, relation_index_path, group_examples_by_query=args.group_examples_by_query,
        add_reverse_relations=args.add_reversed_training_edges, skip_sim=True)
    else:
        train_data = data_utils.load_triples(
        train_path, entity_index_path, relation_index_path, group_examples_by_query=args.group_examples_by_query,
        add_reverse_relations=args.add_reversed_training_edges)
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
    if args.checkpoint_path is not None:
        lf.load_checkpoint(args.checkpoint_path)
    lf.run_train(train_data, dev_data)

def inference(lf):
    lf.batch_size = args.dev_batch_size
    lf.eval()
    if args.model == 'hypere':
        conve_kg_state_dict = get_conve_kg_state_dict(torch.load(args.conve_state_dict_path))
        lf.kg.load_state_dict(conve_kg_state_dict)
        secondary_kg_state_dict = get_complex_kg_state_dict(torch.load(args.complex_state_dict_path))
        lf.secondary_kg.load_state_dict(secondary_kg_state_dict)
    elif args.model == 'triplee':
        conve_kg_state_dict = get_conve_kg_state_dict(torch.load(args.conve_state_dict_path))
        lf.kg.load_state_dict(conve_kg_state_dict)
        complex_kg_state_dict = get_complex_kg_state_dict(torch.load(args.complex_state_dict_path))
        lf.secondary_kg.load_state_dict(complex_kg_state_dict)
        distmult_kg_state_dict = get_distmult_kg_state_dict(torch.load(args.distmult_state_dict_path))
        lf.tertiary_kg.load_state_dict(distmult_kg_state_dict)
    else:
        lf.load_checkpoint(get_checkpoint_path(args))
    print(lf.kg.entity_embeddings)
    print(lf.kg.relation_embeddings)
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()

    eval_metrics = {
        'dev': {},
        'test': {}
    }

    if args.compute_map:
        relation_sets = [
            'concept:athletehomestadium',
            'concept:athleteplaysforteam',
            'concept:athleteplaysinleague',
            'concept:athleteplayssport',
            'concept:organizationheadquarteredincity',
            'concept:organizationhiredperson',
            'concept:personborninlocation',
            'concept:teamplayssport',
            'concept:worksfor'
        ]
        mps = []
        for r in relation_sets:
            print('* relation: {}'.format(r))
            test_path = os.path.join(args.data_dir, 'tasks', r, 'test.pairs')
            test_data, labels = data_utils.load_triples_with_label(
                test_path, r, entity_index_path, relation_index_path, seen_entities=seen_entities)
            pred_scores = lf.forward(test_data, verbose=False)
            mp = src.eval.link_MAP(test_data, pred_scores, labels, lf.kg.all_objects, verbose=True)
            mps.append(mp)
        import numpy as np
        map_ = np.mean(mps)
        print('Overall MAP = {}'.format(map_))
        eval_metrics['test']['avg_map'] = map
    elif args.eval_by_relation_type:
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
        pred_scores = lf.forward(dev_data, verbose=False)
        to_m_rels, to_1_rels, _ = data_utils.get_relations_by_type(args.data_dir, relation_index_path)
        relation_by_types = (to_m_rels, to_1_rels)
        print('Dev set evaluation by relation type (partial graph)')
        src.eval.hits_and_ranks_by_relation_type(
            dev_data, pred_scores, lf.kg.dev_objects, relation_by_types, verbose=True)
        print('Dev set evaluation by relation type (full graph)')
        src.eval.hits_and_ranks_by_relation_type(
            dev_data, pred_scores, lf.kg.all_objects, relation_by_types, verbose=True)
    elif args.eval_by_seen_queries:
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
        pred_scores = lf.forward(dev_data, verbose=False)
        seen_queries = data_utils.get_seen_queries(args.data_dir, entity_index_path, relation_index_path)
        print('Dev set evaluation by seen queries (partial graph)')
        src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.dev_objects, seen_queries, verbose=True)
        print('Dev set evaluation by seen queries (full graph)')
        src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.all_objects, seen_queries, verbose=True)
    else:
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        test_path = os.path.join(args.data_dir, 'test.triples')
        dev_data = data_utils.load_triples(
            dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        # _, test_data = data_utils.load_triples(
        #     test_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False, few_shot=True, lf=lf)
        test_data = data_utils.load_triples(
            test_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        # print('Dev set performance:')
        # pred_scores = lf.forward(dev_data, verbose=False)
        # dev_metrics = src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.dev_objects, verbose=True)
        # eval_metrics['dev'] = {}
        # eval_metrics['dev']['hits_at_1'] = dev_metrics[0]
        # eval_metrics['dev']['hits_at_3'] = dev_metrics[1]
        # eval_metrics['dev']['hits_at_5'] = dev_metrics[2]
        # eval_metrics['dev']['hits_at_10'] = dev_metrics[3]
        # eval_metrics['dev']['mrr'] = dev_metrics[4]
        # src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.all_objects, verbose=True)
        print('Test set performance:')
        pred_scores = lf.forward(test_data, verbose=False)
        test_metrics = src.eval.hits_and_ranks(test_data, pred_scores, lf.kg.all_objects, verbose=True, output=False, kg=lf.kg, model_name=args.model, split_relation=False)
        eval_metrics['dev'] = {}
        eval_metrics['dev']['hits_at_1'] = test_metrics[0]
        eval_metrics['dev']['hits_at_3'] = test_metrics[1]
        eval_metrics['dev']['hits_at_5'] = test_metrics[2]
        eval_metrics['dev']['hits_at_10'] = test_metrics[3]
        eval_metrics['dev']['mrr'] = test_metrics[4]
        # num = 0; hits_10 = 0.0; hits_1 = 0.0; hits_3 = 0.0; hits_5 = 0.0
        # for x in test_data:
        #     print('Test set of relation {} performance:'.format(x))
        #     pred_scores = lf.forward(test_data[x], verbose=False)
        #     test_metrics = src.eval.hits_and_ranks(test_data[x], pred_scores, lf.kg.all_objects, verbose=True)
        #     eval_metrics['test']['hits_at_1'] = test_metrics[0]
        #     eval_metrics['test']['hits_at_3'] = test_metrics[1]
        #     eval_metrics['test']['hits_at_5'] = test_metrics[2]
        #     eval_metrics['test']['hits_at_10'] = test_metrics[3]
        #     eval_metrics['test']['mrr'] = test_metrics[4]
        #     num += len(test_data[x])
        #     hits_1 += float(test_metrics[0]) * len(test_data[x])
        #     hits_3 += float(test_metrics[1]) * len(test_data[x])
        #     hits_5 += float(test_metrics[2]) * len(test_data[x])
        #     hits_10 += float(test_metrics[3]) * len(test_data[x])
        # print('Hits@1 = {}'.format(hits_1 / num))
        # print('Hits@3 = {}'.format(hits_3 / num))
        # print('Hits@5 = {}'.format(hits_5 / num))
        # print('Hits@10 = {}'.format(hits_10 / num))

    return eval_metrics

def run_ablation_studies(args):
    """
    Run the ablation study experiments reported in the paper.
    """
    def set_up_lf_for_inference(args):
        initialize_model_directory(args)
        lf = construct_model(args)
        lf.cuda()
        lf.batch_size = args.dev_batch_size
        lf.load_checkpoint(get_checkpoint_path(args))
        lf.eval()
        return lf

    def rel_change(metrics, ab_system, kg_portion):
        ab_system_metrics = metrics[ab_system][kg_portion]
        base_metrics = metrics['ours'][kg_portion]
        return int(np.round((ab_system_metrics - base_metrics) / base_metrics * 100))

    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()
    dataset = os.path.basename(args.data_dir)
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    dev_data = data_utils.load_triples(
        dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
    to_m_rels, to_1_rels, (to_m_ratio, to_1_ratio) = data_utils.get_relations_by_type(args.data_dir, relation_index_path)
    relation_by_types = (to_m_rels, to_1_rels)
    to_m_ratio *= 100
    to_1_ratio *= 100
    seen_queries, (seen_ratio, unseen_ratio) = data_utils.get_seen_queries(args.data_dir, entity_index_path, relation_index_path)
    seen_ratio *= 100
    unseen_ratio *= 100

    systems = ['ours', '-ad', '-rs']
    mrrs, to_m_mrrs, to_1_mrrs, seen_mrrs, unseen_mrrs = {}, {}, {}, {}, {}
    for system in systems:
        print('** Evaluating {} system **'.format(system))
        if system == '-ad':
            args.action_dropout_rate = 0.0
            if dataset == 'umls':
                # adjust dropout hyperparameters
                args.emb_dropout_rate = 0.3
                args.ff_dropout_rate = 0.1
        elif system == '-rs':
            config_path = os.path.join('configs', '{}.sh'.format(dataset.lower()))
            args = parser.parse_args()
            args = data_utils.load_configs(args, config_path)
        
        lf = set_up_lf_for_inference(args)
        pred_scores = lf.forward(dev_data, verbose=False)
        _, _, _, _, mrr = src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.dev_objects, verbose=True)
        if to_1_ratio == 0:
            to_m_mrr = mrr
            to_1_mrr = -1
        else:
            to_m_mrr, to_1_mrr = src.eval.hits_and_ranks_by_relation_type(
                dev_data, pred_scores, lf.kg.dev_objects, relation_by_types, verbose=True)
        seen_mrr, unseen_mrr = src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.dev_objects, seen_queries, verbose=True)
        mrrs[system] = {'': mrr * 100}
        to_m_mrrs[system] = {'': to_m_mrr * 100}
        to_1_mrrs[system] = {'': to_1_mrr  * 100}
        seen_mrrs[system] = {'': seen_mrr * 100}
        unseen_mrrs[system] = {'': unseen_mrr * 100}
        _, _, _, _, mrr_full_kg = src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.all_objects, verbose=True)
        if to_1_ratio == 0:
            to_m_mrr_full_kg = mrr_full_kg
            to_1_mrr_full_kg = -1
        else:
            to_m_mrr_full_kg, to_1_mrr_full_kg = src.eval.hits_and_ranks_by_relation_type(
                dev_data, pred_scores, lf.kg.all_objects, relation_by_types, verbose=True)
        seen_mrr_full_kg, unseen_mrr_full_kg = src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.all_objects, seen_queries, verbose=True)
        mrrs[system]['full_kg'] = mrr_full_kg * 100
        to_m_mrrs[system]['full_kg'] = to_m_mrr_full_kg * 100
        to_1_mrrs[system]['full_kg'] = to_1_mrr_full_kg * 100
        seen_mrrs[system]['full_kg'] = seen_mrr_full_kg * 100
        unseen_mrrs[system]['full_kg'] = unseen_mrr_full_kg * 100

    # overall system comparison (table 3)
    print('Partial graph evaluation')
    print('--------------------------')
    print('Overall system performance')
    print('Ours(ConvE)\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f}'.format(mrrs['ours'][''], mrrs['-rs'][''], mrrs['-ad']['']))
    print('--------------------------')
    # performance w.r.t. relation types (table 4, 6)
    print('Performance w.r.t. relation types')
    print('\tTo-many\t\t\t\tTo-one\t\t')
    print('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        to_m_ratio, to_m_mrrs['ours'][''], to_m_mrrs['-rs'][''], rel_change(to_m_mrrs, '-rs', ''), to_m_mrrs['-ad'][''], rel_change(to_m_mrrs, '-ad', ''),
        to_1_ratio, to_1_mrrs['ours'][''], to_1_mrrs['-rs'][''], rel_change(to_1_mrrs, '-rs', ''), to_1_mrrs['-ad'][''], rel_change(to_1_mrrs, '-ad', '')))
    print('--------------------------')
    # performance w.r.t. seen queries (table 5, 7)
    print('Performance w.r.t. seen/unseen queries')
    print('\tSeen\t\t\t\tUnseen\t\t')
    print('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        seen_ratio, seen_mrrs['ours'][''], seen_mrrs['-rs'][''], rel_change(seen_mrrs, '-rs', ''), seen_mrrs['-ad'][''], rel_change(seen_mrrs, '-ad', ''),
        unseen_ratio, unseen_mrrs['ours'][''], unseen_mrrs['-rs'][''], rel_change(unseen_mrrs, '-rs', ''), unseen_mrrs['-ad'][''], rel_change(unseen_mrrs, '-ad', '')))
    print()
    print('Full graph evaluation')
    print('--------------------------')
    print('Overall system performance')
    print('Ours(ConvE)\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f}'.format(mrrs['ours']['full_kg'], mrrs['-rs']['full_kg'], mrrs['-ad']['full_kg']))
    print('--------------------------')
    print('Performance w.r.t. relation types')
    print('\tTo-many\t\t\t\tTo-one\t\t')
    print('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        to_m_ratio, to_m_mrrs['ours']['full_kg'], to_m_mrrs['-rs']['full_kg'], rel_change(to_m_mrrs, '-rs', 'full_kg'), to_m_mrrs['-ad']['full_kg'], rel_change(to_m_mrrs, '-ad', 'full_kg'),
        to_1_ratio, to_1_mrrs['ours']['full_kg'], to_1_mrrs['-rs']['full_kg'], rel_change(to_1_mrrs, '-rs', 'full_kg'), to_1_mrrs['-ad']['full_kg'], rel_change(to_1_mrrs, '-ad', 'full_kg')))
    print('--------------------------')
    print('Performance w.r.t. seen/unseen queries')
    print('\tSeen\t\t\t\tUnseen\t\t')
    print('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        seen_ratio, seen_mrrs['ours']['full_kg'], seen_mrrs['-rs']['full_kg'], rel_change(seen_mrrs, '-rs', 'full_kg'), seen_mrrs['-ad']['full_kg'], rel_change(seen_mrrs, '-ad', 'full_kg'),
        unseen_ratio, unseen_mrrs['ours']['full_kg'], unseen_mrrs['-rs']['full_kg'], rel_change(unseen_mrrs, '-rs', 'full_kg'), unseen_mrrs['-ad']['full_kg'], rel_change(unseen_mrrs, '-ad', 'full_kg')))

def export_to_embedding_projector(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_to_embedding_projector()

def export_reward_shaping_parameters(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_reward_shaping_parameters()

def export_fuzzy_facts(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_fuzzy_facts()

def export_error_cases(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.batch_size = args.dev_batch_size
    lf.eval()
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path)
    lf.load_checkpoint(get_checkpoint_path(args))
    print('Dev set performance:')
    pred_scores = lf.forward(dev_data, verbose=False)
    src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.dev_objects, verbose=True)
    src.eval.export_error_cases(dev_data, pred_scores, lf.kg.dev_objects, os.path.join(lf.model_dir, 'error_cases.pkl'))

def compute_fact_scores(lf):
    data_dir = args.data_dir
    train_path = os.path.join(data_dir, 'train.triples')
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    train_data = data_utils.load_triples(train_path, entity_index_path, relation_index_path)
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path)
    test_data = data_utils.load_triples(test_path, entity_index_path, relation_index_path)
    lf.eval()
    lf.load_checkpoint(get_checkpoint_path(args))
    train_scores = lf.forward_fact(train_data)
    dev_scores = lf.forward_fact(dev_data)
    test_scores = lf.forward_fact(test_data)

    print('Train set average fact score: {}'.format(float(train_scores.mean())))
    print('Dev set average fact score: {}'.format(float(dev_scores.mean())))
    print('Test set average fact score: {}'.format(float(test_scores.mean())))

def get_checkpoint_path(args):
    if not args.checkpoint_path:
        return os.path.join(args.model_dir, 'model_best.tar')
    else:
        return args.checkpoint_path

def load_configs(config_path):
    with open(config_path) as f:
        print('loading configuration file {}'.format(config_path))
        for line in f:
            if not '=' in line:
                continue
            arg_name, arg_value = line.strip().split('=')
            if arg_value.startswith('"') and arg_value.endswith('"'):
                arg_value = arg_value[1:-1]
            if hasattr(args, arg_name):
                print('{} = {}'.format(arg_name, arg_value))
                arg_value2 = getattr(args, arg_name)
                if type(arg_value2) is str:
                    setattr(args, arg_name, arg_value)
                elif type(arg_value2) is bool:
                    if arg_value == 'True':
                        setattr(args, arg_name, True)
                    elif arg_value == 'False':
                        setattr(args, arg_name, False)
                    else:
                        raise ValueError('Unrecognized boolean value description: {}'.format(arg_value))
                elif type(arg_value2) is int:
                    setattr(args, arg_name, int(arg_value))
                elif type(arg_value2) is float:
                    setattr(args, arg_name, float(arg_value))
                else:
                    raise ValueError('Unrecognized attribute type: {}: {}'.format(arg_name, type(arg_value2)))
            else:
                raise ValueError('Unrecognized argument: {}'.format(arg_name))
    return args

def run_experiment(args):

    if args.test:
        if 'NELL' in args.data_dir:
            dataset = os.path.basename(args.data_dir)
            args.distmult_state_dict_path = data_utils.change_to_test_model_path(dataset, args.distmult_state_dict_path)
            args.complex_state_dict_path = data_utils.change_to_test_model_path(dataset, args.complex_state_dict_path)
            args.conve_state_dict_path = data_utils.change_to_test_model_path(dataset, args.conve_state_dict_path)
        args.data_dir += '.test'

    if args.process_data:

        # Process knowledge graph data

        process_data()
    else:
        with torch.set_grad_enabled(args.train or args.search_random_seed or args.grid_search):
            if args.search_random_seed:

                # Search for best random seed

                # search log file
                task = os.path.basename(os.path.normpath(args.data_dir))
                out_log = '{}.{}.rss'.format(task, args.model)
                o_f = open(out_log, 'w')

                print('** Search Random Seed **')
                o_f.write('** Search Random Seed **\n')
                o_f.close()
                num_runs = 5

                hits_at_1s = {}
                hits_at_10s = {}
                mrrs = {}
                mrrs_search = {}
                for i in range(num_runs):

                    o_f = open(out_log, 'a')

                    random_seed = random.randint(0, 1e16)
                    print("\nRandom seed = {}\n".format(random_seed))
                    o_f.write("\nRandom seed = {}\n\n".format(random_seed))
                    torch.manual_seed(random_seed)
                    torch.cuda.manual_seed_all(args, random_seed)
                    initialize_model_directory(args, random_seed)
                    lf = construct_model(args)
                    lf.cuda()
                    train(lf)
                    metrics = inference(lf)
                    hits_at_1s[random_seed] = metrics['test']['hits_at_1']
                    hits_at_10s[random_seed] = metrics['test']['hits_at_10']
                    mrrs[random_seed] = metrics['test']['mrr']
                    mrrs_search[random_seed] = metrics['dev']['mrr']
                    # print the results of the hyperparameter combinations searched so far
                    print('------------------------------------------')
                    print('Random Seed\t@1\t@10\tMRR')
                    for key in hits_at_1s:
                        print('{}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    print('------------------------------------------')
                    o_f.write('------------------------------------------\n')
                    o_f.write('Random Seed\t@1\t@10\tMRR\n')
                    for key in hits_at_1s:
                        o_f.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    o_f.write('------------------------------------------\n')

                    # compute result variance
                    import numpy as np
                    hits_at_1s_ = list(hits_at_1s.values())
                    hits_at_10s_ = list(hits_at_10s.values())
                    mrrs_ = list(mrrs.values())
                    print('Hits@1 mean: {:.3f}\tstd: {:.6f}'.format(np.mean(hits_at_1s_), np.std(hits_at_1s_)))
                    print('Hits@10 mean: {:.3f}\tstd: {:.6f}'.format(np.mean(hits_at_10s_), np.std(hits_at_10s_)))
                    print('MRR mean: {:.3f}\tstd: {:.6f}'.format(np.mean(mrrs_), np.std(mrrs_)))
                    o_f.write('Hits@1 mean: {:.3f}\tstd: {:.6f}\n'.format(np.mean(hits_at_1s_), np.std(hits_at_1s_)))
                    o_f.write('Hits@10 mean: {:.3f}\tstd: {:.6f}\n'.format(np.mean(hits_at_10s_), np.std(hits_at_10s_)))
                    o_f.write('MRR mean: {:.3f}\tstd: {:.6f}\n'.format(np.mean(mrrs_), np.std(mrrs_)))
                    o_f.close()
                    
                # find best random seed
                best_random_seed, best_mrr = sorted(mrrs_search.items(), key=lambda x: x[1], reverse=True)[0]
                print('* Best Random Seed = {}'.format(best_random_seed))
                print('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}'.format(
                    hits_at_1s[best_random_seed],
                    hits_at_10s[best_random_seed],
                    mrrs[best_random_seed]))
                with open(out_log, 'a'):
                    o_f.write('* Best Random Seed = {}\n'.format(best_random_seed))
                    o_f.write('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}\n'.format(
                        hits_at_1s[best_random_seed],
                        hits_at_10s[best_random_seed],
                        mrrs[best_random_seed])
                    )
                    o_f.close()

            elif args.grid_search:

                # Grid search

                # search log file
                task = os.path.basename(os.path.normpath(args.data_dir))
                out_log = '{}.{}.gs'.format(task, args.model)
                o_f = open(out_log, 'w')

                print("** Grid Search **")
                o_f.write("** Grid Search **\n")
                hyperparameters = args.tune.split(',')

                if args.tune == '' or len(hyperparameters) < 1:
                    print("No hyperparameter specified.")
                    sys.exit(0)

                grid = hp_range[hyperparameters[0]]
                for hp in hyperparameters[1:]:
                    grid = itertools.product(grid, hp_range[hp])

                hits_at_1s = {}
                hits_at_10s = {}
                mrrs = {}
                grid = list(grid)
                print('* {} hyperparameter combinations to try'.format(len(grid)))
                o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
                o_f.close()

                for i, grid_entry in enumerate(list(grid)):

                    o_f = open(out_log, 'a')

                    if not (type(grid_entry) is list or type(grid_entry) is list):
                        grid_entry = [grid_entry]
                    grid_entry = flatten(grid_entry)
                    print('* Hyperparameter Set {}:'.format(i))
                    o_f.write('* Hyperparameter Set {}:\n'.format(i))
                    signature = ''
                    for j in range(len(grid_entry)):
                        hp = hyperparameters[j]
                        value = grid_entry[j]
                        if hp == 'bandwidth':
                            setattr(args, hp, int(value))
                        else:
                            setattr(args, hp, float(value))
                        signature += ':{}'.format(value)
                        print('* {}: {}'.format(hp, value))
                    initialize_model_directory(args)
                    lf = construct_model(args)
                    lf.cuda()
                    train(lf)
                    metrics = inference(lf)
                    hits_at_1s[signature] = metrics['dev']['hits_at_1']
                    hits_at_10s[signature] = metrics['dev']['hits_at_10']
                    mrrs[signature] = metrics['dev']['mrr']
                    # print the results of the hyperparameter combinations searched so far
                    print('------------------------------------------')
                    print('Signature\t@1\t@10\tMRR')
                    for key in hits_at_1s:
                        print('{}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    print('------------------------------------------\n')
                    o_f.write('------------------------------------------\n')
                    o_f.write('Signature\t@1\t@10\tMRR\n')
                    for key in hits_at_1s:
                        o_f.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    o_f.write('------------------------------------------\n')
                    # find best hyperparameter set
                    best_signature, best_mrr = sorted(mrrs.items(), key=lambda x:x[1], reverse=True)[0]
                    print('* best hyperparameter set')
                    o_f.write('* best hyperparameter set\n')
                    best_hp_values = best_signature.split(':')[1:]
                    for i, value in enumerate(best_hp_values):
                        hp_name = hyperparameters[i]
                        hp_value = best_hp_values[i]
                        print('* {}: {}'.format(hp_name, hp_value))
                    print('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}'.format(
                        hits_at_1s[best_signature],
                        hits_at_10s[best_signature],
                        mrrs[best_signature]
                    ))
                    o_f.write('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}\ns'.format(
                        hits_at_1s[best_signature],
                        hits_at_10s[best_signature],
                        mrrs[best_signature]
                    ))

                    o_f.close()

            elif args.run_ablation_studies:
                run_ablation_studies(args)
            else:
                initialize_model_directory(args)
                lf = construct_model(args)
                lf.cuda()

                if args.train:
                    train(lf)
                elif args.inference:
                    inference(lf)
                elif args.eval_by_relation_type:
                    inference(lf)
                elif args.eval_by_seen_queries:
                    inference(lf)
                elif args.export_to_embedding_projector:
                    export_to_embedding_projector(lf)
                elif args.export_reward_shaping_parameters:
                    export_reward_shaping_parameters(lf)
                elif args.compute_fact_scores:
                    compute_fact_scores(lf)
                elif args.export_fuzzy_facts:
                    export_fuzzy_facts(lf)
                elif args.export_error_cases:
                    export_error_cases(lf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Hop Knowledge Graph Reasoning with Reward Shaping')

    # Experiment control
    parser.add_argument('--process_data', action='store_true',
                    help='process knowledge graph (default: False)')
    parser.add_argument('--train', action='store_true',
                    help='run path selection set_policy training (default: False)')
    parser.add_argument('--inference', action='store_true',
                    help='run knowledge graph inference (default: False)')
    parser.add_argument('--search_random_seed', action='store_true',
                    help='run experiments with multiple random initializations and compute the result statistics '
                            '(default: False)')
    parser.add_argument('--eval', action='store_true',
                    help='compute evaluation metrics (default: False)')
    parser.add_argument('--eval_by_relation_type', action='store_true',
                    help='compute evaluation metrics for to-M and to-1 relations separately (default: False)')
    parser.add_argument('--eval_by_seen_queries', action='store_true',
                    help='compute evaluation metrics for seen queries and unseen queries separately (default: False)')
    parser.add_argument('--run_ablation_studies', action='store_true',
                    help='run ablation studies')
    parser.add_argument('--run_analysis', action='store_true',
                    help='run algorithm analysis and print intermediate results (default: False)')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
                    help='directory where the knowledge graph data is stored (default: None)')
    parser.add_argument('--model_root_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'),
                    help='root directory where the model parameters are stored (default: None)')
    parser.add_argument('--model_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'),
                    help='directory where the model parameters are stored (default: None)')
    parser.add_argument('--gpu', type=int, default=0,
                    help='gpu device (default: 0)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='path to a pretrained checkpoint')

    # Data
    parser.add_argument('--test', action='store_true',
                    help='perform inference on the test set (default: False)')
    parser.add_argument('--group_examples_by_query', action='store_true',
                    help='group examples by topic entity + query relation (default: False)')

    # Network Architecture
    parser.add_argument('--model', type=str, default='point',
                    help='knowledge graph QA model (default: point)')
    parser.add_argument('--entity_dim', type=int, default=200, metavar='E',
                    help='entity embedding dimension (default: 200)')
    parser.add_argument('--relation_dim', type=int, default=200, metavar='R',
                    help='relation embedding dimension (default: 200)')
    parser.add_argument('--history_dim', type=int, default=400, metavar='H',
                    help='action history encoding LSTM hidden states dimension (default: 400)')
    parser.add_argument('--history_num_layers', type=int, default=3, metavar='L',
                    help='action history encoding LSTM number of layers (default: 1)')
    parser.add_argument('--use_action_space_bucketing', action='store_true',
                    help='bucket adjacency list by outgoing degree to avoid memory blow-up (default: False)')
    parser.add_argument('--bucket_interval', type=int, default=10,
                    help='adjacency list bucket size (default: 32)')
    parser.add_argument('--type_only', action='store_true',
                    help='use denote knowledge graph node by entity types only (default: False)')
    parser.add_argument('--relation_only', action='store_true',
                    help='search with relation information only, ignoring entity representation (default: False)')
    parser.add_argument('--relation_only_in_path', action='store_true',
                    help='include intermediate entities in path (default: False)')

    # Knowledge Graph
    parser.add_argument('--num_graph_convolution_layers', type=int, default=0,
                    help='number of graph convolution layers to use (default: 0, no GC is used)')
    parser.add_argument('--graph_convolution_rank', type=int, default=10,
                    help='number of ranks ')
    parser.add_argument('--add_reverse_relations', type=bool, default=True,
                    help='add reverse relations to KB (default: True)')
    parser.add_argument('--add_reversed_training_edges', action='store_true',
                    help='add reversed edges to extend training set (default: False)')
    parser.add_argument('--train_entire_graph', type=bool, default=False,
                    help='add all edges in the graph to extend training set (default: False)')
    parser.add_argument('--emb_dropout_rate', type=float, default=0.3,
                    help='Knowledge graph embedding dropout rate (default: 0.3)')
    parser.add_argument('--zero_entity_initialization', type=bool, default=False,
                    help='Initialize all entities to zero (default: False)')
    parser.add_argument('--uniform_entity_initialization', type=bool, default=False,
                    help='Initialize all entities with the same random embedding (default: False)')

    # Optimization
    parser.add_argument('--num_epochs', type=int, default=200,
                    help='maximum number of pass over the entire training set (default: 20)')
    parser.add_argument('--num_wait_epochs', type=int, default=5,
                    help='number of epochs to wait before stopping training if dev set performance drops')
    parser.add_argument('--num_peek_epochs', type=int, default=1,
                    help='number of epochs to wait for next dev set result check (default: 2)')
    parser.add_argument('--start_epoch', type=int, default=0,
                    help='epoch from which the training should start (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                    help='mini-batch size (default: 256)')
    parser.add_argument('--train_batch_size', type=int, default=256,
                    help='mini-batch size during training (default: 256)')
    parser.add_argument('--dev_batch_size', type=int, default=64,
                    help='mini-batch size during inferece (default: 64)')
    parser.add_argument('--margin', type=float, default=0,
                    help='margin used for base MAMES training (default: 0)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
    parser.add_argument('--learning_rate_decay', type=float, default=1.0,
                    help='learning rate decay factor for the Adam optimizer (default: 1)')
    parser.add_argument('--adam_beta1', type=float, default=0.9,
                    help='Adam: decay rates for the first movement estimate (default: 0.9)')
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                    help='Adam: decay rates for the second raw movement estimate (default: 0.999)')
    parser.add_argument('--grad_norm', type=float, default=10000,
                    help='norm threshold for gradient clipping (default 10000)')
    parser.add_argument('--xavier_initialization', type=bool, default=True,
                    help='Initialize all model parameters using xavier initialization (default: True)')
    parser.add_argument('--random_parameters', type=bool, default=False,
                    help='Inference with random parameters (default: False)')

    # Fact Network
    parser.add_argument('--label_smoothing_epsilon', type=float, default=0.1,
                    help='epsilon used for label smoothing')
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.3,
                    help='ConvE hidden layer dropout rate (default: 0.3)')
    parser.add_argument('--feat_dropout_rate', type=float, default=0.2,
                    help='ConvE feature dropout rate (default: 0.2)')
    parser.add_argument('--emb_2D_d1', type=int, default=10,
                    help='ConvE embedding 2D shape dimension 1 (default: 10)')
    parser.add_argument('--emb_2D_d2', type=int, default=20,
                    help='ConvE embedding 2D shape dimension 2 (default: 20)')
    parser.add_argument('--num_out_channels', type=int, default=32,
                    help='ConvE number of output channels of the convolution layer (default: 32)')
    parser.add_argument('--kernel_size', type=int, default=3,
                    help='ConvE kernel size (default: 3)')
    parser.add_argument('--distmult_state_dict_path', type=str, default='',
                    help='Path to the DistMult network state_dict (default: '')')
    parser.add_argument('--complex_state_dict_path', type=str, default='',
                    help='Path to the ComplEx network state dict (default: '')')
    parser.add_argument('--conve_state_dict_path', type=str, default='',
                    help='Path to the ConvE network state dict (default: '')')
    parser.add_argument('--tucker_state_dict_path', type=str, default='',
                    help='Path to the TuckER network state dict (default: '')')
    parser.add_argument('--ptranse_state_dict_path', type=str, default='',
                    help='Path to the PTransE network state dict (default: '')')
    parser.add_argument('--TransE_state_dict_path', type=str, default='',
                    help='Path to the TransE network state dict (default: '')')

    # Policy Network
    parser.add_argument('--ff_dropout_rate', type=float, default=0.1,
                    help='Feed-forward layer dropout rate (default: 0.1)')
    parser.add_argument('--rnn_dropout_rate', type=float, default=0.0,
                    help='RNN Variational Dropout Rate (default: 0.0)')
    parser.add_argument('--action_dropout_rate', type=float, default=0.1,
                    help='Dropout rate for randomly masking out knowledge graph edges (default: 0.1)')
    parser.add_argument('--action_dropout_anneal_factor', type=float, default=0.95,
                    help='Decrease the action dropout rate once the dev set results stopped increase (default: 0.95)')
    parser.add_argument('--action_dropout_anneal_interval', type=int, default=1000,
                    help='Number of epochs to wait before decreasing the action dropout rate (default: 1000. Action '
                            'dropout annealing is not used when the value is >= 1000.)')
    parser.add_argument('--num_negative_samples', type=int, default=10,
                    help='Number of negative samples to use for embedding-based methods')

    # Reward Shaping
    parser.add_argument('--fn_state_dict_path', type=str, default='',
                    help='(Aborted) Path to the saved fact network model')
    parser.add_argument('--fn_kg_state_dict_path', type=str, default='',
                    help='(Aborted) Path to the saved knowledge graph embeddings used by a fact network')
    parser.add_argument('--reward_shaping_threshold', type=float, default=0,
                    help='Threshold cut off of reward shaping scores (default: 0)')
    parser.add_argument('--mu', type=float, default=1.0,
                    help='Weight over the estimated reward (default: 1.0)')

    # Graph Completion
    parser.add_argument('--theta', type=float, default=0.2,
                    help='Threshold for sifting high-confidence facts (default: 0.2)')

    # Reinforcement Learning
    parser.add_argument('--num_rollouts', type=int, default=20,
                    help='number of rollouts (default: 20)')
    parser.add_argument('--num_rollout_steps', type=int, default=3,
                    help='maximum path length (default: 3)')
    parser.add_argument('--bandwidth', type=int, default=300,
                    help='maximum number of outgoing edges to explore at each step (default: 300)')
    parser.add_argument('--r_bandwidth', type=int, default=10,
                    help='maximum number of unique relation types connecting a pair of entities (default: 10)')
    parser.add_argument('--num_paths_per_entity', type=int, default=3,
                    help='number of paths used to calculate entity potential (default: 3)')
    parser.add_argument('--beta', type=float, default=0.0,
                    help='entropy regularization weight (default: 0.0)')
    parser.add_argument('--gamma', type=float, default=1,
                    help='moving average weight (default: 1)')

    # Policy Gradient
    parser.add_argument('--baseline', type=str, default='n/a',
                    help='baseline used by the policy gradient algorithm (default: n/a)')

    parser.add_argument('--seed', type=int, default=543, metavar='S',
                    help='random seed (default: 543)')

    # Search Decoding
    parser.add_argument('--beam_size', type=int, default=100,
                    help='size of beam used in beam search inference (default: 100)')
    parser.add_argument('--mask_test_false_negatives', type=bool, default=False,
                    help='mask false negative examples in the dev/test set during decoding (default: False. This flag '
                            'was implemented for sanity checking and was not used in any experiment.)')
    parser.add_argument('--visualize_paths', action='store_true',
                    help='generate path visualizations during inference (default: False)')
    parser.add_argument('--save_paths_to_csv', action='store_true',
                    help='save the decoded path into a CSV file (default: False)')

    # Separate Experiments
    parser.add_argument('--export_to_embedding_projector', action='store_true',
                    help='export model embeddings to the Tensorflow Embedding Projector format (default: False)')
    parser.add_argument('--export_reward_shaping_parameters', action='store_true',
                    help='export KG embeddings and fact network parameters for reward shaping models (default: False)')
    parser.add_argument('--compute_fact_scores', action='store_true',
                    help='[Debugging Option] compute embedding based model scores (default: False)')
    parser.add_argument('--export_fuzzy_facts', action='store_true',
                    help='export the facts recovered by embedding based method (default: False)')
    parser.add_argument('--export_error_cases', action='store_true',
                    help='export the error cases of a model')
    parser.add_argument('--compute_map', action='store_true',
                    help='compute the Mean Average Precision evaluation metrics (default: False)')

    # Hyperparameter Search
    parser.add_argument('--tune', type=str, default='',
                    help='Specify the hyperparameters to tune during the search, separated by commas (default: None)')
    parser.add_argument('--grid_search', action='store_true',
                    help='Conduct grid search of hyperparameters')

    parser.add_argument('--use_action_selection', action='store_true',
                    help='search with relation information only, ignoring entity representation (default: False)')
    parser.add_argument('--use_state_prediction', action='store_true',
                    help='search with relation information only, ignoring entity representation (default: False)')
    parser.add_argument('--mask_sim_relation', action='store_true',
                    help='search with relation information only, ignoring entity representation (default: False)')
    parser.add_argument('--max_dynamic_action_size', type=int, default=256,
                    help='mini-batch size during training (default: 256)')
    parser.add_argument('--dynamic_split_bound', type=float, default=256,
                    help='mini-batch size during training (default: 256)')
    parser.add_argument('--avg_entity_per_relation', type=int, default=256,
                    help='mini-batch size during training (default: 256)')
    parser.add_argument('--strategy', type=str, default='sample',
                    help='baseline used by the policy gradient algorithm (default: n/a)')

    args = parser.parse_args()

    hp_range = {
    "beta": [0., .01, .05, .1],
    "emb_dropout_rate": [0, .1, .2, .3],
    "ff_dropout_rate": [0, .1, .2, .3],
    "action_dropout_rate": [.95],
    "bandwidth": [200, 400],
    "relation_only": [True, False]
}
    run_experiment(args)
