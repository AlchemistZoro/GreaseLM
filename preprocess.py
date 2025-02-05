import argparse
from multiprocessing import cpu_count
from preprocess_utils.convert_csqa import convert_to_entailment
from preprocess_utils.convert_obqa import convert_to_obqa_statement
from preprocess_utils.conceptnet import extract_english, construct_graph
from preprocess_utils.grounding import create_matcher_patterns, ground
from preprocess_utils.graph import generate_adj_data_from_grounded_concepts__use_LM
from preprocess_utils.tagging import tag
input_paths = {
    'csqa': {
        'train': './data/csqa/train_rand_split.jsonl',
        'dev': './data/csqa/dev_rand_split.jsonl',
        'test': './data/csqa/test_rand_split_no_answers.jsonl',
    },
    'obqa': {
        'train': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl',
        'dev': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl',
        'test': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl',
    },
    'cpnet': {
        'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
}

output_paths = {
    'cpnet': {
        'csv': './data/cpnet/conceptnet.en.csv',
        'vocab': './data/cpnet/concept.txt',
        'patterns': './data/cpnet/matcher_patterns.json',
        'unpruned-graph': './data/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': './data/cpnet/conceptnet.en.pruned.graph',
    },
    'csqa': {
        'statement': {
            'train': './data/csqa/statement/train.statement.jsonl',
            'dev': './data/csqa/statement/dev.statement.jsonl',
            'test': './data/csqa/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/csqa/grounded/train.grounded.jsonl',
            'dev': './data/csqa/grounded/dev.grounded.jsonl',
            'test': './data/csqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/csqa/graph/train.graph.adj.pk',
            'adj-dev': './data/csqa/graph/dev.graph.adj.pk',
            'adj-test': './data/csqa/graph/test.graph.adj.pk',
        },
        'tagged':{
            'train': './data/csqa/tagged/train.tagged.jsonl',
            'dev': './data/csqa/tagged/dev.tagged.jsonl',
            'test': './data/csqa/tagged/test.tagged.jsonl',
        },
    },
    'obqa': {
        'statement': {
            'train': './data/obqa/statement/train.statement.jsonl',
            'dev': './data/obqa/statement/dev.statement.jsonl',
            'test': './data/obqa/statement/test.statement.jsonl',
            'train-fairseq': './data/obqa/fairseq/official/train.jsonl',
            'dev-fairseq': './data/obqa/fairseq/official/valid.jsonl',
            'test-fairseq': './data/obqa/fairseq/official/test.jsonl',
        },
        'grounded': {
            'train': './data/obqa/grounded/train.grounded.jsonl',
            'dev': './data/obqa/grounded/dev.grounded.jsonl',
            'test': './data/obqa/grounded/test.grounded.jsonl',
        },
        'tagged':{
            'train': './data/obqa/tagged/train.tagged.jsonl',
            'dev': './data/obqa/tagged/dev.tagged.jsonl',
            'test': './data/obqa/tagged/test.tagged.jsonl',
        },
        'graph': {
            'adj-train': './data/obqa/graph/train.graph.adj.pk',
            'adj-dev': './data/obqa/graph/dev.graph.adj.pk',
            'adj-test': './data/obqa/graph/test.graph.adj.pk',
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common', 'csqa', 'obqa'], choices=['common', 'csqa', 'obqa'], nargs='+')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        # 'common': [
        #     {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
        #     {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
        #                                        output_paths['cpnet']['unpruned-graph'], False)},
        #     {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
        #                                        output_paths['cpnet']['pruned-graph'], True)},
        #     {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        # ],
        'csqa': [
        #     {'func': convert_to_entailment, 'args': (input_paths['csqa']['train'], output_paths['csqa']['statement']['train'])},
        #     {'func': convert_to_entailment, 'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement']['dev'])},
        #     {'func': convert_to_entailment, 'args': (input_paths['csqa']['test'], output_paths['csqa']['statement']['test'])},
        #     {'func': ground, 'args': (output_paths['csqa']['statement']['train'], output_paths['cpnet']['vocab'],
        #                               output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['train'], args.nprocs)},
        #     {'func': ground, 'args': (output_paths['csqa']['statement']['dev'], output_paths['cpnet']['vocab'],
        #                               output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['dev'], args.nprocs)},
        #     {'func': ground, 'args': (output_paths['csqa']['statement']['test'], output_paths['cpnet']['vocab'],
        #                               output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['test'], args.nprocs)},
        {'func': tag, 'args': (output_paths['csqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                    output_paths['cpnet']['patterns'], output_paths['csqa']['tagged']['train'], args.nprocs)},
        {'func': tag, 'args': (output_paths['csqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                    output_paths['cpnet']['patterns'], output_paths['csqa']['tagged']['dev'], args.nprocs)},
        {'func': tag, 'args': (output_paths['csqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                    output_paths['cpnet']['patterns'], output_paths['csqa']['tagged']['test'], args.nprocs)},
        #     {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-train'], args.nprocs)},
        #     {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-dev'], args.nprocs)},
        #     {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-test'], args.nprocs)},
        ],

        'obqa': [
            # {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['train'], output_paths['obqa']['statement']['train'], output_paths['obqa']['statement']['train-fairseq'])},
            # {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['dev'], output_paths['obqa']['statement']['dev'], output_paths['obqa']['statement']['dev-fairseq'])},
            # {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['test'], output_paths['obqa']['statement']['test'], output_paths['obqa']['statement']['test-fairseq'])},
            # {'func': ground, 'args': (output_paths['obqa']['statement']['train'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['train'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['obqa']['statement']['dev'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['dev'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['obqa']['statement']['test'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['test'], args.nprocs)},
            {'func': tag, 'args': (output_paths['obqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['tagged']['train'], args.nprocs)},
            {'func': tag, 'args': (output_paths['obqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['tagged']['dev'], args.nprocs)},
            {'func': tag, 'args': (output_paths['obqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['tagged']['test'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-train'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-dev'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-test'], args.nprocs)},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
    # pass

