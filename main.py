import scipy
from scipy.spatial.distance import squareform
import os
import numpy as np

from analysis import analyze_functions2
from parser_utils import str_to_params
from utils import get_all_needed_inputs
from tqdm import tqdm, trange
import pandas as pd
import sys
import seaborn as sns
sns.set()


def main(args=None):
    params = str_to_params(args)
    if not os.path.exists(params.output_folder):
        os.mkdir(params.output_folder)
    with open(os.path.join(params.output_folder, 'args.txt'), 'w+') as f:
        f.write(' '.join(args) if args is not None else ' '.join(sys.argv[1:]))
        f.write('\nparams are:{}\n'.format(params))
    if params.profiler:
        profile(params)
    else:
        main_(params)


def main_(params):
    # can be called using dictobj.DictionaryObject({'metric': 'euclidean'}) or
    # str_to_params('--output_folder result3 --metric euclidean --input_folder ../codes_short/ --files_limit 100 --max_features 2000')
    # for construction of params object
    tokens = pd.read_csv('sorted_freq_list.txt')
    list_of_tokens = tokens['name'].values[:params.select_top_tokens]
    list_of_tokens = [stri.strip().lower() for stri in list_of_tokens]
    bow_matrix, gt_values, lists, raw_lists, vectorizer, filenames_list, all_vulnerabilities, all_start_raw = \
        get_all_needed_inputs_params(params, list_of_tokens)
    # intersting_indices = analyze_functions(bow_matrix, METRIC_FUNCTIONS[params.metric], lists, raw_lists,
    #                   vocab, params, gt_values)
    # to access metrics directly, look in scipy.spatial.distance
    with open(os.path.join(params.output_folder, 'dump_results.numpy_savez'), 'wb+') as f:
        np.savez(f, bow_matrix=bow_matrix, lists=lists,
                 raw_lists=raw_lists, gt_values=gt_values,
                 filenames_list=filenames_list, all_vulnerabilities=all_vulnerabilities,
                 all_start_raw=all_start_raw)
    intersting_indices = np.array(list(range(len(lists))))
    analyze_functions2(bow_matrix[intersting_indices], lists[intersting_indices], raw_lists[intersting_indices],
                       params, gt_values[intersting_indices], filenames_list[intersting_indices],
                       all_vulnerabilities[intersting_indices], all_start_raw[intersting_indices])


def get_all_needed_inputs_params(params, list_of_tokens):
    return get_all_needed_inputs(params.output_folder, params.cores_to_use, params.input_folder, params.vectorizer,
                                 params.max_features, params.ngram_range, params.files_limit,
                                 params.security_keywords, params.min_token_count, list_of_tokens)


def profile(params):
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    main_(params)
    pr.disable()
    pr.print_stats(sort='cumtime')  # cumtime, ncalls


#--matrix_form tfidf --vectorizer count --metric euclidean --input_folder ../codes_short/ --files_limit 100 --output_folder result_fixed_vocab_100_picked
if __name__ == "__main__":
    main()


# import numba as nb
# @nb.vectorize(target="cpu")
# def nb_vf(x):
#     return x+2*x*x+4*x*x*x

