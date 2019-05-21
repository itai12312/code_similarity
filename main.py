import scipy
import sklearn
from scipy.spatial.distance import squareform, pdist
import os
import numpy as np

from analysis import analyze_functions2
from parser_utils import str_to_params
from utils import get_vectors, is_in
from tqdm import tqdm, trange
import pandas as pd
import sys
import copy
from os.path import join
import seaborn as sns
sns.set()


def main(args=None):
    params = str_to_params(args)
    if not os.path.exists(params.output_folder):
        os.mkdir(params.output_folder)
    with open(join(params.output_folder, 'args.txt'), 'w+') as f:
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
    list_of_tokens = get_vocab(params.select_top_tokens, 'short_sorted_freq_list.txt')

    vector_path = join(params.output_folder, 'vectors.npz')
    tfidf_path = join(params.output_folder, 'tfidf.npz')
    distances_path = join(params.output_folder, 'distances.npz')
    
    if 'vectors' in params.stages_to_run or (not os.path.exists(vector_path) and is_in(['tfidf', 'distances', 'clustering'], params.stages_to_run)):
        bow_matrix, gt_values, lists, raw_lists, vectorizer, filenames_list, all_vulnerabilities, all_start_raw = \
            get_all_needed_inputs_params(params, list_of_tokens)
        vocab = vectorizer.vocabulary

    if 'tfidf' in params.stages_to_run or (not os.path.exists(tfidf_path) and is_in(['distances', 'clustering'], params.stages_to_run)):
        if 'vectors' not in params.stages_to_run:
            data = load_vectors(vector_path)
            bow_matrix, lists, raw_lists, gt_values, filenames_list,\
                all_vulnerabilities, all_start_raw, vocab = data
        # intersting_indices = np.array(list(range(len(lists))))
        # if scipy.sparse.issparse(bow_matrix):
        #    matrix = bow_matrix.toarray()
        if params.vectorizer == 'count' and params.matrix_form == 'tfidf':
            matrix = count_to_tfidf(bow_matrix)
        np.savez(tfidf_path, matrix=matrix)
    
    if 'distances' in params.stages_to_run or (not os.path.exists(distances_path) and is_in(['clustering'], params.stages_to_run)):
        if 'tfidf' not in params.stages_to_run:
            matrix = np.load(tfidf_path)['matrix']
        distances = pdist(matrix, metric=params.metric)
        np.savez(distances_path, distances=distances)
    
    if 'clustering' in params.stages_to_run:
        if 'distances' not in params.stages_to_run:
            distances = np.load(distances_path)['distances']
        analyze_functions2(distances, matrix, lists, raw_lists,
                           params, gt_values, filenames_list,
                           all_vulnerabilities, all_start_raw)


def load_vectors(vector_path):
    data = np.load(vector_path)
    data = [data[att] for att in data.files]
    return data


def count_to_tfidf(bow_matrix):
    # new_params = copy.deepcopy(params)
    # new_params.vectorizer = 'tfidf'
    # bow_matrix1, gt_values1, lists1, raw_lists1, vectorizer1, filenames_list1, all_vulnerabilities1, all_start_raw1 = \
    #     get_all_needed_inputs_params(new_params, list_of_tokens)
    c = copy.deepcopy(bow_matrix)
    c[c >= 1] = 1
    c = np.tile(np.sum(c, axis=0), (bow_matrix.shape[0], 1))
    matrix = bow_matrix * 1. / bow_matrix.sum(axis=1)[:, None]
    matrix2 = matrix * (1 + np.log((matrix.shape[0] + 1) / (1 + c)))
    matrix2 = sklearn.preprocessing.data.normalize(matrix2, norm='l2', copy=False)
    matrix = matrix2
    return matrix


def get_vocab(select_top_tokens, path):
    tokens = pd.read_csv(path)
    list_of_tokens = tokens['name'].values[:select_top_tokens]
    list_of_tokens = [stri.strip().lower() for stri in list_of_tokens if isinstance(stri, str)]
    return list_of_tokens


def get_all_needed_inputs_params(params, list_of_tokens):
    return get_vectors(params.output_folder, params.cores_to_use, params.input_folder, params.vectorizer,
                       params.max_features, params.ngram_range, params.files_limit,
                       params.security_keywords, params.min_token_count, list_of_tokens)


def profile(params):
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    main_(params)
    pr.disable()
    pr.print_stats(sort='cumtime')  # cumtime, ncalls


#--matrix_form tfidf --vectorizer count --metric euclidean --input_folder ../codes_short/ --files_limit 100 --output_folder result_fixed_vocab
if __name__ == "__main__":
    main()


# import numba as nb
# @nb.vectorize(target="cpu")
# def nb_vf(x):
#     return x+2*x*x+4*x*x*x

