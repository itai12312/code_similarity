import scipy
import sklearn
from scipy.spatial.distance import squareform, pdist
import os
import numpy as np
import subprocess
from analysis import analyze_functions2
from mp import multi_process_run, BaseTask
from parser_utils import str_to_params
from utils import generate_vectors, is_in, get_filenames
from tqdm import tqdm, trange
import pandas as pd
import sys
import copy
from os.path import join
import seaborn as sns
sns.set()
from multiprocessing import Queue
import multiprocessing

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

def upload_to_gcp(params):
    if params.gcp_bucket is not None:
        folder = params.output_folder
        if folder[-1] != os.sep:
            folder = folder+os.sep
        has_svg = ''
        for filename in os.listdir(folder):
            if filename.endswith('.svg'):
                has_svg = f'{folder}*.svg'
                break
        command = f"gsutil -m cp {has_svg} {folder}*.txt {folder}*.npz gs://{params.gcp_bucket}/{folder}"
        print(f'upload command {command}')
        subprocess.check_output(command, shell=True)


class UserProcessTask(BaseTask):
    def __init__(self, params, list_of_tokens, in_queue):
        super(UserProcessTask, self).__init__(in_queue, None)
        self.params = params
        self.list_of_tokens = list_of_tokens

    def task(self, item):
        print(f'workiin on item {item}')
        self.params.files_limit_start = item
        self.params.files_limit_end = item+self.params.files_limit_step
        # bow_matrix, gt_values, lists, raw_lists, vectorizer, filenames_list, all_vulnerabilities, all_start_raw = \
        get_all_needed_inputs_params(self.params, self.list_of_tokens)
        # vocab = vectorizer.vocabulary
        # upload_to_gcp(self.params)


def main_(params):
    # can be called using dictobj.DictionaryObject({'metric': 'euclidean'}) or
    # str_to_params('--output_folder result3 --metric euclidean --input_folder ../codes_short/ --files_limit 100 --max_features 2000')
    # for construction of params object
    list_of_tokens = get_vocab(params.select_top_tokens, 'short_sorted_freq_list.txt')

    vector_path = join(params.output_folder, 'vectors.npz')
    tfidf_path = join(params.output_folder, 'tfidf.npz')
    distances_path = join(params.output_folder, 'distances.npz')
    if params.cores_to_use == -1:
        params.cores_to_use = multiprocessing.cpu_count()
    true_cores = params.cores_to_use
    params.cores_to_use = 1
    s = params.files_limit_start
    e = min(params.files_limit_end, len(get_filenames(params.input_folder)))
    q = Queue()


    if 'vectors' in params.stages_to_run or (not os.path.exists(vector_path[:-4]+'_all.npz') and is_in(['tfidf', 'distances', 'clustering'], params.stages_to_run)):
        count = s
        q_len = 0
        while count < e:
            if not os.path.exists(vector_path[:-4]+str(count)+'.npz'):
                q.put(count)
                q_len += 1
            count += params.files_limit_step
        if q_len > 0:
            multi_process_run(UserProcessTask(params, list_of_tokens, q), true_cores)
        bow_matrix, lists, all_ends_raw, gt_values, filenames_list, all_vulnerabilities, all_start_raw, vocab = load_vectors_iter_folder(e, s, params.files_limit_step, vector_path)
        np.savez_compressed(os.path.join(params.output_folder, f'vectors_all.npz'), bow_matrix=bow_matrix, lists=lists,
                            all_start_ends=all_ends_raw, gt_values=gt_values,
                            filenames_list=filenames_list, all_vulnerabilities=all_vulnerabilities,
                            all_start_raw=all_start_raw)
        upload_to_gcp(params)

    if 'tfidf' in params.stages_to_run or (not os.path.exists(tfidf_path) and is_in(['distances', 'clustering'], params.stages_to_run)):
        if 'vectors' not in params.stages_to_run:
            # data = load_vectors(vector_path)
            # bow_matrix, lists, raw_lists, gt_values, filenames_list,\
            #    all_vulnerabilities, all_start_raw, vocab = data
            bow_matrix, lists, all_ends_raw, gt_values, filenames_list, all_vulnerabilities, all_start_raw, vocab = load_vectors_iter(vector_path)
        # intersting_indices = np.array(list(range(len(lists))))
        # if scipy.sparse.issparse(bow_matrix):
        #    matrix = bow_matrix.toarray()
        if params.vectorizer == 'count' and params.matrix_form == 'tfidf':
            matrix = count_to_tfidf(bow_matrix)
        np.savez_compressed(tfidf_path, matrix=matrix)
        upload_to_gcp(params)
    
    if 'distances' in params.stages_to_run or (not os.path.exists(distances_path) and is_in(['clustering'], params.stages_to_run)):
        if 'vectors' not in params.stages_to_run and 'tfidf' not in params.stages_to_run:
            bow_matrix, lists, all_ends_raw, gt_values, filenames_list, all_vulnerabilities, all_start_raw, vocab = load_vectors_iter(vector_path)
        if 'tfidf' not in params.stages_to_run:
            matrix = np.load(tfidf_path)['matrix']
        distances = pdist(matrix, metric=params.metric)
        np.savez_compressed(distances_path, distances=distances)
        upload_to_gcp(params)
    
    if 'clustering' in params.stages_to_run:
        if 'distances' not in params.stages_to_run:
            distances = np.load(distances_path)['distances']
        analyze_functions2(distances, matrix, lists, all_ends_raw,
                           params, gt_values, filenames_list,
                           all_vulnerabilities, all_start_raw)
        upload_to_gcp(params)
    print('finished')
    if params.shutdown:
        subprocess.run('sudo shutdown', shell=True) # sudo shutdown 0 on aws machines


def load_vectors_iter(vector_path):
    bow_matrix, lists, all_ends_raw, gt_values, filenames_list, \
    all_vulnerabilities, all_start_raw = load_vectors(vector_path[:-4]+'_all.npz')
    vocab = load_vectors(vector_path[:-4] + '_vocab.npz')
    return bow_matrix, lists, all_ends_raw, gt_values, filenames_list, \
           all_vulnerabilities, all_start_raw, vocab


def load_vectors_iter_folder(e, s, step, vector_path):
    count = 0
    vocab = load_vectors(vector_path[:-4] + '_vocab.npz')
    while count < e:
        print(f'loading {count}')
        if count == s:
            bow_matrix = load_vectors(vector_path[:-4]+str(count)+'.npz', load=scipy.sparse.load_npz, ret_as_is=True).toarray()
            lists, all_ends_raw, gt_values, filenames_list, \
            all_vulnerabilities, all_start_raw = load_vectors(vector_path[:-4]+'_metadata'+str(count)+'.npz')
        else:
            temp_bow_matrix = load_vectors(vector_path[:-4]+str(count)+'.npz', load=scipy.sparse.load_npz, ret_as_is=True).toarray()
            temp_lists, temp_ends_raw, temp_gt_values, temp_filenames_list, \
            temp_all_vulnerabilities, temp_all_start_raw = load_vectors(vector_path[:-4]+'_metadata'+str(count)+'.npz')
    
            bow_matrix = np.concatenate([bow_matrix, temp_bow_matrix])
            lists = np.concatenate([lists, temp_lists])
            all_ends_raw = np.concatenate([all_ends_raw, temp_ends_raw])
            gt_values = np.concatenate([gt_values, temp_gt_values])
            filenames_list = np.concatenate([filenames_list, temp_filenames_list])
            all_vulnerabilities = np.concatenate([all_vulnerabilities, temp_all_vulnerabilities])
            all_start_raw = np.concatenate([all_start_raw, temp_all_start_raw])
            # assert (vocab == temp_vocab).all()
        count += step
    return bow_matrix, lists, all_ends_raw, gt_values, filenames_list, \
           all_vulnerabilities, all_start_raw, vocab


def load_vectors(vector_path, load=np.load, ret_as_is=False):
    data = load(vector_path)
    if ret_as_is:
        return data
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
    return generate_vectors(params, params.output_folder, params.cores_to_use, params.input_folder, params.vectorizer,
                            params.max_features, params.ngram_range,
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

