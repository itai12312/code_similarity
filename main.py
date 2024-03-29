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
import random

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

    vector_path = join(params.output_folder, 'vectors/vectors.npz')
    tfidf_path = join(params.output_folder, 'tfidf.npz')
    distances_path = join(params.output_folder, 'distances.npz')
    vectors_all_path = os.path.join(params.output_folder, f'vectors_all.npz')
    vectors_folder = join(params.output_folder)

    if params.cores_to_use == -1:
        params.cores_to_use = multiprocessing.cpu_count()
    true_cores = params.cores_to_use
    params.cores_to_use = 1
    s = params.files_limit_start
    e = min([params.files_limit_end,111000,len(os.listdir(os.path.join(params.input_folder, 'tokenized1')))])
    params.select_functions_limit = min(params.select_functions_limit, e)
    q = Queue()


    if 'vectors' in params.stages_to_run or (not os.path.exists(vectors_all_path) and is_in(['tfidf', 'distances', 'clustering'], params.stages_to_run)):
        count = s
        q_len = 0
        while count < e:
            if not os.path.exists(vector_path[:-4]+str(count)+'.npz'):
                q.put(count)
                q_len += 1
            count += params.files_limit_step
        if q_len > 0:
            multi_process_run(UserProcessTask(params, list_of_tokens, q), true_cores)
        bow_matrix, lists, all_ends_raw, gt_values, filenames_list, all_vulnerabilities, all_start_raw, vocab, idf, all_functions_count = load_vectors_iter_folder(e, s, params.files_limit_step, vector_path, params.output_folder, params.security_keywords, indices=params.select_functions_limit)
        np.savez_compressed(vectors_all_path, bow_matrix=bow_matrix, lists=lists,
                            all_start_ends=all_ends_raw, gt_values=gt_values,
                            filenames_list=filenames_list, all_vulnerabilities=all_vulnerabilities,
                            all_start_raw=all_start_raw,idf=idf,all_functions_count=all_functions_count)
        upload_to_gcp(params)
    q.close()

    if 'tfidf' in params.stages_to_run or (not os.path.exists(tfidf_path) and is_in(['distances', 'clustering'], params.stages_to_run)):
        if 'vectors' not in params.stages_to_run:
            # data = load_vectors(vector_path)
            # bow_matrix, lists, raw_lists, gt_values, filenames_list,\
            #    all_vulnerabilities, all_start_raw, vocab = data
            bow_matrix, lists, all_ends_raw, gt_values, filenames_list, all_vulnerabilities, all_start_raw, vocab, idf, all_functions_count = load_vectors_iter(vectors_folder) # load_vectors_iter_folder(e,s, params.files_limit_step, vector_path, indices=params.select_functions_limit)
        # intersting_indices = np.array(list(range(len(lists))))
        # if scipy.sparse.issparse(bow_matrix):
        #    matrix = bow_matrix.toarray()
        if params.vectorizer == 'count' and params.matrix_form == 'tfidf':
            matrix = count_to_tfidf(bow_matrix, idf, all_functions_count)
        np.savez_compressed(tfidf_path, matrix=matrix)
        upload_to_gcp(params)
    
    if 'distances' in params.stages_to_run or (not os.path.exists(distances_path) and is_in(['clustering'], params.stages_to_run)):
        if 'tfidf' not in params.stages_to_run:
            matrix = np.load(tfidf_path)['matrix']
        distances = pdist(matrix, metric=params.metric)
        np.savez_compressed(distances_path, distances=distances)
        upload_to_gcp(params)
    
    if 'clustering' in params.stages_to_run:
        if 'vectors' not in params.stages_to_run and 'tfidf' not in params.stages_to_run:
            bow_matrix, lists, all_ends_raw, gt_values, filenames_list, all_vulnerabilities, all_start_raw, vocab, idf, all_functions_count = load_vectors_iter(vectors_folder)
        if 'tfidf' not in params.stages_to_run:
            matrix = np.load(tfidf_path)['matrix']
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
    all_vulnerabilities, all_start_raw, idf, all_functions_count, _ = load_vectors(join(vector_path,'vectors_all.npz'))
    vocab, _ = load_vectors(join(vector_path, 'vectors_vocab.npz'))
    return bow_matrix, lists, all_ends_raw, gt_values, filenames_list, \
           all_vulnerabilities, all_start_raw, vocab, idf, all_functions_count


def load_vectors_iter_folder(end, start, step, vector_path, main_folder, keywords, indices=None, load_indices=True):
    indices_path = join(main_folder, 'vectors_indices.npy')
    if not os.path.exists(indices_path):
        number_of_functions = 0
        count = start
        while count < end:
            _, len_matrix = load_vectors(vector_path[:-4]+str(count)+'.npz', load=scipy.sparse.load_npz, ret_as_is=True)
            count += step
            number_of_functions += len_matrix
        if indices is None:
            save_indices = np.array(list(range(number_of_functions)))
        elif isinstance(indices, int):
            save_indices = np.array(random.sample(range(number_of_functions), indices))
        else:
            save_indices = indices
        np.save(indices_path, save_indices)
    save_indices = np.load(indices_path)
    count = start
    functions_count = 0
    vocab, _ = load_vectors(join(main_folder, 'vectors','vectors_vocab.npz'))
    if keywords is not None:
        def gt(item):
            ret = is_in(keywords, item)
            return 1 if ret else 0
    else:
        def gt(item):
            return 1 if item !='' else 0
    get_gt = np.vectorize(gt)
    while count < end:
        print(f'loading {count}')
        if count == start:
            bow_matrix, current_function_count, idf = load_vectors(vector_path[:-4]+str(count)+'.npz', functions_count, save_indices, load=scipy.sparse.load_npz, ret_as_is=True ,load_indices=load_indices)
            bow_matrix = bow_matrix.toarray()
            lists, all_ends_raw, _, filenames_list, \
            all_vulnerabilities, all_start_raw, _ = load_vectors(vector_path[:-4]+'_metadata'+str(count)+'.npz', functions_count, save_indices ,load_indices=load_indices)
            functions_count += current_function_count
            gt_values = get_gt(all_vulnerabilities)
        else:
            temp_bow_matrix, current_function_count, temp_idf = load_vectors(vector_path[:-4]+str(count)+'.npz', functions_count, save_indices, load=scipy.sparse.load_npz, ret_as_is=True ,load_indices=load_indices)
            temp_bow_matrix = temp_bow_matrix.toarray()
            temp_lists, temp_ends_raw, _, temp_filenames_list, \
            temp_all_vulnerabilities, temp_all_start_raw, _ = load_vectors(vector_path[:-4]+'_metadata'+str(count)+'.npz', functions_count, save_indices ,load_indices=load_indices)
            temp_gt_values = get_gt(temp_all_vulnerabilities)
            if len(temp_bow_matrix) > 0:
                bow_matrix = np.concatenate([bow_matrix, temp_bow_matrix])
                lists = np.concatenate([lists, temp_lists])
                all_ends_raw = np.concatenate([all_ends_raw, temp_ends_raw])
                gt_values = np.concatenate([gt_values, temp_gt_values])
                filenames_list = np.concatenate([filenames_list, temp_filenames_list])
                all_vulnerabilities = np.concatenate([all_vulnerabilities, temp_all_vulnerabilities])
                all_start_raw = np.concatenate([all_start_raw, temp_all_start_raw])
            # assert (vocab == temp_vocab).all()
            functions_count += current_function_count
            idf += temp_idf
        count += step
    return bow_matrix, lists, all_ends_raw, gt_values, filenames_list, \
           all_vulnerabilities, all_start_raw, vocab, idf, functions_count


def load_vectors(vector_path, functions_count=None, save_indices=None, load=np.load, ret_as_is=False, load_indices=True):
    data = load(vector_path)
    if ret_as_is:
        functions_in_disk = data.shape[0]
        if save_indices is not None:
            legal_indices = save_indices-functions_count
            legal_indices = legal_indices[(legal_indices>=0) & (legal_indices < functions_in_disk)]
            if load_indices is False:
                mask = np.ones((data.shape[0], ), dtype=bool)  # np.ones_like(a,dtype=bool)
                mask[legal_indices] = False
                legal_indices = legal_indices
            idf = copy.deepcopy(data.toarray())
            idf[idf>1] = 1
            return data[legal_indices], functions_in_disk, idf.sum(axis=0)
        else:
            return data, functions_in_disk
    data = [data[att] for att in data.files]
    functions_in_disk = data[0].shape[0]
    if save_indices is not None:
        legal_indices = save_indices - functions_count
        legal_indices = legal_indices[(legal_indices >= 0) & (legal_indices < functions_in_disk)]
        if load_indices is False:
            mask = np.ones((data[0].shape[0],), dtype=bool)  # np.ones_like(a,dtype=bool)
            mask[legal_indices] = False
            legal_indices = legal_indices
        data = [d[legal_indices] for d in data]
    return (*data, functions_in_disk)


def count_to_tfidf(bow_matrix, idf, functions_count):
    # new_params = copy.deepcopy(params)
    # new_params.vectorizer = 'tfidf'
    # bow_matrix1, gt_values1, lists1, raw_lists1, vectorizer1, filenames_list1, all_vulnerabilities1, all_start_raw1 = \
    #     get_all_needed_inputs_params(new_params, list_of_tokens)
    #c = copy.deepcopy(bow_matrix)
    #c[c >= 1] = 1
    #c = np.tile(np.sum(c, axis=0), (bow_matrix.shape[0], 1))
    c = np.tile(idf, (bow_matrix.shape[0],1))
    matrix = bow_matrix * 1. / bow_matrix.sum(axis=1)[:, None]
    matrix2 = matrix * (1 + np.log((functions_count + 1) / (1 + c)))
    matrix2 = sklearn.preprocessing.data.normalize(matrix2, norm='l2', copy=False)
    matrix = matrix2
    return matrix


def get_vocab(select_top_tokens, path):
    tokens = pd.read_csv(path)
    list_of_tokens = tokens['name'].values[:select_top_tokens]
    list_of_tokens = [stri.strip().lower() for stri in list_of_tokens if isinstance(stri, str)]
    return list_of_tokens


def get_all_needed_inputs_params(params, list_of_tokens):
    return generate_vectors(params, join(params.output_folder, 'vectors'), params.cores_to_use, params.input_folder, params.vectorizer,
                            params.max_features, params.ngram_range,
                            params.security_keywords, params.min_token_count, list_of_tokens)


def profile(params):
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    main_(params)
    pr.disable()
    pr.print_stats(sort='cumtime')  # cumtime, ncalls

# --matrix_form tfidf --vectorizer count --metric cosine --input_folder ../codes_short/ --output_folder results_small --stages_to_run vectors tfidf distances clustering --min_cluster_length 5
# --matrix_form tfidf --vectorizer count --metric cosine --input_folder ../codes/ --output_folder resuls_large --stages_to_run clustering
if __name__ == "__main__":
    main()


# import numba as nb
# @nb.vectorize(target="cpu")
# def nb_vf(x):
#     return x+2*x*x+4*x*x*x

