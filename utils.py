import pandas as pd
from os.path import isfile, join
import seaborn as sns
# import nltk
import copy
import os
import math
import parse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pathos.multiprocessing as multiprocessing
from tqdm import tqdm, trange
import numpy as np
import scipy

def generate_vectors(params, output_folder, cores_to_use, input_folder, vectorizer,
                     max_features, ngram_range=1, security_keywords=None,
                     min_token_count=-1, list_of_tokens=None):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    print(f'using {cores_to_use} cores')
    mypath = input_folder
    vectorizer = {'count': CountVectorizer, 'tfidf': TfidfVectorizer}[vectorizer]
    vectorizer = vectorizer(max_df=1.0, min_df=1, max_features=max_features, ngram_range=(1, ngram_range), vocabulary=list_of_tokens)
    vectorizer1, lists, bow_matrix, all_ends_raw, gt_values, filenames_list, all_vulnerabilities, all_start_raw = vectorize_folder(mypath, params,
                                                                                            vectorizer, output_folder,
                                                                                            cores_to_use,
                                                                                            input_folder,
                                                                                            security_keywords,
                                                                                            min_token_count, list_of_tokens)
    # bow_matrix = bow_matrix.toarray()
    np.savez_compressed(os.path.join(output_folder, f'vectors_metadata{params.files_limit_start}.npz'), lists=lists,
             all_start_ends=all_ends_raw, gt_values=gt_values,
             filenames_list=filenames_list, all_vulnerabilities=all_vulnerabilities,
             all_start_raw=all_start_raw)
    vocab_path = os.path.join(output_folder, f'vectors_vocab.npz')
    if not os.path.exists(vocab_path):
        np.savez_compressed(vocab_path, voacb=vectorizer1.vocabulary)
    scipy.sparse.save_npz(os.path.join(output_folder, f'vectors{params.files_limit_start}.npz'),bow_matrix)
    #for i, token in enumerate(list_of_tokens):
    #    assert sum([1 for c in lists[0].split(" ") if c == token]) == bow_matrix.toarray()[0][i]
    # return bow_matrix, gt_values, lists, raw_lists, vectorizer1, filenames_list, all_vulnerabilities, all_start_raw


def str_ok(stri):
    return len(stri.replace("\n", "")) > 2


def isin(a, b):
    for word in a:
        for sent in b:
            if word in sent:
                return True
    return False


def is_in(a,b):
    for word1 in a:
        if word1 in b:
            return True
    return False


def create_functions_list_from_filename(item):
    (filename, gt, keywords, min_token_count, list_of_tokens) = item
    try:
        #  engine='python',
        df = pd.read_csv(filename, header=None, encoding='utf8')  #  error_bad_lines=False
        with open(filename.replace("tokenized1","c_sharp_code").replace(".tree-viewer.txt", "")) as f:
            data = f.read().splitlines()
    except Exception as e:
        return [],[],[],[], f'{e}', [filename], [], []
    # original_df = copy.deepcopy(df)
    df = df[df[0].notnull()]
    if len(df.index) == 0:
        return [],[], [],[],  f'no functions found!', [filename], [], []
    # df = create_functions_list_from_df(df)
    starters = df.loc[df[0] == "BEGIN_METHOD"]
    enders = df.loc[df[0] == "END_METHOD"]
    if len(starters) != len(enders):
        return [],[], [],[], f'has different number of start and end in parsed!!!', [filename], [], []
    if len(starters) == 0 or len(enders) == 0:
        return [],[], [],[],  f'no functions found!', [filename], [], []
    zipped = list(zip(starters.index, enders.index))
    functions_list = [df[0].iloc[begin:end+1].str.cat(sep=' ') for begin, end in zipped]
    def filter_alpha(stri):
        ret = ''.join(c for c in stri if c.isalnum() or c in [' ', '_', '-'])
        return ' '.join(ret.split())
    functions_list = [filter_alpha(fl.lower()) for fl in functions_list]
    # # functions_list = [function for function in functions_list if len(function.replace("\n", "")) > 0]
    raw_start = df.loc[starters.index+1]
    # raw_end = df.loc[enders.index-1]
    # df[2] = pd.to_numeric(df[2])
    curs = []
    # real_curs = []
    # lim = len(df.values) - 1
    # ls = list(df.index)
    cidx = len(df.values)-1
    while is_not_ok(df.values[cidx,2]) and cidx > 0:
        cidx -= 1
    if cidx == 0:
        return [], [], [], [], f'no tokenized found!', [filename], [], []
    for idx in range(len(enders.index)):
        cur = enders.index[idx]+2
        if cur in df.index:
            frealidx = list(df.index).index(cur)
            if not is_not_ok(df.values[frealidx, 2]):
                realidx = frealidx
        elif idx == len(enders.index) - 1:
            realidx = cidx
        else:
            temp = df.loc[cur+2, 2]
            # realidx = cur+2
            idxi = idx
            realidx = cidx
            c = 0
            print(f'in {filename} at {idx} out of {len(enders.index)}')
            while is_not_ok(temp) and c < 10 and idxi < len(enders.index) - 1 and idxi+1 <= len(starters.index):
                # fix idxi out of bounds too!!!
                if starters.index[idxi+1] <= len(df.index):
                    loc = df.index[starters.index[idxi+1]-1]
                    if loc in list(df.index):
                        realidx = list(df.index).index(loc)
                        temp = df.values[realidx, 2]
                        if is_not_ok(temp):
                            print(f'interesting {idx} {idxi} {filename}')
                    else:
                        temp = math.nan
                else:
                    temp = math.nan
                idxi += 1
                c+= 1
            else:
                if is_not_ok(temp):
                    realidx = cidx
        # steps = 0
        # steps_num = 1000000
        # while is_not_ok(temp) and realidx < lim and steps < steps_num:
        #     realidx += 1
        #     temp = df.values[realidx, 2]
        #     steps += 1
        # if (realidx == lim) or steps == steps_num:
        #     real_curs.append(-1)
        #     realidx = -1
        # else:
        #     real_curs.append(df.index[realidx])
        if is_not_ok(df.values[realidx,2]):
            assert False, f'real idx is {realidx}'
        curs.append(df.index[realidx])
    raw_end = df.loc[curs]
    assert len(raw_end) == len(raw_start)
    raw_ranges = list(zip(raw_start.values[:, 2], raw_end.values[:, 2]))
    functions_raw = [('\n'.join(data[int(begin):int(end)]))
                      for idx, (begin, end) in enumerate(raw_ranges)]
    if '\\' in filename:
        separting_string = '\\tokenized1\\'
    else:
        separting_string = '/tokenized1/'
    rootpath, realfilename = filename.split(separting_string)  # parse.parse(f'{{}}{os.sep}tokenized1{os.sep}{{}}', filename)
    gt_values = []
    filenames = []
    vulnerabilities = []
    for (begin, _) in raw_ranges:
        indices = (gt['nMethod_Line'] == int(begin)+1) & ("\\"+realfilename.replace('.tree-viewer.txt', '') == gt['nFile_Name'])
        possibble = gt.loc[indices, 'qName'].values  # nMethod_Line
        possibble = set(possibble)
        possibble = [poss.lower() for poss in possibble]
        if len(possibble) > 0:  # int(begin+1) in set(possibble):
            if keywords is None or isin(keywords, possibble):
                gt_values.append(1)
                vulnerabilities.append(','.join(set(possibble)))
            else:
                gt_values.append(0)
                vulnerabilities.append('')
            # type_of_vurn = gt.loc[ & ("\\"+realfilename.replace('.tree-viewer.txt', '') == gt['nFile_Name']), 'qName'].values
        else:
            gt_values.append(0)
            vulnerabilities.append('')
        filenames.append(filename)
    ok = [((len(functions_raw[idx].split("\n")) >= min_token_count >-1) or min_token_count == -1) and (list_of_tokens is None or is_in(l.split(" "), list_of_tokens)) for idx, l in enumerate(functions_list)]
    return filter(ok, functions_list), filter(ok, zipped), filter(ok,list(raw_end.values[:, 2])), \
           filter(ok,gt_values), "", filter(ok,filenames), filter(ok,vulnerabilities), filter(ok, list(raw_start.values[:, 2]))


def intify(l):
    return [(int(item[0]), int(item[1])) for item in l]


def is_not_ok(temp):
    if isinstance(temp, str):
        temp = int(temp)
    return temp is None or math.isnan(temp)


def filter(ok, array):
    return [b for a,b in zip(ok, array) if a]


def create_functions_list_from_filenames_list(files_list, output_folder, core_count, input_folder, security_keywords=None, min_token_count=-1, list_of_tokens=None):
    functions_list = []
    raw_list = []
    gt_values = []
    filenames_list = []
    all_vulnerabilities = []
    all_start_raw = []
    tokenized = []
    sizecounter = len(files_list)
    with open(join(output_folder, 'error_parsing.txt'), 'w+') as f:
        gt = pd.read_csv(os.path.join(input_folder, 'results1.csv'), engine='python', encoding='utf8', error_bad_lines=False)
        with tqdm(total=sizecounter, unit='files') as pbar:
            if core_count > 1:
                with multiprocessing.Pool(processes=core_count) as p:
                    for i, (temp, temp_tokenized, temp_raw, temp_gt, code, filenames, vulnerabilities, start_raw) in (enumerate(p.imap(create_functions_list_from_filename, [(file_name, gt, security_keywords, min_token_count, list_of_tokens) for file_name in files_list], chunksize=10))):
                        functions_list, gt_values, raw_list, filenames_list, all_vulnerabilities, all_start_raw, tokenized= inner_loop(code, f, filenames, functions_list, gt_values, pbar, raw_list,
                                                                                         temp, temp_gt, temp_raw, filenames_list,
                                                                                         vulnerabilities, all_vulnerabilities,
                                                                                                             start_raw, all_start_raw, temp_tokenized, tokenized)
            else:
                for i, filename in enumerate(files_list):
                    data = create_functions_list_from_filename((filename, gt, security_keywords, min_token_count, list_of_tokens))
                    temp, temp_tokenized, temp_raw, temp_gt, code, filenames, vulnerabilities, start_raw = data
                    functions_list, gt_values, raw_list, filenames_list, all_vulnerabilities, all_start_raw, tokenized = inner_loop(code, f, filenames, functions_list, gt_values, pbar, raw_list,
                                                                                     temp, temp_gt, temp_raw, filenames_list,
                                                                                     vulnerabilities, all_vulnerabilities,
                                                                                                          start_raw,
                                                                                                          all_start_raw, temp_tokenized, tokenized)
                    
                    
                    #for j, function in enumerate(functions_list):
                    #    print(filename, j)
    
    return functions_list, raw_list, gt_values, filenames_list, all_vulnerabilities, all_start_raw, tokenized


def inner_loop(error_code, f, filenames, functions_list, gt_values, pbar, raw_list, temp, temp_gt, temp_raw,
               filenames_list, vulnerabilities, all_vulnerabilities, start_raw, all_start_raw, temp_tokenized, tokenized):
    if len(start_raw) > 0:
        all_start_raw += start_raw
        functions_list += temp
        tokenized += temp_tokenized
        raw_list += temp_raw
        gt_values += temp_gt
        filenames_list += filenames
        all_vulnerabilities += vulnerabilities
    if error_code != "":
        f.write(f'{filenames[0]}: {error_code}\n')
    # pbar.update(sizecounter[file_idx])
    pbar.update()
    return functions_list, gt_values, raw_list, filenames_list, all_vulnerabilities, all_start_raw, tokenized


def vectorize_text(text, vectorizer):
    bow_matrix = vectorizer.fit_transform(text)

    return vectorizer, bow_matrix


def get_filenames(mypath):
    mypath = os.path.join(mypath, 'tokenized1')
    filenames = [join(mypath, f) for f in os.listdir(mypath) if isfile(join(mypath, f))]
    return filenames


def vectorize_folder(path, params, vectorizer, output_folder, core_count, input_folder, security_keywords, min_token_count, list_of_tokens):
    files_list = get_filenames(path)
    functions_list, raw_list, gt_values, filenames_list, all_vulnerabilities, all_start_raw, tokenized = create_functions_list_from_filenames_list(files_list[params.files_limit_start:params.files_limit_end], output_folder,
                                                                                                    core_count, input_folder, security_keywords, min_token_count, list_of_tokens)
    vectorizer, bow_matrix = vectorize_text(functions_list, vectorizer)

    return vectorizer, np.array(tokenized), bow_matrix, np.array(raw_list), np.array(gt_values), np.array(filenames_list), np.array(all_vulnerabilities), np.array(all_start_raw)


