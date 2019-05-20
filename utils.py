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


def get_all_needed_inputs(output_folder, cores_to_use, input_folder, vectorizer,
                          max_features, ngram_range=1, files_limit=100, security_keywords=None,
                          min_token_count=-1, list_of_tokens=None):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if cores_to_use == -1:
        cores_to_use = multiprocessing.cpu_count()
    print(f'using {cores_to_use} cores')
    #mypath = join(input_folder, 'tokenized1')
    mypath = input_folder
    vectorizer = {'count': CountVectorizer, 'tfidf': TfidfVectorizer}[vectorizer]
    vectorizer = vectorizer(max_df=1.0, min_df=1, max_features=max_features, ngram_range=(1, ngram_range), vocabulary=list_of_tokens)
    vectorizer1, lists, bow_matrix, raw_lists, gt_values, filenames_list, all_vulnerabilities, all_start_raw = vectorize_folder(mypath, files_limit,
                                                                                            vectorizer, output_folder,
                                                                                            cores_to_use,
                                                                                            input_folder,
                                                                                            security_keywords,
                                                                                            min_token_count, list_of_tokens)
    # if params.matix_form == '0-1':
    #     assert params.vectorizer == 'count'
    #     bow_matrix[bow_matrix >= 1.] = 1
    #     # bow_matrix[bow_matrix == 0.] = 0
    #     # bow_matrix = bow_matrix.astype(int)
    # elif params.matix_form == 'tf-idf':
    #     bow_matrix = bow_matrix * 1. / bow_matrix.sum(axis=1)[:, None]
    return bow_matrix, gt_values, lists, raw_lists, vectorizer1, filenames_list, all_vulnerabilities, all_start_raw


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
        return [],[],[], f'{e}', [filename], [], []
    original_df = copy.deepcopy(df)
    df = df[df[0].notnull()]
    if len(df.index) == 0:
        return [], [],[],  f'no functions found!', [filename], [], []
    # df = create_functions_list_from_df(df)
    starters = df.loc[df[0] == "BEGIN_METHOD"]
    enders = df.loc[df[0] == "END_METHOD"]
    if len(starters) != len(enders):
        return [], [],[], f'has different number of start and end in parsed!!!', [filename], [], []
    if len(starters) == 0 or len(enders) == 0:
        return [], [],[],  f'no functions found!', [filename], [], []
    zipped = list(zip(starters.index, enders.index))
    functions_list = [df[0].iloc[begin:end+1].str.cat(sep=' ') for begin, end in zipped]
    def filter_alpha(stri):
        return ''.join(c for c in stri if c.isalnum() or c == ' ')
    functions_list = [filter_alpha(fl.lower()) for fl in functions_list]
    # # functions_list = [function for function in functions_list if len(function.replace("\n", "")) > 0]
    raw_start = df.loc[starters.index+1]
    # raw_end = df.loc[enders.index-1]
    # df[2] = pd.to_numeric(df[2])
    curs = []
    real_curs = []
    for idx in range(len(enders.index)):
        cur = enders.index[idx]
        realidx = list(df.index).index(cur)
        temp = df.values[realidx, 2]
        steps = 0
        steps_num = 7
        while is_not_ok(temp) and realidx < len(df.values)-1 and steps < steps_num:
            realidx += 1
            temp = df.values[realidx, 2]
            steps += 1
        if (is_not_ok(temp) and realidx == len(df.values)-1) or steps == steps_num:
            real_curs.append(-1)
        else:
            real_curs.append(df.index[realidx])
        curs.append(df.index[realidx])
    raw_end = df.loc[curs]
    assert len(raw_end) == len(raw_start)
    raw_ranges = list(zip(raw_start.values[:, 2], raw_end.values[:, 2]))
    functions_raw = [('\n'.join(data[int(begin):int(end)] if real_curs[idx] > -1 else data[int(begin):]))
                      for idx, (begin, end) in enumerate(raw_ranges)]
    if '\\' in filename:
        separting_string = '\\tokenized1\\'
    else:
        separting_string = '/tokenized1/'
    rootpath, realfilename = filename.split(separting_string)  # parse.parse(f'{{}}{os.sep}tokenized1{os.sep}{{}}', filename)
    gt_values = []
    filenames = []
    vulnerabilities = []
    for (begin, end) in raw_ranges:
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
    return filter(ok, functions_list), filter(ok,functions_raw), \
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
    sizecounter = len(files_list)
    with open(join(output_folder, 'error_parsing.txt'), 'w+') as f:
        gt = pd.read_csv(os.path.join(input_folder, 'results1.csv'), engine='python', encoding='utf8', error_bad_lines=False)
        with tqdm(total=sizecounter, unit='files') as pbar:
            if core_count > 1:
                with multiprocessing.Pool(processes=core_count) as p:
                    for i, (temp, temp_raw, temp_gt, code, filenames, vulnerabilities, start_raw) in (enumerate(p.imap(create_functions_list_from_filename, [(file_name, gt, security_keywords, min_token_count, list_of_tokens) for file_name in files_list], chunksize=10))):
                        functions_list, gt_values, raw_list, filenames_list, all_vulnerabilities, all_start_raw= inner_loop(code, f, filenames, functions_list, gt_values, pbar, raw_list,
                                                                                         temp, temp_gt, temp_raw, filenames_list,
                                                                                         vulnerabilities, all_vulnerabilities,
                                                                                                             start_raw, all_start_raw)
            else:
                for i, filename in enumerate(files_list):
                    temp, temp_raw, temp_gt, code, filenames, vulnerabilities, start_raw = create_functions_list_from_filename((filename, gt, security_keywords, min_token_count, list_of_tokens))
                    functions_list, gt_values, raw_list, filenames_list, all_vulnerabilities, all_start_raw = inner_loop(code, f, filenames, functions_list, gt_values, pbar, raw_list,
                                                                                     temp, temp_gt, temp_raw, filenames_list,
                                                                                     vulnerabilities, all_vulnerabilities,
                                                                                                          start_raw,
                                                                                                          all_start_raw)
                    
                    
                    #for j, function in enumerate(functions_list):
                    #    print(filename, j)
    
    return functions_list, raw_list, gt_values, filenames_list, all_vulnerabilities, all_start_raw


def inner_loop(error_code, f, filenames, functions_list, gt_values, pbar, raw_list, temp, temp_gt, temp_raw,
               filenames_list, vulnerabilities, all_vulnerabilities, start_raw, all_start_raw):
    if len(start_raw) > 0:
        all_start_raw += start_raw
        functions_list += temp
        raw_list += temp_raw
        gt_values += temp_gt
        filenames_list += filenames
        all_vulnerabilities += vulnerabilities
    if error_code != "":
        f.write(f'{filenames[0]}: {error_code}\n')
    # pbar.update(sizecounter[file_idx])
    pbar.update()
    return functions_list, gt_values, raw_list, filenames_list, all_vulnerabilities, all_start_raw


def vectorize_text(text, vectorizer):
    bow_matrix = vectorizer.fit_transform(text)

    return vectorizer, bow_matrix


def get_filenames(mypath):
    mypath = os.path.join(mypath, 'tokenized1')
    filenames = [join(mypath, f) for f in os.listdir(mypath) if isfile(join(mypath, f))]
    return filenames


def vectorize_folder(path, limit, vectorizer, output_folder, core_count, input_folder, security_keywords, min_token_count, list_of_tokens):
    files_list = get_filenames(path)
    functions_list, raw_list, gt_values, filenames_list, all_vulnerabilities, all_start_raw = create_functions_list_from_filenames_list(files_list[:limit], output_folder,
                                                                                                    core_count, input_folder, security_keywords, min_token_count, list_of_tokens)
    vectorizer, bow_matrix = vectorize_text(functions_list, vectorizer)

    return vectorizer, np.array(functions_list), bow_matrix, np.array(raw_list), np.array(gt_values), np.array(filenames_list), np.array(all_vulnerabilities), np.array(all_start_raw)


