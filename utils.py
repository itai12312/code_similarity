import pandas as pd
from os.path import isfile, join
import seaborn as sns
# import nltk
import os
import math
import parse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pathos.multiprocessing as multiprocessing
from tqdm import tqdm, trange
import numpy as np


def get_all_needed_inputs(output_folder, cores_to_use, input_folder, vectorizer,
                          max_features, ngram_range=1, files_limit=100):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if cores_to_use == -1:
        cores_to_use = multiprocessing.cpu_count()
    print(f'using {cores_to_use} cores')
    mypath = join(input_folder, 'tokenized1')
    vectorizer = {'count': CountVectorizer, 'tfidf': TfidfVectorizer}[vectorizer]
    vectorizer = vectorizer(max_df=0.95, min_df=2, max_features=max_features, ngram_range=(1, ngram_range))
    vectorizer1, lists, bow_matrix, raw_lists, gt_values, filenames_list = vectorize_folder(mypath, files_limit,
                                                                                            vectorizer, output_folder,
                                                                                            cores_to_use,
                                                                                            input_folder)
    # if params.matix_form == '0-1':
    #     assert params.vectorizer == 'count'
    #     bow_matrix[bow_matrix >= 1.] = 1
    #     # bow_matrix[bow_matrix == 0.] = 0
    #     # bow_matrix = bow_matrix.astype(int)
    # elif params.matix_form == 'tf-idf':
    #     bow_matrix = bow_matrix * 1. / bow_matrix.sum(axis=1)[:, None]
    vocab = list(vectorizer1.vocabulary_.keys())
    return bow_matrix, gt_values, lists, raw_lists, vocab, vectorizer1, filenames_list


def str_ok(stri):
    return len(stri.replace("\n", "")) > 2


def create_functions_list_from_filename(item):
    (filename, gt) = item
    try:
        #  engine='python',
        df = pd.read_csv(filename, header=None, encoding='utf8')  #  error_bad_lines=False
    except Exception as e:
        # print(filename, e)
        return [],[],[], f'{e}', filename
    df = df[df[0].notnull()]
    if len(df.index) == 0:
        return [], [],[],  f'no functions found!', filename
    # df = create_functions_list_from_df(df)
    starters = df.loc[df[0] == "BEGIN_METHOD"]
    enders = df.loc[df[0] == "END_METHOD"]
    if len(starters) != len(enders):
        return [], [],[], f'has different number of start and end in parsed!!!', filename
    if len(starters) == 0 or len(enders) == 0:
        return [], [],[],  f'no functions found!', filename
    zipped = list(zip(starters.index, enders.index))
    functions_list = [df[0].iloc[begin:end+1].str.cat(sep=' ') for begin, end in zipped]
    # # functions_list = [function for function in functions_list if len(function.replace("\n", "")) > 0]
    with open(filename.replace("{0}tokenized1{0}".format(os.sep),
                               "{0}c_sharp_code{0}".format(os.sep)).replace(".tree-viewer.txt", "")) as f:
        data = f.read().splitlines()
    raw_start = df.loc[starters.index+1]
    # raw_end = df.loc[enders.index-1]
    # df[2] = pd.to_numeric(df[2])
    curs = []
    for idx in range(len(enders.index)):
        cur = enders.index[idx]
        realidx = list(df.index).index(cur)
        temp = df.values[realidx, 2]
        while (temp is None or math.isnan(temp)):  # and realidx < len(df.values)-1:
            realidx -= 1
            temp = df.values[realidx, 2]
        curs.append(df.index[realidx])
    raw_end = df.loc[curs]
    assert len(raw_end) == len(raw_start)
    raw_ranges = list(zip(raw_start.values[:,2], raw_end.values[:,2]))
    functions_raw = ['\n'.join(data[int(begin):int(end)])
                     for (begin, end) in raw_ranges]

    rootpath, realfilename = parse.parse('{}/tokenized1/{}', filename)
    gt_values = []
    filenames = []
    for (begin, end) in raw_ranges:
        possibble = gt.loc[(gt['nMethod_Line'] == begin+1) & ("\\"+realfilename.replace('.tree-viewer.txt', '') == gt['nFile_Name']), 'qName'].values  # nMethod_Line
        if len(possibble) > 0: # int(begin+1) in set(possibble):
            gt_values.append(1)
            # type_of_vurn = gt.loc[ & ("\\"+realfilename.replace('.tree-viewer.txt', '') == gt['nFile_Name']), 'qName'].values
        else:
            gt_values.append(0)
        filenames.append(filename)
    return functions_list, functions_raw, gt_values, "", filenames


def create_functions_list_from_filenames_list(files_list, output_folder, core_count, input_folder):
    functions_list = []
    raw_list = []
    gt_values = []
    filenames_list = []
    sizecounter = len(files_list)
    with open(join(output_folder, 'error_parsing.txt'), 'w+') as f:
        gt = pd.read_csv(os.path.join(input_folder, 'results1.csv'), engine='python', encoding='utf8', error_bad_lines=False)
        with tqdm(total=sizecounter, unit='files') as pbar:
            if core_count > 1:
                with multiprocessing.Pool(processes=core_count) as p:
                    for i, (temp, temp_raw, temp_gt, code, filenames) in (enumerate(p.imap(create_functions_list_from_filename, [(file_name, gt) for file_name in files_list], chunksize=10))):
                        functions_list, gt_values, raw_list, filenames_list = inner_loop(code, f, filenames, functions_list, gt_values, pbar, raw_list, temp, temp_gt, temp_raw, filenames_list)
            else:
                for i, filename in enumerate(files_list):
                    temp, temp_raw, temp_gt, code, filenames = create_functions_list_from_filename((filename, gt))
                    functions_list, gt_values, raw_list, filenames_list = inner_loop(code, f, filenames, functions_list, gt_values, pbar, raw_list, temp, temp_gt, temp_raw, filenames_list)
    return functions_list, raw_list, gt_values, filenames_list


def inner_loop(code, f, filenames, functions_list, gt_values, pbar, raw_list, temp, temp_gt, temp_raw, filenames_list):
    functions_list += temp
    raw_list += temp_raw
    gt_values += temp_gt
    filenames_list += filenames
    if code != "":
        f.write(f'{filenames[0]}: {code}\n')
    # pbar.update(sizecounter[file_idx])
    pbar.update()
    return functions_list, gt_values, raw_list, filenames_list


def vectorize_text(text, vectorizer):
    bow_matrix = vectorizer.fit_transform(text)
    return vectorizer, bow_matrix


def get_filenames(mypath):
    filenames = [join(mypath, f) for f in os.listdir(mypath) if isfile(join(mypath, f))]
    return filenames


def vectorize_folder(path, limit, vectorizer, output_folder, core_count, input_folder):
    files_list = get_filenames(path)
    functions_list, raw_list, gt_values, filenames_list = create_functions_list_from_filenames_list(files_list[:limit], output_folder, core_count, input_folder)
    vectorizer, bow_matrix = vectorize_text(functions_list, vectorizer)
    return vectorizer, np.array(functions_list), bow_matrix, np.array(raw_list), np.array(gt_values), np.array(filenames_list)


