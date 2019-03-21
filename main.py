import argparse
import traceback

import pandas as pd
from gensim.models import word2vec
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import os
from os import listdir
from os.path import isfile, join
import pathos.multiprocessing as multiprocessing
import numpy as np
import sklearn
from utils import tsnescatterplot, create_functions_list_from_df

def main(args=None):
    parser = get_parser()
    params = parser.parse_args(args=args)
    if params.profiler:
        profile(params)
    else:
        main_(params)


def main_(params):
    if not os.path.exists(params.output_folder):
        os.mkdir(params.output_folder)
    mypath = join(params.input_folder, 'tokenized1')
    vectorizer1, lists, bow_matrix, raw_lists = vectorize_folder(mypath, params.files_limit, params.max_features)
    if params.matix_form == '0-1':
        bow_matrix[bow_matrix > 1] = 1
    elif params.matix_form == 'tf-idf':
        bow_matrix = bow_matrix*1./bow_matrix.sum(axis=1)[:,None]
    metric = {'jaccard': scipy.spatial.distance.jaccard,
              'cosine': scipy.spatial.distance.cosine,
              'euclidean': scipy.spatial.distance.euclidean,
              'cityblock': scipy.spatial.distance.cityblock}[params.metric]
    analyze_functions(bow_matrix, metric, lists, raw_lists, params.output_folder)


def analyze_functions(matrix, metric, lists, raw_lists, output_folder):
    # vfunc = np.vectorize(lambda a:metric(a.toarray(), matrix[0].toarray()), otypes=float)
    # out = vfunc(matrix[1:])
    with open(join(output_folder, 'close_functions.txt'), 'w+') as f:
        for j in range(10):
            idx = get_closest_idx(matrix, metric, j)
            f.write(f'results: input {j}\n')
            f.write(lists[j].replace("\n", "")+"\n")
            f.write(raw_lists[j].replace("\n", "")+"\n")
            f.write(f'closest match: {idx}\n')
            f.write(lists[idx].replace("\n", "")+"\n")
            f.write(raw_lists[idx].replace("\n", "")+"\n")
            pass


def get_closest_idx(matrix, metric, j):
    res = [metric(matrix[i].toarray(), matrix[j].toarray()) for i in range(matrix.shape[0])]
    res = np.array(res)
    results = np.argsort(-res, axis=0)
    idx = list(set(results[:3])-set([j]))[0]
    return idx


class ConstantAray:
    def __init__(self, val, len):
        self.val = val
        self.len = len

    def __getitem__(self, item):
        return self.val

    def __len__(self):
        return self.len


def fit_labels(y):
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(list(set(y)))
    output = le.transform(y)
    return output


def generating_model(n_lists, params):
    model_name = "model1.pkl"
    if params.override or not os.path.exists(model_name):
        print("Training model...")
        model = word2vec.Word2Vec(n_lists, workers=params.num_workers,
                                  size=params.num_features, min_count=params.min_word_count,
                                  window=params.context, sample=params.downsampling)
        # If you don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)
        # It can be helpful to create a meaningful model name and
        # save the model for later use. You can load it later using Word2Vec.load()
        model.save(model_name)
    else:
        model = word2vec.Word2Vec.load(model_name)
    return model


def profile(params):
    import fileinput
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    main_(params)
    pr.disable()
    # cumtime, ncalls
    pr.print_stats(sort='cumtime')


def make_feature_vec(words, model, num_features):
    feature_vec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec,model[word])
    feature_vec = np.divide(feature_vec,nwords)
    return feature_vec


def get_avg_features(reviews, model, num_features):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features), dtype="float32")
    for review in reviews:
        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[counter] = make_feature_vec(review, model, num_features)
        counter = counter + 1.
    return reviewFeatureVecs


def mult_speed_up(func, array):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        # with tqdm(total=len(array)) as pbar:
        # results = []
        # for i, res in tqdm(enumerate(pool.imap(func, array))):  # imap_unordered
        #     pbar.update()
        #     results.append(res)
        results = pool.map(func, array)
    # results = pool.map(func, array)
    # pool.close()
    # pool.join()
    return results


def get_parser():
    parser = argparse.ArgumentParser(description="Main")
    parser.add_argument('--input_folder', action="store", dest="input_folder", help="input_folder", default="../codes/")
    parser.add_argument('--output_folder', action="store", dest="output_folder", help="output_folder", default="results")
    parser.add_argument('--classifier', action="store", dest="classifier", help="randomforest for now", default="randomforest")
    parser.add_argument('--metric', action="store", dest="metric", help="jaccard or cosine", default="jaccard")
    parser.add_argument('--matix_form', action="store", dest="matix_form", help="0-1 or tf-idf or none", default="none")
    parser.add_argument('--max_features', action="store", dest="max_features", type=int, default=100)
    parser.add_argument('--files_limit', action="store", dest="files_limit", type=int, default=100)
    parser.add_argument('--override', action="store", dest="override", default=True, type=lambda x:x.lower not in ['false', '0', 'n'])
    parser.add_argument('--profiler', action="store", dest="profiler", default=False, type=lambda x:x.lower in ['true', '1', 'y'])

    parser.add_argument('--num_features', action="store", dest="num_features", type=int, default=300)
    parser.add_argument('--min_word_count', action="store", dest="min_word_count", type=int, default=40)
    parser.add_argument('--num_workers', action="store", dest="num_workers", type=int, default=4)
    parser.add_argument('--context', action="store", dest="context", type=int, default=10)
    parser.add_argument('--downsampling', action="store", dest="downsampling", type=int, default=1e-3)
    return parser

import numba as nb
@nb.vectorize(target="cpu")
def nb_vf(x):
    return x+2*x*x+4*x*x*x

def create_functions_list_from_filenames_list(files_list):
    functions_list = []
    raw_list = []
    for filename in files_list:
        try:

            temp, temp_raw = create_functions_list_from_df(filename)
            functions_list +=temp
            raw_list+= temp_raw
        except Exception as e:
            print(filename)
            print(e)
            # print(traceback.print_exc())
            continue
    return functions_list, raw_list


def vectorize_text(text, max_features):
    # create the transform
    vectorizer = CountVectorizer(max_features = max_features)
    # build vocab
    bow_matrix = vectorizer.fit_transform(text)
    return vectorizer, bow_matrix


def get_filenames(mypath):
    filenames = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    return filenames


def vectorize_folder(path, limit, max_features):
    files_list = get_filenames(path)
    functions_list, raw_list = create_functions_list_from_filenames_list(files_list[:limit])
    vectorizer, bow_matrix = vectorize_text(functions_list, max_features)
    return vectorizer, functions_list, bow_matrix, raw_list


def main1(lists, params):
    n_lists = [l.lower().split(" ") for l in lists]
    # embeddings = [vectorizer1.transform(l) for l in n_lists]
    embedding_model = generating_model(n_lists, params)
    os.makedirs(params.output_folder, exist_ok=True)
    all_vocab = list(embedding_model.wv.vocab.keys())
    with open(os.path.join(params.output_folder, 'common_words.txt'), 'w+') as f:
        f.write(f'{embedding_model.doesnt_match(all_vocab[:3])}\n')
        f.write(f'{embedding_model.most_similar(all_vocab[0])}\n')
        f.write(f'{embedding_model.wv.similarity(all_vocab[-2], all_vocab[-1])}\n')
        f.write(f'{embedding_model.wv.most_similar(positive=[all_vocab[-3]], negative=[all_vocab[-4]], topn=3)}\n')
    # tsnescatterplot(params.output_folder, embedding_model, [], {"Secure": all_vocab})
    # y = fit_labels(lists)
    model = {'randomforest': RandomForestClassifier(n_estimators=100)}[params.classifier]
    # word_to_vec_plt(lists, ConstantAray(0, len(lists)), embedding_model, params.output_folder, model)

if __name__ == "__main__":
    main()
