import time
import argparse
import traceback
import itertools
import pandas as pd
from gensim.models import word2vec
import scipy
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
from os import listdir
from os.path import isfile, join
import pathos.multiprocessing as multiprocessing
import numpy as np
import sklearn
from utils import tsnescatterplot, create_functions_list_from_filename, plot_confusion_matrix_
from tqdm import tqdm, trange
# tqdm.auto
import sys
import itertools
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import copy
sns.set()

METRIC_FUNCTIONS = {'jaccard': scipy.spatial.distance.jaccard,
                    'cosine': scipy.spatial.distance.cosine,
                    'euclidean': scipy.spatial.distance.euclidean,
                    'cityblock': scipy.spatial.distance.cityblock}


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


def str_to_params(args):
    parser = get_parser()
    params = parser.parse_args(args=args)
    return params


def main_(params):
    # can be called using dictobj.DictionaryObject({'metric': 'euclidean'}) or
    # str_to_params('--output_folder result3 --metric euclidean --input_folder ../codes_short/ --files_limit 100 --max_features 2000')
    bow_matrix, gt_values, lists, raw_lists, vocab, vectorizer, filenames_list = get_all_needed_inputs(params)
    # intersting_indices = analyze_functions(bow_matrix, METRIC_FUNCTIONS[params.metric], lists, raw_lists,
    #                   vocab, params, gt_values)
    intersting_indices = np.array(list(range(len(lists))))
    analyze_functions2(bow_matrix[intersting_indices], lists[intersting_indices], raw_lists[intersting_indices],
                       vocab, params, gt_values[intersting_indices], vectorizer, filenames_list[intersting_indices])


def get_all_needed_inputs(params):
    if not os.path.exists(params.output_folder):
        os.mkdir(params.output_folder)
    if params.cores_to_use == -1:
        params.cores_to_use = multiprocessing.cpu_count()
    print(f'using {params.cores_to_use} cores')
    mypath = join(params.input_folder, 'tokenized1')
    vectorizer = {'count': CountVectorizer, 'tfidf': TfidfVectorizer}[params.vectorizer]
    vectorizer = vectorizer(max_df=0.95, min_df=2, max_features=params.max_features, ngram_range=(1, params.ngram_range))
    vectorizer1, lists, bow_matrix, raw_lists, gt_values, filenames_list = vectorize_folder(mypath, params.files_limit,
                                                                            vectorizer, params.output_folder,
                                                                            params.cores_to_use,
                                                                            params.input_folder)
    # if params.matix_form == '0-1':
    #     assert params.vectorizer == 'count'
    #     bow_matrix[bow_matrix >= 1.] = 1
    #     # bow_matrix[bow_matrix == 0.] = 0
    #     # bow_matrix = bow_matrix.astype(int)
    # elif params.matix_form == 'tf-idf':
    #     bow_matrix = bow_matrix * 1. / bow_matrix.sum(axis=1)[:, None]
    vocab = list(vectorizer1.vocabulary_.keys())
    return bow_matrix, gt_values, lists, raw_lists, vocab, vectorizer1, filenames_list


def analyze_functions2(matrix1, lists, raw_lists, vocab, params, gt_values, vectorizer, filenames):
    # distances = [[metric(matrix[i].toarray(), matrix[j].toarray()) for i in range(matrix.shape[0])] for j in range(matrix.shape[0])]
    # distances = np.array(distances)
    if params.vectorizer == 'count' and params.matrix_form == 'tfidf':
        matrix = matrix1.toarray() * 1. / matrix1.toarray().sum(axis=1)[:, None]
    elif params.vectorizer == 'count' and params.matrix_form == '0-1':
        matrix = copy.deepcopy(matrix1)
        matrix[matrix >= 1.] = 1
    # matrix = matrix.toarray()
    distances = pdist(matrix, metric=params.metric)
    # distances = sklearn.metrics.pairwise_distances(matrix.toarray(), metric=params.metric)
    # fig = plt.figure(figsize=(25, 10))

    lnk = linkage(distances, params.clustering_method)
    # TODO:get list of filenames and locations!!
    cluster = AgglomerativeClustering(n_clusters=params.n_clusters, affinity=params.metric, linkage=params.clustering_method)
    results = cluster.fit_predict(matrix)
    colors = ['blue', 'orange', 'olive', 'green', 'cyan', 'brown', 'purple', 'pink', 'red']
    # fl = fcluster(lnk, params.n_clusters,criterion='maxclust')
    # assert fl == results
    link_cols = {}
    dflt_col = "#808080"   # Unclustered gray
    for i, i12 in enumerate(lnk[:, :2].astype(int)):
        c1, c2 = (link_cols[x] if x > len(lnk) else colors[results[x]] for x in i12)
        link_cols[i+1+len(lnk)] = c1 if c1 == c2 else dflt_col

    def get_color(k):
        return link_cols[k]
    plt.close('all')
    plt.title('clustering method {}, metric {}'.format(params.clustering_method, params.metric))
    z = dendrogram(lnk, labels=gt_values, color_threshold=None, link_color_func=get_color)
    # z = dendrogram(lnk, labels=list(range(len(gt_values))))
    with open(os.path.join(params.output_folder, 'dendogram_list.txt'), 'w+') as f:
        f.write(f'order of leaves is {z["leaves"]}\n')
        f.write(f'names of files is {filenames}\n')
    plt.ylim(0, 5.5)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(params.output_folder, 'dendogram.svg'))
    plt.show()
    # sns.clustermap(matrix.toarray())
    sns.clustermap(matrix, metric=params.metric, method=params.clustering_method, cmap="Blues", standard_scale=1)
    # plt.tight_layout()
    plt.savefig(os.path.join(params.output_folder, 'dendogram_with_heatmap.svg'))
    plt.show()
    # df = pd.DataFrame.from_dict({'content': raw_lists, 'target': gt_values}, orient='columns')
    # data = df.content.values.tolist()
    assert params.vectorizer == 'count'

    lda = LatentDirichletAllocation(n_components=params.n_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=params.seed, n_jobs=-1).fit(matrix1)

    # vectorizer.get_feature_names() vs vocab?
    # tfidf = matrix1.toarray() * 1. / matrix1.toarray().sum(axis=1)[:, None]
    # nmf = NMF(n_components=params.n_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    # display_topics(nmf, vectorizer.get_feature_names(), no_top_words)

    display_topics(lda, vectorizer.get_feature_names(), params)
    clf = {'randomforest': RandomForestClassifier(n_estimators=100, random_state=params.seed)}[params.classifier]
    clf.fit(matrix, gt_values)
    pred = clf.predict(matrix)
    confusion = confusion_matrix(gt_values, pred)
    plot_confusion_matrix_(confusion, params.output_folder, show_amount=True, classes=['secure', 'not secure'])
    # plt.figure(figsize=(10, 7))
    # plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
    pass
def display_topics(model, feature_names, params):
    with open(os.path.join(params.output_folder, 'topic_modelling.txt'), 'w+') as f:
        for topic_idx, topic in enumerate(model.components_):
            f.write("Topic %d: {}\n" .format(topic_idx))
            f.write("".join([f'{feature_names[i]} {topic[i]}\n' for i in topic.argsort()[:-params.no_top_words - 1:-1]]))
            f.write("\n")


def analyze_functions(matrix, metric, lists, raw_lists, vocab, params, gt_values):
    # vfunc = np.vectorize(lambda a:metric(a.toarray(), matrix[0].toarray()), otypes=float)
    # out = vfunc(matrix[1:])
    cur_time = time.time()
    if not os.path.exists(join(params.output_folder, 'samples')):
        os.mkdir(join(params.output_folder, 'samples'))
    all_indices = set()
    with open(join(params.output_folder, 'close_functions.txt'), 'w+') as f:
        for j in range(params.top_similar_functions):
            confusion, idx, score, j = get_closest_function(j, matrix, metric)
            f.write(f'results1: input {j} with score {score}\n')
            f.write(f'{confusion}\n')
            f.write(lists[j].replace("\n", "")+"\n")
            f.write(f'closest match: {idx}\n')
            f.write(lists[idx].replace("\n", "")+"\n")
            with open(join(params.output_folder, 'samples', 'close_functions_{}_input.txt'.format(j)), 'w+') as f1:
                f1.write(raw_lists[j])
                # f1.write(raw_lists[j].replace("\n", "")+"\n")
            with open(join(params.output_folder, 'samples', 'close_functions_{}_closest.txt'.format(j)), 'w+') as f2:
                f2.write(raw_lists[idx])
                #f2.write(raw_lists[idx].replace("\n", "")+"\n")
            all_indices.update([idx, j])
    print(f'analysis took {time.time()-cur_time} seconds')
    return np.array(all_indices)


def get_closest_function(j, matrix, metric):
    idx, score = get_closest_idx(matrix, metric, j)
    assert j != idx
    confusion = {}
    for var1, var2 in itertools.product(range(2), range(2)):
        indices = np.where((matrix[j].toarray() == var1) & (matrix[idx].toarray() == var2))
        confusion[(var1, var2)] = len(indices[1])
    return confusion, idx, score, j


def get_closest_idx(matrix, metric, j):
    res = [metric(matrix[i].toarray(), matrix[j].toarray()) for i in range(matrix.shape[0])]
    res = np.array(res)
    results = np.argsort(res, axis=0)
    idx = list(set(results[:2])-set([j]))[0]
    return idx, res[idx]


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
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput
    # with PyCallGraph(output=GraphvizOutput()):
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
        # results1 = []
        # for i, res in tqdm(enumerate(pool.imap(func, array))):  # imap_unordered
        #     pbar.update()
        #     results1.append(res)
        results = pool.map(func, array)
    # results1 = pool.map(func, array)
    # pool.close()
    # pool.join()
    return results


def get_parser():
    parser = argparse.ArgumentParser(description="Main")
    parser.add_argument('--input_folder', action="store", dest="input_folder", help="input_folder", default="../codes/")
    parser.add_argument('--output_folder', action="store", dest="output_folder", help="output_folder", default="results")
    parser.add_argument('--classifier', action="store", dest="classifier", help="randomforest for now", default="randomforest")
    parser.add_argument('--metric', action="store", dest="metric", help="jaccard or cosine", default="cosine")
    parser.add_argument('--vectorizer', action="store", dest="vectorizer", help="count or tfidf", default="tfidf")
    parser.add_argument('--clustering_method', action="store", dest="clustering_method", help="single complete average ward weighted centroid median", default="average")
    parser.add_argument('--matrix_form', action="store", dest="matrix_form", help="0-1 or tf-idf or none", default="none")
    parser.add_argument('--max_features', action="store", dest="max_features", type=int, default=100)
    parser.add_argument('--files_limit', action="store", dest="files_limit", type=int, default=100)
    parser.add_argument('--override', action="store", dest="override", default=True, type=lambda x:x.lower not in ['false', '0', 'n'])
    parser.add_argument('--profiler', action="store_true", dest="profiler", default=False)  # type=lambda x:x.lower in ['true', '1', 'y']

    parser.add_argument('--num_features', action="store", dest="num_features", type=int, default=300)
    parser.add_argument('--ngram_range', action="store", dest="ngram_range", type=int, default=1)
    parser.add_argument('--n_clusters', action="store", dest="n_clusters", type=int, default=7)
    parser.add_argument('--n_topics', action="store", dest="n_topics", type=int, default=20)
    parser.add_argument('--top_similar_functions', action="store", dest="top_similar_functions", type=int, default=10)
    parser.add_argument('--min_word_count', action="store", dest="min_word_count", type=int, default=40)
    parser.add_argument('--num_workers', action="store", dest="num_workers", type=int, default=4)
    parser.add_argument('--cores_to_use', action="store", dest="cores_to_use", type=int, default=1)
    parser.add_argument('--context', action="store", dest="context", type=int, default=10)
    parser.add_argument('--seed', action="store", dest="seed", type=int, default=0)
    parser.add_argument('--no_top_words', action="store", dest="no_top_words", type=int, default=10)
    parser.add_argument('--downsampling', action="store", dest="downsampling", type=int, default=1e-3)
    return parser


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
    # create the transform
    # build vocab
    bow_matrix = vectorizer.fit_transform(text)
    return vectorizer, bow_matrix


def get_filenames(mypath):
    filenames = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    return filenames


def vectorize_folder(path, limit, vectorizer, output_folder, core_count, input_folder):
    files_list = get_filenames(path)
    functions_list, raw_list, gt_values, filenames_list = create_functions_list_from_filenames_list(files_list[:limit], output_folder, core_count, input_folder)
    vectorizer, bow_matrix = vectorize_text(functions_list, vectorizer)
    return vectorizer, np.array(functions_list), bow_matrix, np.array(raw_list), np.array(gt_values), np.array(filenames_list)


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


# import numba as nb
# @nb.vectorize(target="cpu")
# def nb_vf(x):
#     return x+2*x*x+4*x*x*x

