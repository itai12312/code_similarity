import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import itertools
import sklearn
import traceback
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import os
from os.path import join
import time
from gensim.models import word2vec

sns.set_style()


def filter_type(x):
    return isinstance(x, (int, float))


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



def tsnescatterplot(output_folder, model, all_words, words_freq_genre):
    # red green blue gray coral brown yellow azure plum pink lime olive
    colors = {'Secure': 'green'}
    words_list = [(word, genre, colors[genre]) for genre in words_freq_genre for word in words_freq_genre[genre]]
    others = list(set(all_words)-set([item[0] for item in words_list]))
    others = [(word, 'non', 'grey') for word in others]
    all_words = np.array([w for w in others + words_list if w[0] in model.wv])
    arrays = np.array([model.wv[word] for word in all_words[:, 0]])
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=50).fit_transform(arrays)
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': all_words[:, 1],
                       'color': all_words[:, 2]})
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker=".",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                  }
                     )
    # Adds annotations one by one with a loop
    # for line in range(0, df.shape[0]):
    #     p1.text(df["x"][line],
    #             df['y'][line],
    #             '  ' + df["words"][line].title(),
    #             horizontalalignment='left',
    #             verticalalignment='bottom', size='medium',
    #             color=df['color'][line],
    #             weight='normal'
    #             ).set_size(15)
    plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
    plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)

    plt.title('t-SNE visualization for genres')
    plt.savefig(os.path.join(output_folder, 'tsne.png'))
    # plt.show()

def plott(x, y, model, figname, output_folder):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.14)
    plotting(xtrain, ytrain, xtest, ytest, model, figname, output_folder)


def text_to_vec(text, model, i):
    c = 0
    v = np.array([0.0]*model[list(model.wv.vocab.keys())[0]].size)
    for sent in text:
        for word in sent:
            if word in model:
                if word in model:
                    v += model[word]
                    c += 1
    return v/c if c > 0 else v


def word_to_vec_plt(reduced_results, y, embedding_model, output_folder, model):
    features = np.array([text_to_vec(reduced_results[i], embedding_model, i) for i in range(len(reduced_results))])
    plott(features, y, model, 'word_to_vec_approach.png', output_folder)


def plotting(X_train, y_train, X_test, y_test, model, figname, output_folder):
    plt.close()
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    # ypredtrain = model.predict(X_train)
    # print('acc for train is {}'.format(sum(ypredtrain==y_train)/len(y_train)))
    confusion = sklearn.metrics.confusion_matrix(y_test, ypred)
    plt.imshow(confusion, interpolation='nearest')
    plt.xlabel('pred')
    plt.ylabel('gt')
    plt.colorbar()

    aaa = list(range(confusion.shape[0]))
    for (j, i) in itertools.product(aaa, aaa):
        plt.text(i, j, confusion[j, i], ha='center', va='center', color='blue')
    classes = sorted(set(y_test))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.savefig(join(output_folder, figname))
    # plt.show()
    with open(join(output_folder, 'f1_score_{}.txt'.format(figname)), 'w+') as f:
        f.write('acc for test is {}\n'.format(np.trace(confusion)/np.sum(confusion, axis=(1,0))))
        f.write('class id, precision, recall, f1 for {}\n'.format(figname))
        for class_id in range(len(confusion)):
            try:
                precision = confusion[class_id, class_id]/sum(confusion[:, class_id]) if sum(confusion[:, class_id]) >0 else 0
                recall = confusion[class_id, class_id]/sum(confusion[class_id, :]) if sum(confusion[class_id, :]) >0 else 0
                f.write('{}, {}, {}, {}\n'.format(class_id, precision, recall, 2*precision*recall/(precision+recall) if precision+recall >0 else 0))
            except Exception as e:
                print(e)
                print(traceback.print_exc())
    plt.close()
    return np.where(y_test != ypred)



def main2(params):
    df = pd.read_csv(join(params.input_folder,
                          'tokenized1/084_update_quality_minmax_sizeFixture.cs.tree-viewer.txt'), header=None)
    df = df[df[0].notnull()]
    df.applymap(filter_type)
    matrix = CountVectorizer(max_features=10)
    X = matrix.fit_transform(df[0]).toarray()
    print(matrix.vocabulary_)
    print(matrix.get_params())
    df[0].iloc[0:10].str.cat(sep=' ')
    starters = df.loc[df[0] == "BEGIN_METHOD"]
    enders = df.loc[df[0] == "END_METHOD"]
    zipped = list(zip(starters.index, enders.index))
    functions_list = []
    for begin, end in zipped:
        functions_list.append(df[0].iloc[begin:end+1].str.cat(sep=' '))
    # create_functions_list_from_df(df)


def fix_blocks_and_methods(input_df):
    df = input_df

    # Create a column with the lines of the PREVIOUS token
    df[4] = df[2].shift(1)
    df[4] = np.where(np.isnan(df[4]),df[4].shift(1),df[4]) # Fix "still NaN" values
    df[4] = np.where(np.isnan(df[4]),df[4].shift(1),df[4]) # Fix "still NaN" values
    df[4] = np.where(np.isnan(df[4]),df[4].shift(1),df[4]) # Fix "still NaN" values
    df[4] = np.where(np.isnan(df[4]),df[4].shift(1),df[4]) # Fix "still NaN" values
    df[4] = np.where(np.isnan(df[4]),df[4].shift(1),df[4]) # Fix "still NaN" values
    df[4] = np.where(np.isnan(df[4]),df[4].shift(1),df[4]) # Fix "still NaN" values

    # Copy the line number to the END_BLOCK and END_METHOD
    df[2] = np.where(df[0]=="END_BLOCK",np.where(np.isnan(df[2]),df[4],df[2]),df[2])
    df[2] = np.where(df[0]=="END_METHOD",np.where(np.isnan(df[2]),df[4],df[2]),df[2])

    # Create a column with the lines of the NEXT token
    df[4] = df[2].shift(-1)
    df[4] = np.where(np.isnan(df[4]),df[4].shift(-1),df[4]) # Fix "still NaN" values
    df[4] = np.where(np.isnan(df[4]),df[4].shift(-1),df[4]) # Fix "still NaN" values
    df[4] = np.where(np.isnan(df[4]),df[4].shift(-1),df[4]) # Fix "still NaN" values
    df[4] = np.where(np.isnan(df[4]),df[4].shift(-1),df[4]) # Fix "still NaN" values
    df[4] = np.where(np.isnan(df[4]),df[4].shift(-1),df[4]) # Fix "still NaN" values
    df[4] = np.where(np.isnan(df[4]),df[4].shift(-1),df[4]) # Fix "still NaN" values

    # Copy the line number for BEGIN_BLOCK and BEGIN_METHOD
    df[2] = np.where(df[0]=="BEGIN_BLOCK",np.where(np.isnan(df[2]),df[4],df[2]),df[2])
    df[2] = np.where(df[0]=="BEGIN_METHOD",np.where(np.isnan(df[2]),df[4],df[2]),df[2])

    # Set Id for BLOCK/METHOD lines
    df[1] = np.where(df[0]=="BEGIN_METHOD",8001,df[1])
    df[1] = np.where(df[0]=="END_METHOD",8002,df[1])
    df[1] = np.where(df[0]=="BEGIN_BLOCK",8003,df[1])
    df[1] = np.where(df[0]=="END_BLOCK",8004,df[1])

    # Set column to 0 for METHOD/BLOCK lines
    df[3] = np.where(df[0]=="BEGIN_METHOD",0,df[3])
    df[3] = np.where(df[0]=="END_METHOD",0,df[3])
    df[3] = np.where(df[0]=="BEGIN_BLOCK",0,df[3])
    df[3] = np.where(df[0]=="END_BLOCK",0,df[3])

    # Remove helper column
    return(df.drop(columns=[4]))



def plot_confusion_matrix_(confusion, path, log_scale=False, show_amount=False, classes=None):
    plt.close('all')
    results = confusion.astype(int)
    if log_scale:
        results = np.log10(results + 1)
    plt.imshow(results, interpolation='nearest')
    plt.xlabel('gt')
    plt.ylabel('pred')
    plt.title('in {} scale'.format({True: 'log', False: 'regular'}[log_scale]))
    plt.colorbar()
    if classes is not None:
        tick_marks = np.arange(len(classes))
        if show_amount:
            classes_order = list(range(len(classes)))
            for (j, i) in itertools.product(classes_order, classes_order):
                plt.text(i, j, results[j, i], ha='center', va='center', color='blue')
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'confusion.svg'))
    # plt.close('all')
    plt.show()
    results = _calculate_per_class_stats(classes, confusion)
    _dump_per_class_stats(path, results)


def _dump_per_class_stats(folder_path, results):
    columns = ['class_id', 'class_name', 'recall', 'precision', 'f1', 'most-mistaken recall', 'most-mistaken precision']
    data = pd.DataFrame(data=np.array(results), columns=columns)
    data = data.sort_values(by=['recall'])
    data.to_csv('{}.csv'.format(os.path.join(folder_path, 'per_class_report_path')), float_format='%.4f')


def get_most_common_mistaken(arr, class_id, classes):
    results = np.argsort(-arr, axis=0)
    idx = list(set(results[:2])-set([class_id]))[0]
    val = arr[idx] / sum(arr) if sum(arr) > 0 else 0
    class_name = classes[idx] if classes is not None else ''
    return '{}_{:.4f}'.format(class_name, val)


def _calculate_per_class_stats(classes, confusion):
    results = []
    for class_id in range(len(classes)):
        class_name = classes[class_id] if classes is not None else ''
        gt = sum(confusion[class_id, :])
        pred = sum(confusion[:, class_id])
        if gt > 0:
            recall = confusion[class_id, class_id] * 1. / gt if gt > 0 else 0
            precision = confusion[class_id, class_id] * 1. / pred if pred > 0 else 0
            f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0
            results.append([class_id, class_name, recall, precision, f1,
                            get_most_common_mistaken(confusion[:, class_id], class_id, classes),  # for precision
                            get_most_common_mistaken(confusion[class_id, :], class_id, classes)])  # for recall
    return results


def display_topics(model, feature_names, params):
    with open(os.path.join(params.output_folder, 'topic_modelling.txt'), 'w+') as f:
        for topic_idx, topic in enumerate(model.components_):
            f.write("Topic %d: {}\n" .format(topic_idx))
            f.write("".join([f'{feature_names[i]} {topic[i]}\n' for i in topic.argsort()[:-params.no_top_words - 1:-1]]))
            f.write("\n")


def run_lda(params):
    # sns.clustermap(matrix.toarray())
    # sns.clustermap(matrix, metric=params.metric, method=params.clustering_method, cmap="Blues", standard_scale=1)
    # plt.tight_layout()
    # plt.savefig(os.path.join(params.output_folder, 'dendogram_with_heatmap.svg'))
    # plt.show()
    # df = pd.DataFrame.from_dict({'content': raw_lists, 'target': gt_values}, orient='columns')
    # data = df.content.values.tolist()
    assert params.vectorizer == 'count'
    # lda = LatentDirichletAllocation(n_topics=params.n_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=params.seed, n_jobs=-1).fit(matrix1)
    # # vectorizer.get_feature_names() vs vocab?
    # # tfidf = matrix1.toarray() * 1. / matrix1.toarray().sum(axis=1)[:, None]
    # # nmf = NMF(n_components=params.n_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    # # display_topics(nmf, vectorizer.get_feature_names(), no_top_words)
    #
    # display_topics(lda, vectorizer.get_feature_names(), params)
    # clf = {'randomforest': RandomForestClassifier(n_estimators=100, random_state=params.seed)}[params.classifier]
    # clf.fit(matrix, gt_values)
    # pred = clf.predict(matrix)
    # confusion = confusion_matrix(gt_values, pred)
    # plot_confusion_matrix_(confusion, params.output_folder, show_amount=True, classes=['secure', 'not secure'])
    # plt.figure(figsize=(10, 7))


def analyze_functions(matrix, metric, lists, raw_lists, vocab, params, gt_values):
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
