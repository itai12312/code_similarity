# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:40:11 2019

@author: adarw
"""

import argparse
import itertools
import traceback

import pandas as pd
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import os
from os import listdir
from os.path import isfile, join
import pathos.multiprocessing as multiprocessing
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def filter_type(x):
    return isinstance(x, (int, float))


def main(args=None):
    print("debug0")
    parser = get_parser()
    print("debug1")
    params = parser.parse_args(args=args)
    print(params.input_folder)
    #mypath = join(params.input_folder, 'tokenized1')
    mypath = "C:\\temp\\C# code\\tokenized2"
    print("debug2")
    vectorizer1, lists = vectorize_folder(mypath, params.files_limit, params.max_features)
    n_lists = [l.lower().split(" ") for l in lists]
    # embeddings = [vectorizer1.transform(l) for l in n_lists]
    print("debug3")
    #print(lists)
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
    print("debug4")
    outputFolder = "C:\\temp\\C# code\\output1"
    all_vocab = list(model.wv.vocab.keys())
    os.makedirs(outputFolder, exist_ok=True)
    print("debug41")
    print(model.wv.vocab.keys())
    #print(model.wv.most_similar("namespace"))
    print("debug411")
    #print(model.wv.doesnt_match("Argument case evt".split()))
    print("debug42")
    with open(outputFolder + "\\common_words.txt",'w') as f:
        print("debug43")
        f.write(f'most common words: ' + str(all_vocab[:5])+f'\n')
        f.write(f'model.wv.doesnt_match for the words above\n')
        f.write(f'{model.wv.doesnt_match(all_vocab[:5])}\n\n')
        print("debug44")
        f.write(f'Similarities:\n')
        f.write(all_vocab[0]+f'\n')
        f.write(f'{model.wv.most_similar(all_vocab[0])}\n')
        f.write(all_vocab[5]+f'\n')
        f.write(f'{model.wv.most_similar(all_vocab[5])}\n')
        f.write(all_vocab[12]+f'\n')
        f.write(f'{model.wv.most_similar(all_vocab[12])}\n\n')
        print("debug45")
        f.write(all_vocab[-2] + f' ' + all_vocab[-1] + f' similarity\n')
        f.write(f'{model.wv.similarity(all_vocab[-2], all_vocab[-1])}\n\n')
        print("debug46")
        f.write(f'5 most similar between ' + all_vocab[-3] + f' (positive) and ' + all_vocab[-4] + f' (negative)\n')
        f.write(f'{model.wv.most_similar(positive=[all_vocab[-3]], negative=[all_vocab[-4]], topn=5)}\n\n')
    print("debug5")
    tsnescatterplot(params.output_folder, model, [], {"Secure": list(model.wv.vocab.keys())})
    word_to_vec_plt(lists, ['Secure' for item in lists], model, params.output_folder)
    print("debug6")


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
    create_functions_list_from_df(df)


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
        if counter%1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[counter] = make_feature_vec(review, model, num_features)
        counter = counter + 1.
    return reviewFeatureVecs


def text_to_vec(text, model, i):
    c = 0
    v = np.array([0.0]*300)
    for sent in text:
        for word in sent:
            if word in model:
                if word in model:
                    v += model[word]
                    c += 1
    return v/c if c > 0 else v


def word_to_vec_plt(reduced_results, y, model, output_folder):
    features = np.array([text_to_vec(reduced_results[i], model, i) for i in range(len(reduced_results))])
    # y = all_lyrics[:lim]["genre"]
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(list(set(y)))
    y = le.transform(y)
    plott(features, y, RandomForestClassifier(n_estimators=100), 'word_to_vec_approach.png', output_folder)


def plott(x, y, model, figname, output_folder):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.14)
    plotting(xtrain, ytrain, xtest, ytest, model, figname, output_folder)


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
    parser.add_argument('--input_folder', action="store", dest="input_folder", help="input_folder", default=None)
    parser.add_argument('--output_folder', action="store", dest="output_folder", help="output_folder", default="")
    parser.add_argument('--max_features', action="store", dest="max_features", type=int, default=200)
    parser.add_argument('--files_limit', action="store", dest="files_limit", type=int, default=60000)
    parser.add_argument('--override', action="store", dest="override", default=True, type=lambda x:x.lower in ['false'])

    parser.add_argument('--num_features', action="store", dest="num_features", type=int, default=3000)
    parser.add_argument('--min_word_count', action="store", dest="min_word_count", type=int, default=20)
    parser.add_argument('--num_workers', action="store", dest="num_workers", type=int, default=4)
    parser.add_argument('--context', action="store", dest="context", type=int, default=10)
    parser.add_argument('--downsampling', action="store", dest="downsampling", type=int, default=1e-3)
    return parser


def create_functions_list_from_df(df):
    df = df[df[0].notnull()]
    starters = df.loc[df[0] == "BEGIN_METHOD"]
    enders = df.loc[df[0] == "END_METHOD"]
    zipped = list(zip(starters.index, enders.index))
    functions_list = []
    for begin, end in zipped:
        functions_list.append(df[0].iloc[begin:end+1].str.cat(sep=' '))
    return functions_list


def create_functions_list_from_filenames_list(files_list):
    functions_list = []
    for filename in files_list:
        try:
            df = pd.read_csv(filename, header = None)
            functions_list += create_functions_list_from_df(df)
        except:
            continue
    return functions_list


def vectorize_text(text, max_features):
    # create the transform
    vectorizer = CountVectorizer(max_features = max_features)
    # build vocab
    vectorizer.fit(text)
    return vectorizer


def get_filenames(mypath):
    filenames = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    return filenames


def vectorize_folder(path, limit, max_features):
    files_list = get_filenames(path)
    functions_list = create_functions_list_from_filenames_list(files_list[:limit])
    vectorizer = vectorize_text(functions_list, max_features)
    return vectorizer, functions_list


if __name__ == "__main__":
    main()
