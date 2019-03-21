import pandas as pd
from os.path import isfile, join
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# import nltk
import itertools
import sklearn
import traceback
import os

def filter_type(x):
    return isinstance(x, (int, float))

def str_ok(stri):
    return len(stri.replace("\n", "")) > 2

def create_functions_list_from_df(filename):
    df = pd.read_csv(filename, header = None, encoding='utf8', error_bad_lines=False)
    df = df[df[0].notnull()]
    starters = df.loc[df[0] == "BEGIN_METHOD"]
    enders = df.loc[df[0] == "END_METHOD"]
    if len(starters) != len(enders):
        print(f'{filename} has different number of start and end in parsed!!!')
    # if len(starters) == 0 or len(enders) == 0:
    #     print(f'no functions found! {filename}')
    #     return [], []
    zipped = list(zip(starters.index, enders.index))
    functions_list = [df[0].iloc[begin:end+1].str.cat(sep=' ') for begin, end in zipped if str_ok(df[0].iloc[begin:end+1].str.cat(sep=' '))]
    # # functions_list = [function for function in functions_list if len(function.replace("\n", "")) > 0]
    # with open(filename.replace("/tokenized1/", "/c_sharp_code/").replace(".tree-viewer.txt", "")) as f:
    #     data = f.read().split("\n")
    # raw_ranges = list(zip(starters.values[:,2], enders.values[:,2]))
    # functions_raw = [data[begin:end].str.cat(sep=' ') if ((not math.isnan(begin) and not math.isnan(end)) and False) else '' for begin, end in raw_ranges if str_ok(df[0].iloc[begin:end+1].str.cat(sep=' '))]
    functions_raw = ['' for i in range(len(functions_list))]
    return functions_list, functions_raw


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
