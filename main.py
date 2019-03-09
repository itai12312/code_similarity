import argparse
import pandas as pd
from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
import os
from os import listdir
from os.path import isfile, join
import pathos.multiprocessing as multiprocessing
import numpy as np
import sklearn
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


def main(args=None):
    parser = get_parser()
    params = parser.parse_args(args=args)
    df = pd.read_csv(join(params.input_folder,
                          'tokenized1/084_update_quality_minmax_sizeFixture.cs.tree-viewer.txt'), header=None)
    # list(df[0])
    df = df[df[0].notnull()]
    df.applymap(lambda x: isinstance(x, (int, float)))
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
    files_list1 =['tokenized1/084_update_quality_minmax_sizeFixture.cs.tree-viewer.txt',
                  'tokenized1/084_update_quality_minmax_size.cs.tree-viewer.txt',
                  'tokenized1/085_expand_transmission_urlbase.cs.tree-viewer.txt',
                  'tokenized1/085_expand_transmission_urlbaseFixture.cs.tree-viewer.txt',
                  'tokenized1/086_pushbullet_device_ids.cs.tree-viewer.txt']
    files_list1 = [join(params.input_folder, file_name) for file_name in files_list1]
    functions_list1 = create_functions_list_from_filenames_list(files_list1)
    # vectorize(functions_list1).vocabulary_
    mypath = join(params.input_folder, 'tokenized1')
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    create_functions_list_from_filenames_list(onlyfiles[0:5])
    vectorizer1, lists = vectorize_folder(mypath, 5, params.max_features)
    n_lists = [l.split(" ") for l in lists]
    vecs = [vectorizer1.transform(l) for l in n_lists]
    vectorizer1.vocabulary_
    model_name = "model1.pkl"
    if params.override or not os.path.exists(model_name):
        print("Training model...")
        model = word2vec.Word2Vec(n_lists, workers=params.num_workers, \
                                  size=params.num_features, min_count = params.min_word_count, \
                                  window = params.context, sample = params.downsampling)

        # If you don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)

        # It can be helpful to create a meaningful model name and
        # save the model for later use. You can load it later using Word2Vec.load()
        model.save(model_name)
    else:
        model = word2vec.Word2Vec.load(model_name)
    print(model.doesnt_match("man woman child".split()))
    print(model.most_similar("man"))
    print(model.wv.similarity('queen', 'king'))
    print(model.wv.most_similar(positive=["woman", "family"], negative=["man"], topn=3))


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        if counter%1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
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


def word_to_vec_plt(reduced_results, y, model):
    features = np.array([text_to_vec(reduced_results[i], model, i) for i in range(len(reduced_results))])
    # y = all_lyrics[:lim]["genre"]
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(list(set(y)))
    y = le.transform(y)
    plott(features, y, RandomForestClassifier(n_estimators=100), 'word_to_vec_approach.png')

def plott(x ,y , model, figname):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.14)
    plotting(xtrain, ytrain, xtest, ytest, model, figname)


def plotting(X_train, y_train, X_test, y_test, model, figname):
    plt.close()
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    #ypredtrain = model.predict(X_train)
    #print('acc for train is {}'.format(sum(ypredtrain==y_train)/len(y_train)))
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
    plt.savefig(figname)
    # plt.show()
    with open('f1_score_{}.txt'.format(figname), 'w+') as f:
        f.write('acc for test is {}\n'.format(np.trace(confusion)/np.sum(confusion, axis=(1,0))))
        f.write('class id, precision, recall, f1 for {}\n'.format(figname))
        for class_id in range(len(confusion)):
            try:
                precision = confusion[class_id, class_id]/sum(confusion[:, class_id]) if sum(confusion[:, class_id]) >0 else 0
                recall = confusion[class_id, class_id]/sum(confusion[class_id, :]) if sum(confusion[class_id, :]) >0 else 0
                f.write('{}, {}, {}, {}\n'.format(class_id, precision, recall, 2*precision*recall/(precision+recall) if precision+recall >0 else 0))
            except Exception as e:
                print(e.message)
                print(traceback.print_exc())
    plt.close()
    return np.where(y_test!=ypred)


def tsnescatterplot(model, all_words, words_freq_genre):
    colors = {'Hip-Hip': 'red', 'Country': 'green', 'Not Available': 'blue', 'Other': 'gray',
              'Pop': 'coral', 'R&B': 'brown', 'Electronic': 'yellow', 'Metal': 'azure',
              'Folk': 'plum', 'Jazz': 'pink', 'Indie': 'lime', 'Rock': 'olive'}
    words_list = [(word, genre, colors[genre]) for genre in words_freq_genre for word in words_freq_genre[genre]]
    others = list(set(all_words)-set([item[0] for item in words_list]))
    others = [(word, 'non', 'grey') for word in others]
    all_words = np.array([w for w in others + words_list if w[0] in model.wv])
    arrays = np.array([model.wv[word] for word in all_words[:, 0]])
    # model, word, list_names
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    # arrays = np.empty((0, 300), dtype='f')
    # word_labels = [word]
    # color_list = ['red']
    #
    # # adds the vector of the query word
    # arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    #
    # # gets list of most similar words
    # close_words = model.wv.most_similar([word])
    #
    # # adds the vector for each of the closest words to the array
    # for wrd_score in close_words:
    #     wrd_vector = model.wv.__getitem__([wrd_score[0]])
    #     word_labels.append(wrd_score[0])
    #     color_list.append('blue')
    #     arrays = np.append(arrays, wrd_vector, axis=0)
    #
    # # adds the vector for each of the words from list_names to the array
    # for wrd in list_names:
    #     wrd_vector = model.wv.__getitem__([wrd])
    #     word_labels.append(wrd)
    #     color_list.append('green')
    #     arrays = np.append(arrays, wrd_vector, axis=0)

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
    plt.savefig('tsne.png')
    if not os.path.exists('tsne1.png'):
        plt.savefig('tsne1.png')
    # plt.show()


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
    parser = argparse.ArgumentParser(description="Edgify Computer Vision Training")
    parser.add_argument('--input_folder', action="store", dest="input_folder", help="input_folder")
    parser.add_argument('--output_folder', action="store", dest="output_folder", help="output_folder")
    parser.add_argument('--max_features', action="store", dest="max_features", type=int, default=100)
    parser.add_argument('--override', action="store", dest="override", default=True, type=lambda x:x.lower in ['false'])

    parser.add_argument('--num_features', action="store", dest="num_features", type=int, default=300)
    parser.add_argument('--min_word_count', action="store", dest="min_word_count", type=int, default=40)
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
