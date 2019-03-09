import argparse
import pandas as pd
from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
import os
from os import listdir
from os.path import isfile, join
import pathos.multiprocessing as multiprocessing

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
    vectorizer1 = vectorize_folder(mypath, 5, params.max_features)
    vectorizer1.vocabulary_
    model_name = "model1.pkl"
    if params.override or not os.path.exists(model_name):
        print("Training model...")
        sentences = reduce(operator.add, reduced_results)
        model = word2vec.Word2Vec(sentences, workers=params.num_workers, \
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
