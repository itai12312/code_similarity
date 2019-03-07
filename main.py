

def main(args=None):
    df = pd.read_csv('./tokenized1/084_update_quality_minmax_sizeFixture.cs.tree-viewer.txt', header = None)
    # list(df[0])
    df = df[df[0].notnull()]
    df.applymap(lambda x: isinstance(x, (int, float)))
    from sklearn.feature_extraction.text import CountVectorizer
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



def create_functions_list_from_df(df):
    df = df[df[0].notnull()]
    starters = df.loc[df[0] == "BEGIN_METHOD"]
    enders = df.loc[df[0] == "END_METHOD"]
    zipped = list(zip(starters.index, enders.index))
    functions_list = []
    for begin, end in zipped:
        functions_list.append(df[0].iloc[begin:end+1].str.cat(sep=' '))
    return functions_list


# In[172]:


def create_functions_list_from_filenames_list(files_list):
    functions_list = []
    for filename in files_list:
        try:
            df = pd.read_csv(filename, header = None)
            functions_list += create_functions_list_from_df(df)
        except:
            continue
    return functions_list


def vectorize_text(text, max_features=20):
    # create the transform
    vectorizer = CountVectorizer(max_features = max_features)
    # build vocab
    vectorizer.fit(text)
    return vectorizer


def get_filenames(path):
    filenames = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    return filenames


def vectorize_folder(path):
    files_list = get_filenames(path)
    functions_list = create_functions_list_from_filenames_list(files_list)
    vectorizer = vectorize_text(functions_list)
    return vectorizer


if __name__ == "__main__":
    main()
