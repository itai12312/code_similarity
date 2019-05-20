import random
from os.path import join, isfile
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import re


import os

def create_with_countVectorizer(tokens_list):
    vectorizer = CountVectorizer(decode_error="replace", max_features=10)
    vec_train = vectorizer.fit_transform(tokens_list)
    # Save vectorizer.vocabulary_
    pickle.dump(vectorizer.vocabulary_, open("feature.pkl", "wb"))

    # Load it later
    # transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("feature.pkl", "rb")))

def create_tokens_list_from_df(df, filename):
    df = df[df[0].notnull()]
    tokens_list = []
    tokens_list.append(df[0].str.cat(sep=' '))
    return tokens_list

def create_tokens_list_from_df2(df, filename):
    df = df[df[0].notnull() & (df[0].str.len()>2)]
    tokens_list = list(df[0].str.lower())
    return tokens_list

def create_tokens_list_from_filenames_list(files_list):
    tokens_list = []
    exceptions_count = 0
    for filename in files_list:
        try:
            df = pd.read_csv(filename, header = None, encoding='utf8', error_bad_lines=False)
            tokens = create_tokens_list_from_df2(df, filename)
            tokens_list += tokens
        except Exception as e:
            print(filename)
            print(e)
            exceptions_count+=1
            continue
    print(exceptions_count)
    return tokens_list

def get_filenames(mypath):
    filenames = [join(mypath, f) for f in os.listdir(mypath) if isfile(join(mypath, f))]
    return filenames


def wordListToFreqDict(wordlist):
    wordSet = set(wordlist)
    uniqueWordList = list(wordSet)
    wordfreq = count_occurrences(wordlist, uniqueWordList)  #  10 min

    pickle.dump(uniqueWordList, open("uniqueWordList.pkl", "wb"))
    pickle.dump(wordfreq, open("wordfreq.pkl", "wb"))

    # uniqueWordList2 = pickle.load(open("uniqueWordList.pkl", "rb"))
    # wordfreq2 = pickle.load(open("wordfreq.pkl", "rb"))
    return dict(zip(uniqueWordList, wordfreq))

def count_occurrences(wordlist, uniqueWordList):
    i = 0
    wordfreq = []
    for p in uniqueWordList:
        occurrences = wordlist.count(p)
        wordfreq.append(occurrences)
        print(i)
        i+=1
    return wordfreq

# Sort a dictionary of word-frequency pairs in
# order of descending frequency.
def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

def create_tokens_dictionary(tokens_list):
    dictionary = wordListToFreqDict(tokens_list)
    sorteddict = sortFreqDict(dictionary)
    return sorteddict


def save_to_txt_file(sorted_freq_list, filename = 'sorted_freq_list.txt'):
    n = len(sorted_freq_list)
    errors = 0
    with open(filename, 'w') as f:
        f.write(str(n) + "\n")
        for item in sorted_freq_list:
            try:
                f.write(str(item[0]) + ", " + item[1] + "\n")
            except:
                errors += 1
    return errors


def create_vocabulary(path, percentage = 0.02):
    files_list = get_filenames(path)
    n = len(files_list)  #42000
    randomly_chosen_files = random.choices(files_list, k = int(n * percentage))
    tokens_list = create_tokens_list_from_filenames_list(randomly_chosen_files)  #25 sec
    # pickle.dump(tokens_list, open("tokens_list.pkl", "wb"))   #1 sec

    # tokens_list2 = pickle.load(open("tokens_list.pkl", "rb"))  #1 sec
    sorted_freq_list = create_tokens_dictionary(tokens_list)  # 10 min
    pickle.dump(sorted_freq_list, open("sorted_freq_list.pkl", "wb"))

    # sorted_freq_list2 = pickle.load(open("sorted_freq_list.pkl", "rb"))
    save_to_txt_file(sorted_freq_list)
    
    
    #### this was moved to create_with_countVectorizer() ### 
    # vectorizer = CountVectorizer(decode_error="replace", max_features=10)
    # vec_train = vectorizer.fit_transform(tokens_list)
    # # Save vectorizer.vocabulary_
    # pickle.dump(vectorizer.vocabulary_, open("feature.pkl", "wb"))
    # 
    # # Load it later
    # # transformer = TfidfTransformer()
    # loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("feature.pkl", "rb")))

    print("done")
    return sorted_freq_list


def cut_vocabulary(sorted_freq_list, first_n_to_take = 2000, first_n_to_remove = 0, minimum_occurrences = 20):
    short_sorted_freq_list = []
    for i in range(len(sorted_freq_list)):
        if i < first_n_to_remove:
            continue

        tuple = sorted_freq_list[i]
        if (i > first_n_to_remove + first_n_to_take) or (tuple[0] < minimum_occurrences):
            break

        short_sorted_freq_list.append(tuple)

    pickle.dump(short_sorted_freq_list, open("short_sorted_freq_list.pkl", "wb"))
    save_to_txt_file(short_sorted_freq_list, 'short_sorted_freq_list.txt')
    return short_sorted_freq_list


def clean_vocabulary(sorted_freq_list):
    clean_sorted_freq_list = []
    for i in range(len(sorted_freq_list)):
        tuple = sorted_freq_list[i]
        token_name = tuple[1]
        x = re.search("^[\w_]+\d*$", token_name)
        if (x == None):
            print(token_name)
            continue

        clean_sorted_freq_list.append(tuple)

    pickle.dump(clean_sorted_freq_list, open("clean_sorted_freq_list.pkl", "wb"))
    save_to_txt_file(clean_sorted_freq_list, 'clean_sorted_freq_list.txt')
    return clean_sorted_freq_list


path = "D:\\Y-Data\\Proj\\tokenized1"

sorted_freq_list = create_vocabulary(path, 0.1)
clean_vocabulary(sorted_freq_list)
cut_vocabulary(sorted_freq_list)
