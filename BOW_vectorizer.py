#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Counting-the-frequency-of-words,-selecting-most-frequent-words-as-features---CountVectorizer--" data-toc-modified-id="Counting-the-frequency-of-words,-selecting-most-frequent-words-as-features---CountVectorizer---1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Counting the frequency of words, selecting most frequent words as features - CountVectorizer -</a></span></li><li><span><a href="#How-to-combine-multiple-rows-into-a-single-row-with-pandas" data-toc-modified-id="How-to-combine-multiple-rows-into-a-single-row-with-pandas-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>How to combine multiple rows into a single row with pandas</a></span></li><li><span><a href="#separate-functions-from-each-other" data-toc-modified-id="separate-functions-from-each-other-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>separate functions from each other</a></span></li></ul></div>

# In[1]:


import pandas as pd
import numpy as np


# In[66]:


df = pd.read_csv('./tokenized1/084_update_quality_minmax_sizeFixture.cs.tree-viewer.txt', header = None)
# list(df[0])
df = df[df[0].notnull()]
df[0]


# In[42]:


df.applymap(lambda x: isinstance(x, (int, float)))


# #### Counting the frequency of words, selecting most frequent words as features - CountVectorizer -

# In[58]:


# creating the feature matrix 
from sklearn.feature_extraction.text import CountVectorizer
matrix = CountVectorizer(max_features=10)
X = matrix.fit_transform(df[0]).toarray()


# In[60]:


matrix.vocabulary_


# In[70]:


matrix.get_params()


# #### How to combine multiple rows into a single row with pandas

# In[67]:


df[0].iloc[0:10].str.cat(sep=' ')


# In[68]:


get_ipython().run_line_magic('pinfo', 'matrix')


# #### separate functions from each other

# In[105]:


starters = df.loc[df[0] == "BEGIN_METHOD"]
enders = df.loc[df[0] == "END_METHOD"]
starters


# In[106]:


enders


# In[119]:


zipped = list(zip(starters.index, enders.index))
zipped


# In[118]:


for begin, end in zipped:
    print(df[0].iloc[begin:end+1].str.cat(sep=' '))


# In[121]:


functions_list = []
for begin, end in zipped:
    functions_list.append(df[0].iloc[begin:end+1].str.cat(sep=' '))
functions_list


# In[123]:


create_functions_list_from_df(df)


# In[173]:


files_list1 =['./tokenized1/084_update_quality_minmax_sizeFixture.cs.tree-viewer.txt', 
              './tokenized1/084_update_quality_minmax_size.cs.tree-viewer.txt',
              './tokenized1/085_expand_transmission_urlbase.cs.tree-viewer.txt',
             './tokenized1/085_expand_transmission_urlbaseFixture.cs.tree-viewer.txt',
             './tokenized1/086_pushbullet_device_ids.cs.tree-viewer.txt']
functions_list1 = create_functions_list_from_files_list(files_list1)


# In[174]:


functions_list1


# In[158]:


vectorize(functions_list1).vocabulary_


# In[122]:


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


# In[161]:


def vectorize_text(text, max_features=20):
    # create the transform
    vectorizer = CountVectorizer(max_features = max_features)
    # build vocab
    vectorizer.fit(text)
    return vectorizer


# In[170]:


def get_filenames(path):
    filenames = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    return filenames


# In[162]:


def vectorize_folder(path):
    files_list = get_filenames(path)
    functions_list = create_functions_list_from_filenames_list(files_list)
    vectorizer = vectorize_text(functions_list)
    return vectorizer


# In[168]:


from os import listdir
from os.path import isfile, join

mypath = './tokenized1'
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles


# In[169]:


create_functions_list_from_filenames_list(onlyfiles[0:5])


# In[176]:


mypath = './tokenized1'
vectorizer1 = vectorize_folder(mypath)


# In[177]:


vectorizer1.vocabulary_


# In[ ]:




