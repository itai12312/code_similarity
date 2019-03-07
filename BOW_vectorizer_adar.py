#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Counting-the-frequency-of-words,-selecting-most-frequent-words-as-features---CountVectorizer--" data-toc-modified-id="Counting-the-frequency-of-words,-selecting-most-frequent-words-as-features---CountVectorizer---1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Counting the frequency of words, selecting most frequent words as features - CountVectorizer -</a></span></li><li><span><a href="#How-to-combine-multiple-rows-into-a-single-row-with-pandas" data-toc-modified-id="How-to-combine-multiple-rows-into-a-single-row-with-pandas-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>How to combine multiple rows into a single row with pandas</a></span></li><li><span><a href="#separate-functions-from-each-other" data-toc-modified-id="separate-functions-from-each-other-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>separate functions from each other</a></span></li></ul></div>

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


folder = 'C:\\temp\\C# code\\tokenized1\\'
df = pd.read_csv(folder + '084_update_quality_minmax_sizeFixture.cs.tree-viewer.txt', header = None)
# list(df[0])
df = df[df[0].notnull()]
df[0]


# In[133]:


df.applymap(lambda x: isinstance(x, (int, float)))


# #### Counting the frequency of words, selecting most frequent words as features - CountVectorizer -

# In[65]:


# creating the feature matrix 
from sklearn.feature_extraction.text import CountVectorizer
matrix = CountVectorizer(max_features=10)
cv_fit = matrix.fit_transform(df[0])
X = cv_fit.toarray()


# In[66]:


matrix.vocabulary_


# In[93]:


sizes = sorted(np.asarray(cv_fit.sum(axis=0))[0],reverse=True)
print(sizes)


# In[94]:


values = list(matrix.vocabulary_.keys())
values


# In[96]:


matrix.get_params()


# #### How to combine multiple rows into a single row with pandas

# In[97]:


df[0].iloc[0:10].str.cat(sep=' ')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'matrix')


# #### separate functions from each other

# In[98]:


starters = df.loc[df[0] == "BEGIN_METHOD"]
enders = df.loc[df[0] == "END_METHOD"]
starters


# In[99]:


enders


# In[100]:


zipped = list(zip(starters.index, enders.index))
zipped


# In[101]:


for begin, end in zipped:
    print(df[0].iloc[begin:end+1].str.cat(sep=' '))


# In[102]:


functions_list = []
for begin, end in zipped:
    functions_list.append(df[0].iloc[begin:end+1].str.cat(sep=' '))
functions_list


# In[106]:


def create_functions_list_from_df(df):
    df = df[df[0].notnull()]
    starters = df.loc[df[0] == "BEGIN_METHOD"]
    enders = df.loc[df[0] == "END_METHOD"]
    zipped = list(zip(starters.index, enders.index))
    functions_list = []
    for begin, end in zipped:
        functions_list.append(df[0].iloc[begin:end+1].str.cat(sep=' '))
    return functions_list


# In[138]:


create_functions_list_from_df(df)[0]


# In[113]:


def create_functions_list_from_filenames_list(files_list):
    functions_list = []
    for filename in files_list:
        try:
            df = pd.read_csv(folder + filename, header = None)
            functions_list += create_functions_list_from_df(df)
        except:
            continue
    return functions_list


# In[114]:


files_list1 =['084_update_quality_minmax_sizeFixture.cs.tree-viewer.txt', 
              '084_update_quality_minmax_size.cs.tree-viewer.txt',
              '085_expand_transmission_urlbase.cs.tree-viewer.txt',
             '.085_expand_transmission_urlbaseFixture.cs.tree-viewer.txt',
             '.086_pushbullet_device_ids.cs.tree-viewer.txt']
functions_list1 = create_functions_list_from_filenames_list(files_list1)


# In[142]:


functions_list1[0:10]


# In[124]:


def vectorize_text(text, max_features=20):
    # create the transform
    vectorizer = CountVectorizer(max_features = max_features)
    # build vocab
    vectorizer.fit(text)
    return vectorizer


# In[135]:


#vectorize(functions_list1).vocabulary_
matrix = CountVectorizer(max_features=30)
cv_fit = matrix.fit_transform(functions_list1)
matrix.vocabulary_


# In[141]:


cv_fit


# In[136]:


sizes = sorted(np.asarray(cv_fit.sum(axis=0))[0],reverse=True)
print(sizes)
values = list(matrix.vocabulary_.keys())
values


# In[137]:


def get_filenames(path):
    filenames = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    return filenames


# In[119]:


def vectorize_folder(path):
    files_list = get_filenames(path)
    functions_list = create_functions_list_from_filenames_list(files_list)
    vectorizer = vectorize_text(functions_list)
    return vectorizer


# In[120]:


from os import listdir
from os.path import isfile, join

mypath = folder
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles


# In[125]:


create_functions_list_from_filenames_list(onlyfiles[0:5])


# In[126]:


mypath = folder
vectorizer1 = vectorize_folder(mypath)


# In[ ]:


vectorizer1.vocabulary_


# In[ ]:




