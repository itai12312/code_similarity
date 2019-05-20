# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:55:26 2019

@author: Adar
"""

from main import main
from vocabulary import create_vocabulary, clean_vocabulary, cut_vocabulary

path = "D:\\Y-Data\\Proj\\tokenized1"

## I'm commenting all parts that already run in the past, in order to make them run again - uncomment them

sorted_freq_list = create_vocabulary(path = "D:\\Y-Data\\Proj\\tokenized1", percentage = 0.1)
clean_freq_list = clean_vocabulary(sorted_freq_list) # ideally this one should use a file and not the return value
freq_list = cut_vocabulary(clean_freq_list) # ideally this one should use a file and not the return value

create_vectors_for_methods() #create a vectors file including file name, function name, full vector, list of vlunerabilities in matrix
normalize_vectors() # into a new file
create_distance_matrix() #into a new file
cluster_methods(minimal_cluster = 20) # into a set of files - each file for cluster (do not create file for clusters of less than 20 methods)

