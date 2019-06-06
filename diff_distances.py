# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:53:50 2019

@author: Adar
"""

import numpy as np
from scipy import spatial

TRIALS = 25
DISTANCE_TYPE = 1 # 1 - cosine, 2 - euclidean


def find_center_of_cluster(cluster):   
    mean = np.mean(cluster, axis=1)
    #print("mean vector")
    #print(mean)
    return(mean)
    
def find_stdev_of_cluster(cluster):   
    std = np.std(cluster, axis=1)
    #print("std vector")
    #print(std)
    return(std)

def find_distance_between_cluster_and_type(cluster, vector_types, vulnerable):
    relevant_vectors = cluster[vector_types == vulnerable]
    #print("vectors that are", vulnerable)
    #print(relevant_vectors)
    total_vectors = len(cluster)
    total_vectors_of_type = len(relevant_vectors)
    dist = 0
    for i in range(total_vectors):
        for j in range(total_vectors_of_type):
            dist += distance(cluster[i], relevant_vectors[j])
    return dist/total_vectors/total_vectors_of_type

def distance(vector1, vector2, distance_type = DISTANCE_TYPE):
    # type = 1 -- cosine
    # type = 2 -- euclidean
    
    #print("distance between")
    #print(vector1)
    #print(vector2)
    dist = 0
    if distance_type == 1:
        dist = spatial.distance.cosine(vector1, vector2)
    if distance_type == 2:
        dist = np.linalg.norm(vector1-vector2)
    #print("is", dist)
    return(dist)

