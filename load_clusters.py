# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 08:38:52 2019

@author: USER
"""

EXPAND_SIZE = 2000

import functools
print = functools.partial(print, flush=True)
import sys
sys.stdout = open('output.log', 'w')
#sys.stdout = sys.__stdout__

import numpy as np
import expand_clusters as expand

path = "D:\\Y-Data\\Proj\\clusters\\"

print("vectors all")
vectors = np.load(path + "vectors_all.npz")
print(vectors.files) 
gt = vectors["gt_values"]
print(gt)
print(len(gt))
print(gt.sum())
print(gt[gt>0])
gt_index = np.where(gt !=0 )
print(gt_index)



print("vectors indices - vectors_not_in_cluster")
vectors_not_in_cluster = np.load(path + "vectors_indices.npy", allow_pickle=True)
print(vectors_not_in_cluster)
print(len(vectors_not_in_cluster))


print("clusters")
clusters = np.load(path + "vectors_clusters (1).npy", allow_pickle=True)
s = 0

for i in range(len(clusters)):
    s += len(clusters[i])
print("inside clusters:",s)
print("number of clusters",len(clusters))   

count = 0
for j in range(len(clusters)):
    cluster = clusters[j]
    for k in range(len(cluster)):
        count += 1
print("total of",count,"vectors in clusters")

'''
print("create a list of vectors in no cluster")
non_cluster = list()
for i in range(len(vectors_not_in_cluster)):
    vector_i = vectors_not_in_cluster[i]
    found = False
    for j in range(len(clusters)):
        cluster = clusters[j]
        for k in range(len(cluster)):
            if (vector_i == cluster[k]):
                found = True
        if (found):
            break
    if (not found):
        non_cluster.append(vector_i)

print(len(non_cluster))
#print(non_cluster)

print("add the non-cluster list to the list of clusters")

def convert_clusters_to_lists(clusters):
    clusters_as_lists = []
    for cluster in clusters:
        clusters_as_lists.append(list(cluster))
    return clusters_as_lists

def convert_clusters_to_ndarrays(clusters):
    clusters_as_ndarrays = []
    for cluster in clusters:
        clusters_as_ndarrays.append(np.asarray(cluster))
    return clusters_as_ndarrays


clusters_list = convert_clusters_to_lists(clusters)
clusters_list.append(list(non_cluster))
clusters = convert_clusters_to_ndarrays(clusters_list)


print("number of clusters",len(clusters))   

#print (clusters)
'''

print("tfidf10000")
tfidf = np.load(path + "tfidf.npz")
print(tfidf.files) 
tfidf_matrix = tfidf["matrix"]
print("tfidf matrix shape",tfidf_matrix.shape)


print("tfidf all")
tfidf = np.load(path + "tfidf (1).npz")
print(tfidf.files) 
tf_idf_full_matrix = tfidf["matrix"]
print("tfidf full matrix shape",tf_idf_full_matrix.shape)


print("expanding")
gt_index = gt_index[0]
index_list = np.random.random_integers(0,750000,EXPAND_SIZE)
for i in range(len(gt_index)):
    index_list[i] = gt_index[i]

print("create centroids vector")
def create_centroid(cluster):
    cluster_size= len(cluster)
    print(cluster)
    vector_size = len(cluster[0])
    centroid = np.zeros(vector_size)
    for i in range(cluster_size):
        centroid += cluster[i]
    centroid = centroid/cluster_size
    return centroid

centroids = list()
#for i in range(len(clusters)):
#    centroids.append(create_centroid(clusters[i]))
centroids = np.asanyarray(centroids)
    
# new_clusters = expand.expand_clusters_with_knn(clusters, centroids, tf_idf_full_matrix, tf_idf_full_matrix[index_list], index_list)
new_clusters = expand.expand_clusters_with_knn(clusters, False, tf_idf_full_matrix, tf_idf_full_matrix[index_list], index_list)
s = 0
for i in range(len(new_clusters)):
    s += len(new_clusters[i])
print("inside clusters:",s)


print("create clusters with gt")

DISTANCE_TYPE = 1
from scipy import spatial

def create_cluster(index_cluster, vectors_list):
    #print(vectors_list)
    vector_size = len(vectors_list[0])
    cluster_size= len(index_cluster)
    matrix = np.zeros((cluster_size, vector_size))
    for i in range(cluster_size):
        matrix[i] += vectors_list[index_cluster[i]]
    return matrix
    
def create_types(index_cluster, gt):
    cluster_size= len(index_cluster)
    vector = np.zeros(cluster_size)
    for i in range(cluster_size):
        vector[i] = gt[index_cluster[i]]
    return vector

def find_distance_between_cluster_and_type(cluster, vector_types, vulnerable):
    relevant_vectors = cluster[vector_types == vulnerable]
    total_vectors = len(cluster)
    total_vectors_of_type = len(relevant_vectors)
    if (total_vectors_of_type==0):
        return -1
    dist = 0
    for i in range(total_vectors):
        for j in range(total_vectors_of_type):
            dist += distance(cluster[i], relevant_vectors[j])
    return dist/total_vectors/total_vectors_of_type

def find_distance_between_centroid_and_type(cluster, centroid, vector_types, vulnerable):
    relevant_vectors = cluster[vector_types == vulnerable]
    total_vectors_of_type = len(relevant_vectors)
    if (total_vectors_of_type==0):
        return -1
    dist = 0
    for j in range(total_vectors_of_type):
        dist += distance(centroid, relevant_vectors[j])
    return dist/total_vectors_of_type

def distance(vector1, vector2, distance_type = DISTANCE_TYPE):
    # type = 1 -- cosine
    # type = 2 -- euclidean
    
    dist = 0
    if distance_type == 1:
        dist = spatial.distance.cosine(vector1, vector2)
    if distance_type == 2:
        dist = np.linalg.norm(vector1-vector2)
    return(dist)


from after_clustering import split_data, run_all_classifiers

def compare_distances():
    rel = 0
    dist = 0
    d_yes = 0
    d_no = 0
    num_plus = 0
    not_vulnerable = 0
    TRIALS = len(new_clusters)
    list_yes = list()
    list_no = list()
    for i in range(TRIALS):
        cluster = tf_idf_full_matrix[new_clusters[i]]
        #centroid = create_centroid(cluster)
        vectors_types = create_types(new_clusters[i], gt)
        print(i,"of",TRIALS,"total", i/TRIALS*100,"%")
        
        ## analyze clusters
        distance_yes = find_distance_between_cluster_and_type(cluster, vectors_types, vulnerable = 1)
        distance_no = find_distance_between_cluster_and_type(cluster, vectors_types, vulnerable = 0)
        #distance_yes = find_distance_between_centroid_and_type(cluster, centroid, vectors_types, vulnerable = 1)
        #distance_no = find_distance_between_centroid_and_type(cluster, centroid, vectors_types, vulnerable = 0)
        if (distance_yes<0):
            not_vulnerable += 1
            continue

        X_train, X_test, y_train, y_test = split_data(cluster, vectors_types)
        print(X_train, X_test, y_train, y_test)
        print(i)
        print(cluster)
        print(vectors_types)
        if (X_train.sum()>0 and y_train.sum()>0):
            run_all_classifiers(X_train, X_test, y_train, y_test)



        list_yes.append(distance_yes)
        list_no.append(distance_no)
        d_yes += distance_yes
        d_no += distance_no
        #print(d_yes, d_no)
        if (distance_no>distance_yes):
            num_plus+=1
        relation = distance_yes/distance_no
        dist += distance_yes - distance_no
        rel += relation
    print("list yes:", list_yes)
    print("list no:", list_no)
    from scipy import stats
    print("t-test results:")
    test = stats.ttest_ind(list_no,list_yes)
    print(test)

    print("clusters with no vulnerability", not_vulnerable, "(", not_vulnerable/TRIALS*100,"% )")
    print("average distance relation (vuln/nonVuln):", rel/TRIALS)
    print("average distance difference (vuln-nonVuln):", dist/TRIALS)
    print("percentage of cases where nonVuln is closer to the cluster than vuln:", num_plus/TRIALS*100, "%")
    
    
compare_distances()

