import numpy as np
from collections import Counter

from after_clustering import create_cluster_and_types, create_random_vector, VECTOR_SIZE, distance, DISTANCE_TYPE


def create_small_clusters(number_of_clusters = 10):
    list_of_clusters = []
    for i in range(number_of_clusters):
        cluster, vectors_types = create_cluster_and_types()
        list_of_clusters.append(cluster)
    return list_of_clusters

def create_random_vectors(number_of_vectors = 20):
    list_of_vectors = []
    for i in range(number_of_vectors):
        list_of_vectors.append(create_random_vector(VECTOR_SIZE))
    return list_of_vectors


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


def expand_clusters_with_knn(clusters, centroids, old_vectors, new_vectors, vectors_index, k = 5):
    '''
    receives existing clusters, and adds the new vectors to them, according to k closest neighbors
    :param clusters: list of list of arrays, or list of ndarrays
    :param new_vectors: list of ndarrays
    :return: expanded clusters
    '''
    
    if len(clusters) == 0 or clusters == None:
        return clusters

    ndarray = False
    if isinstance(clusters[0], np.ndarray):
        ndarray = True
        clusters = convert_clusters_to_lists(clusters)
    clusters = add_vectors_to_clusters(clusters, centroids, k, old_vectors, new_vectors, vectors_index)

    if ndarray:
        clusters = convert_clusters_to_ndarrays(clusters)

    return clusters


def add_vectors_to_clusters(clusters, centroids, k, old_vectors, new_vectors, vectors_index):
    import copy
    original_clusters = copy.deepcopy(clusters)
    for i, vector in enumerate(new_vectors):
        if (i%10 == 0):
            print(i, len(new_vectors),i/len(new_vectors)*100,"%")
        add_vector_to_cluster(original_clusters, vector, i, clusters, old_vectors, vectors_index, k)
        if (i>5000):
            break
    return clusters

def add_vector_to_cluster(original_clusters, new_vector, vector_index, clusters, old_vectors, vectors_index, k):
    distances = get_distances_dict(original_clusters, old_vectors, new_vector)
    chosen_cluster = get_closest_cluster(distances, k)
    clusters[chosen_cluster].append(vectors_index[vector_index])


def get_closest_cluster(distances, k, distance_type = DISTANCE_TYPE):
    if distance_type == 1:
        reverse = True

    top_k = []
    # Extracting top k clusters
    for dist in sorted(distances.keys())[:k]:
        #print(dist)
        top_k.append(distances[dist]["cluster_number"])
    counter = Counter(top_k)
    chosen_cluster = counter.most_common(1)[0][0]
    return chosen_cluster


def get_distances_dict(clusters, old_vectors, new_vector):
    distances = {}

    # Calculating distance between the new vector and vectors in clusters
    for cluster_number, cluster in enumerate(clusters):
        for vector_number, vector in enumerate(cluster):
            old_vector = old_vectors[vector]
            dist = distance(new_vector, old_vector)

            distances[dist] = {"cluster_number" : cluster_number,
                               "vector_number_in_cluster" : vector_number}
    return distances

def randonm_expand():
    clusters = create_small_clusters()
    new_vectors = create_random_vectors()
    new_clusters = expand_clusters_with_knn(clusters, new_vectors, new_vectors)