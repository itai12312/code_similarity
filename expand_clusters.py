import numpy as np
from collections import Counter

from scipy.spatial.distance import cdist


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


def create_centroids(clusters_idxs, tf_idf_full_matrix):
    centroids = []
    for cluster_idxs in clusters_idxs:
        centroid = np.average(tf_idf_full_matrix[cluster_idxs], axis=0)
        centroids.append(centroid)
    return centroids


def expand_clusters_with_knn(clusters_idxs, use_centroids, tf_idf_full_matrix, new_vectors, vectors_index, k = 5):
    '''
    receives existing clusters, and adds the new vectors to them, according to k closest neighbors
    '''
    
    if len(clusters_idxs) == 0 or clusters_idxs == None:
        return clusters_idxs

    centroids = create_centroids(clusters_idxs, tf_idf_full_matrix) if use_centroids else None

    ndarray = False
    if isinstance(clusters_idxs[0], np.ndarray):
        ndarray = True
        clusters_idxs = convert_clusters_to_lists(clusters_idxs)
    clusters_idxs = add_vectors_to_clusters(clusters_idxs, centroids, k, tf_idf_full_matrix, new_vectors, vectors_index)

    if ndarray:
        clusters_idxs = convert_clusters_to_ndarrays(clusters_idxs)

    return clusters_idxs


def add_vectors_to_clusters(clusters_idxs, centroids, k, tf_idf_full_matrix, new_vectors, vectors_index):
    import copy
    original_clusters_idxs = copy.deepcopy(clusters_idxs)
    for i, vector in enumerate(new_vectors):
        if (i%10 == 0):
            print(i, len(new_vectors),i/len(new_vectors)*100,"%")
        add_vector_to_cluster(original_clusters_idxs, centroids, vector, i, clusters_idxs, tf_idf_full_matrix, vectors_index, k)
        if (i>5000):
            break
    return clusters_idxs

def add_vector_to_cluster(original_clusters, centroids, new_vector, vector_index, clusters_idxs, tf_idf_full_matrix, vectors_index, k):
    distances = get_distances_dict_faster(original_clusters, tf_idf_full_matrix, new_vector, k)
    chosen_cluster = get_closest_cluster(distances, k)
    clusters_idxs[chosen_cluster].append(vectors_index[vector_index])


def get_closest_cluster(distances, k, distance_type = DISTANCE_TYPE):
    top_k = []
    k = min(k, len(distances.keys()))
    # Extracting top k clusters
    for dist in sorted(distances.keys())[:k]:
        #print(dist)
        top_k.append(distances[dist]["cluster_number"])
    counter = Counter(top_k)
    chosen_cluster = counter.most_common(1)[0][0]
    return chosen_cluster


def get_distances_dict(original_clusters, centroids, tf_idf_full_matrix, new_vector):
    distances = {}

    # Calculating distance between the new vector and vectors in clusters
    for cluster_number, cluster in enumerate(original_clusters):
        if centroids != None and len(centroids) != 0:
            dist = distance(new_vector, centroids[cluster_number])
            distances[dist] = {"cluster_number": cluster_number,
                               "centroid" : centroids[cluster_number]}
        else:
            for vector_number, vector in enumerate(cluster):
                old_vector = tf_idf_full_matrix[vector]
                dist = distance(new_vector, old_vector)

                distances[dist] = {"cluster_number" : cluster_number,
                                   "vector_number_in_cluster" : vector_number}

    return distances


def get_distances_dict_faster(original_clusters, tf_idf_full_matrix, new_vector, k):
    distances = {}

    # Calculating distance between the new vector and vectors in clusters
    for cluster_number, cluster in enumerate(original_clusters):
        cluster_vectors = tf_idf_full_matrix[original_clusters[cluster_number]]
        dists = cdist(new_vector.reshape(1, new_vector.shape[0]), cluster_vectors, 'cosine')
        sorted_dists = sorted(dists[0])[:k]
        for dist in sorted_dists:
            distances[dist] = {"cluster_number": cluster_number}

    return distances


def randonm_expand():
    clusters = create_small_clusters()
    new_vectors = create_random_vectors()
    new_clusters = expand_clusters_with_knn(clusters, new_vectors, new_vectors)