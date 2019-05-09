from collections import defaultdict
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import os

from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


def analyze_functions2(matrix1, lists, raw_lists, params, gt_values, vectorizer, filenames, all_vulnerabilities, all_start_raw):
    # vocab = list(vectorizer.vocabulary_.keys())
    if params.vectorizer == 'count' and params.matrix_form == 'tfidf':
        matrix = matrix1.toarray() * 1. / matrix1.toarray().sum(axis=1)[:, None]
    elif params.vectorizer == 'count' and params.matrix_form == '0-1':
        matrix = matrix1.toarray() * 1. / matrix1.toarray().sum(axis=1)[:, None]
        matrix[matrix >= 1.] = 1
    else:
        matrix = matrix1.toarray()
    # distances = sklearn.metrics.pairwise_distances(matrix.toarray(), metric=params.metric)
    # fig = plt.figure(figsize=(25, 10))
    distances = pdist(matrix, metric=params.metric)
    lnk = linkage(distances, params.clustering_method)
    # TODO:get list of filenames and locations!!
    # cluster = AgglomerativeClustering(n_clusters=params.n_clusters, affinity=params.metric, linkage=params.clustering_method)
    # results = cluster.fit_predict(matrix)
    # colors = ['blue', 'orange', 'olive', 'green', 'cyan', 'brown', 'purple', 'pink', 'red']
    # fl = fcluster(lnk, params.n_clusters,criterion='maxclust')
    # assert fl == results
    # link_cols = {}
    # dflt_col = "#808080"   # Unclustered gray
    # for i, i12 in enumerate(lnk[:, :2].astype(int)):
    #     c1, c2 = (link_cols[x] if x > len(lnk) else colors[results[x]] for x in i12)
    #     link_cols[i+1+len(lnk)] = c1 if c1 == c2 else dflt_col
    # def get_color(k):
    #     return link_cols[k]
    plt.close('all')
    fig = plt.figure(figsize=(150, 70))
    plt.title('clustering method {}, metric {}'.format(params.clustering_method, params.metric))
    z = dendrogram(lnk, labels=[f'{idx}_{val}' for idx, val in enumerate(gt_values)],
                   color_threshold=0.17,
                   orientation='right', leaf_font_size=8, leaf_rotation=0) # , link_color_func=get_color)
    cluster_idxs = defaultdict(list)
    for c, pi in zip(z['color_list'], z['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
    intersting_clusters = dump_program_to_list_and_get_intersting_clusters('dendogram_list.txt', filenames, gt_values,
                                                                           lists, params, raw_lists, all_vulnerabilities, all_start_raw, z)
    if params.cluster_analysis_count > -1:
        intersting_clusters = intersting_clusters[:params.cluster_analysis_count]
    plt.grid(axis='y')
    plt.savefig(os.path.join(params.output_folder, 'dendogram.svg'))
    x_trains, x_tests, y_trains, y_tests = [], [], [], []
    with open(os.path.join(params.output_folder, 'cluster_analysis.txt'), 'w+') as f_cluster:
        for cluster_id, cluster in enumerate(intersting_clusters):
            clustering_type = 'average'
            clustering_metric = params.metric
            distances1 = pdist(matrix[cluster], metric=clustering_metric)
            lnk1 = linkage(distances1, clustering_type)
            plt.close('all')
            fig = plt.figure() # figsize=(70, 70)
            plt.title(f'cluster clustering method {clustering_type}, metric {clustering_metric}')
            z1 = dendrogram(lnk1,  orientation='right', leaf_rotation=0, leaf_font_size=8,
                            labels=[f'{idx}_{val}' for idx, val in zip(cluster, gt_values[cluster])],
                            color_threshold=0.10)
            plt.grid(axis='y')
            plt.savefig(os.path.join(params.output_folder, f'dendogram_{cluster_id}_{clustering_metric}_{clustering_type}.svg'))
            dump_program_to_list_and_get_intersting_clusters(f'cluster_{cluster_id}_list', filenames[cluster],
                                                             gt_values[cluster], lists[cluster], params, raw_lists[cluster],
                                                             all_vulnerabilities[cluster], all_start_raw[cluster], z1, cluster)

            confusion, x_train, x_test, y_train, y_test = predictions(gt_values, matrix)
            x_trains.append(x_train)
            x_tests.append(x_test)
            y_trains.append(y_train)
            y_tests.append(y_test)
            f_cluster.write(f'confusion for clusters {cluster_id} is\n')
            f_cluster.write(f'{confusion}\n')
        pass
        clf = MultinomialNB()
        clf.fit(np.concatenate(x_trains), np.concatenate(y_trains))
        pred = clf.predict(np.concatenate(x_tests))
        f_cluster.write(f'for all, confusion is\n')
        f_cluster.write(f'{confusion_matrix(pred, np.concatenate(y_tests))}\n')
        # plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')


def predictions(gt_values, matrix):
    clf = MultinomialNB()
    x_train, x_test, y_train, y_test = train_test_split(matrix, gt_values)
    clf.fit(x_train, y_train)
    res = clf.predict(x_test)
    return confusion_matrix(res, y_test), x_train, x_test, y_train, y_test


def dump_program_to_list_and_get_intersting_clusters(output_filename, filenames, gt_values, lists, params, raw_lists, all_vulnerabilities, all_start_raw, z, cluster=None):
    intersting_clusters = []
    with open(os.path.join(params.output_folder, output_filename), 'w+') as f:
        f.write(f'order of leaves is {z["leaves"]}\n')
        f.write(f'names of files is {filenames}\n')
        prev = -1
        for i in range(len(lists) - 1):
            if i == 0 or z['color_list'][i] != z['color_list'][i - 1]:
                f.write(f'finished new cluster with len {i - prev}\n')
                if i - prev > 18:
                    f.write('***\n')
                    intersting_clusters.append(np.array([z["leaves"][j] for j in range(prev, i)]))
                prev = i
            prog_id = z["leaves"][i]

            if cluster is not None:
                orig_prog_id = cluster[prog_id]
                begin_message = f'program # {orig_prog_id}, in cluster list {prog_id}, index in list {i}'
            else:
                begin_message = f'program # {prog_id}, location in cluser {i}'
            f.write(begin_message+f' with gt {gt_values[prog_id]}\n  in filename{filenames[prog_id]} starts@{all_start_raw[prog_id]} and vurn: {all_vulnerabilities[prog_id]}\n')
            f.write(f"{raw_lists[prog_id]}\n")
        f.write(f'finished new cluster with len {i - prev}\n')
        if i - prev > 18:
            intersting_clusters.append(np.array([z["leaves"][j] for j in range(prev, i)]))
            f.write('***\n')
    return intersting_clusters
