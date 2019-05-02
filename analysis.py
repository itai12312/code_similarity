from collections import defaultdict
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import os

from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering


def analyze_functions2(matrix1, lists, raw_lists, vocab, params, gt_values, vectorizer, filenames, all_vulnerabilities):
    if params.vectorizer == 'count' and params.matrix_form == 'tfidf':
        matrix = matrix1.toarray() * 1. / matrix1.toarray().sum(axis=1)[:, None]
    elif params.vectorizer == 'count' and params.matrix_form == '0-1':
        matrix = matrix1.toarray() * 1. / matrix1.toarray().sum(axis=1)[:, None]
        matrix[matrix >= 1.] = 1
    # distances = sklearn.metrics.pairwise_distances(matrix.toarray(), metric=params.metric)
    # fig = plt.figure(figsize=(25, 10))
    distances = pdist(matrix, metric=params.metric)
    lnk = linkage(distances, params.clustering_method)
    # TODO:get list of filenames and locations!!
    cluster = AgglomerativeClustering(n_clusters=params.n_clusters, affinity=params.metric, linkage=params.clustering_method)
    results = cluster.fit_predict(matrix)
    colors = ['blue', 'orange', 'olive', 'green', 'cyan', 'brown', 'purple', 'pink', 'red']
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
                                                                           lists, params, raw_lists, all_vulnerabilities, z)
    plt.grid(axis='y')
    plt.savefig(os.path.join(params.output_folder, 'dendogram.svg'))
    for cluster_id, cluster in enumerate(intersting_clusters):
        clustering_type = 'average'
        clustering_metric = params.metric
        distances1 = pdist(matrix[cluster], metric=clustering_metric)
        lnk1 = linkage(distances1, clustering_type)
        plt.close('all')
        fig = plt.figure(figsize=(70, 70))
        plt.title(f'cluster clustering method {clustering_type}, metric {clustering_metric}')
        z1 = dendrogram(lnk1,  orientation='right', leaf_rotation=0, leaf_font_size=8,
                        labels=[f'{idx}_{val}' for idx, val in zip(cluster, gt_values[cluster])],
                        color_threshold=0.10)
        plt.grid(axis='y')
        plt.savefig(os.path.join(params.output_folder, f'dendogram_{cluster_id}_{clustering_metric}_{clustering_type}.svg'))
        dump_program_to_list_and_get_intersting_clusters(f'cluster_{cluster_id}_list', filenames[cluster],
                                                         gt_values[cluster], lists[cluster], params, raw_lists[cluster],
                                                         all_vulnerabilities[cluster],z1, cluster)

        from sklearn.naive_bayes import MultinomialNB
        from sklearn.model_selection import train_test_split
        clf = MultinomialNB()
        x_train, x_test, y_train, y_test = train_test_split(matrix[cluster], gt_values[cluster])
        clf.fit(x_train, y_train)
        res = clf.predict(x_test)
        acc = sum(res==y_test)/len(res)
        print('acc for cluster is {}'.format(acc))
    pass
    # plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')


def dump_program_to_list_and_get_intersting_clusters(output_filename, filenames, gt_values, lists, params, raw_lists, all_vulnerabilities,z, cluster=None):
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
                f.write(f'program # {orig_prog_id} # in cluster list {prog_id}, {i} with gt {gt_values[prog_id]} and vurn {all_vulnerabilities[prog_id]}\n')
            else:
                f.write(f'program # {prog_id}, location in cluser {i} with gt {gt_values[prog_id]} and vurn: {all_vulnerabilities[prog_id]}\n')
            f.write(f"{raw_lists[prog_id]}\n")
        f.write(f'finished new cluster with len {i - prev}\n')
        if i - prev > 18:
            intersting_clusters.append(np.array([z["leaves"][j] for j in range(prev, i)]))
            f.write('***\n')
    return intersting_clusters
