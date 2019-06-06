# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from scipy import spatial

VECTOR_SIZE = 1500
CLUSTER_SIZE = 100
PERCENT_TRUE = 0.4
TRIALS = 25
DISTANCE_TYPE = 1 # 1 - cosine, 2 - euclidean



def create_random_vector(vector_size):
    vector = np.random.uniform(size=vector_size)/3
    vector[vector<0.2] = 0
    return vector
    
def create_cluster(basic_vector, cluster_size):
    vector_size = len(basic_vector)
    matrix = np.random.rand(cluster_size, vector_size)
    matrix[matrix<0.2] = 0
    for i in range(cluster_size):
        matrix[i] += basic_vector
    return matrix
    
def create_types(cluster, percent_yes):
    sum_lines = np.sum(cluster, axis=1)
    percentile = np.percentile(sum_lines, percent_yes*100)
    vector = np.where(sum_lines>percentile, 0, 1)
    return vector

def find_center_of_cluster(cluster):   
    mean = np.mean(cluster, axis=1)
    return(mean)
    
def find_stdev_of_cluster(cluster):   
    std = np.std(cluster, axis=1)
    return(std)

def find_distance_between_cluster_and_type(cluster, vector_types, vulnerable):
    relevant_vectors = cluster[vector_types == vulnerable]
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
    
    dist = 0
    if distance_type == 1:
        dist = spatial.distance.cosine(vector1, vector2)
    if distance_type == 2:
        dist = np.linalg.norm(vector1-vector2)
    return(dist)

def create_cluster_and_types():
    random_vector = create_random_vector(vector_size = VECTOR_SIZE)
    cluster = create_cluster(basic_vector = random_vector, cluster_size = CLUSTER_SIZE) # all vectors in the cluster
    vectors_types = create_types(cluster, percent_yes = PERCENT_TRUE)
    print(vectors_types)
    return cluster, vectors_types


def compare_distances():
    rel = 0
    dist = 0
    d_yes = 0
    d_no = 0
    num_plus = 0
    for i in range(TRIALS):
        print(i,"of",TRIALS,"total", i/TRIALS*100,"%")
        ## create clusters
        cluster, vectors_types = create_cluster_and_types()
        
        ## cnclyze clusters
        cluster_center = find_center_of_cluster(cluster)
        cluster_std = find_stdev_of_cluster(cluster)
        distance_yes = find_distance_between_cluster_and_type(cluster, vectors_types, vulnerable = 1)
        distance_no = find_distance_between_cluster_and_type(cluster, vectors_types, vulnerable = 0)
        d_yes += distance_yes
        d_no += distance_no
        #print(d_yes, d_no)
        if (distance_no<distance_yes):
            num_plus+=1
        relation = distance_yes/distance_no
        dist += distance_yes - distance_no
        rel += relation
    print("average relation", rel/TRIALS)
    print("average distance", dist/TRIALS)
    print("percentage of cases where result is bigger:", num_plus/TRIALS*100, "%")




from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def split_data(cluster, vectors_types):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(cluster, vectors_types, random_state=0)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def print_report(test, test_text, X_train, X_test, y_train, y_test):
    print(test_text)
    print(y_test)
    pred = test.predict(X_test)
    print(pred)
    print('Accuracy on training set: {:.2f}'.format(test.score(X_train, y_train)))
    print('Accuracy  on test set: {:.2f}'.format(test.score(X_test, y_test)))
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    print()

def logistic_regression(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    print_report(logreg,  'Logistic Regression', X_train, X_test, y_train, y_test)


def decision_tree(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    tree_classifier = DecisionTreeClassifier()
    clf = tree_classifier.fit(X_train, y_train)
    print_report(clf,  'Decision Tree', X_train, X_test, y_train, y_test)

def KNN(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print_report(knn,  'K-NN', X_train, X_test, y_train, y_test)

def LDA(X_train, X_test, y_train, y_test):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    print_report(lda,  'LDA', X_train, X_test, y_train, y_test)

def Naive_Bais(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    print_report(gnb,  'Naive Bayes', X_train, X_test, y_train, y_test)

def SVM(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(X_train, y_train)
    print_report(svm,  'SVM', X_train, X_test, y_train, y_test)

def random_forest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    rf =  RandomForestClassifier()
    rf.fit(X_train, y_train)
    print_report(rf,  'Random Forest', X_train, X_test, y_train, y_test)

def AdaBoost(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import AdaBoostClassifier
    abc =  AdaBoostClassifier()
    abc.fit(X_train, y_train)
    print_report(abc,  'AdaBoost', X_train, X_test, y_train, y_test)

def neural_network(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPClassifier
    mlp =  MLPClassifier()
    mlp.fit(X_train, y_train)
    print_report(mlp,  'Neural Network', X_train, X_test, y_train, y_test)


def run_all_classifiers(X_train, X_test, y_train, y_test):
    logistic_regression(X_train, X_test, y_train, y_test)
    decision_tree(X_train, X_test, y_train, y_test)
    KNN(X_train, X_test, y_train, y_test)
    LDA(X_train, X_test, y_train, y_test)
    Naive_Bais(X_train, X_test, y_train, y_test)
    SVM(X_train, X_test, y_train, y_test)
    random_forest(X_train, X_test, y_train, y_test)
    AdaBoost(X_train, X_test, y_train, y_test)
    neural_network(X_train, X_test, y_train, y_test)


def main():
    compare_distances()
    
    cluster, vectors_types = create_cluster_and_types()
    X_train, X_test, y_train, y_test = split_data(cluster, vectors_types)
    run_all_classifiers(X_train, X_test, y_train, y_test)

#main()