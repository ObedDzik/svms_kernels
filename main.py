import os
from pathlib import Path
import numpy as np
import csv

from data_preprocessing import get_heart, get_breastCancer, get_hepatitis
from models import Svm_Model, Random_Forest, Knn
from kernels import cauchy_kernel, laplace_kernel, r_quadratic
from visualization import calculate_metrics, metric_bars, decision_boundary_visualization


# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

def main():
    
    models = ['svm_linear_kernel', 'svm_sigmoid_kernel', 'svm_rbf_kernel',\
               'svm_cauchy_kernel', 'svm_laplacian_kernel', 'svm_rational_quadratic_kernel',\
                'random forest', 'knn']
    dataset_types = ['heart disease', 'breast cancer', 'hepatitis']
    kernels = ['linear', 'sigmoid', 'rbf', cauchy_kernel, laplace_kernel, r_quadratic]

    # Heart
    X_train_heart, X_test_heart, y_train_heart, y_test_heart, heart_classes = get_heart()
    X_train_breast, X_test_breast, y_train_breast, y_test_breast, breast_classes = get_breastCancer()
    X_train_hep, X_test_hep, y_train_hep, y_test_hep, hep_classes = get_hepatitis()

    jh = 0
    jb = 0
    jp = 0

    heart_classifiers = []
    breast_classifiers = []
    hep_classifiers = []

    heart_model_keys = ['linear_heart','sigmoid_heart','rbf_heart','cauchy_kernel_heart','laplace_kernel_heart',\
                         'r_quadratic_heart','r_forest_heart','k_heart']
    
    breast_model_keys = ['linear_breast','sigmoid_breast','rbf_breast','cauchy_kernel_breast','laplace_kernel_breast',\
                         'r_quadratic_breast','r_forest_breast', 'k_breast']
    
    hep_model_keys= ['linear_hep','sigmoid_hep','rbf_hep','cauchy_kernel_hep','laplace_kernel_hep',\
                         'r_quadratic_hep','r_forest_hep','k_hep']

    heart_predictions = np.zeros((X_test_heart.shape[0],len(heart_model_keys)))
    breast_predictions = np.zeros((X_test_breast.shape[0],len(breast_model_keys)))
    hep_predictions = np.zeros((X_test_hep.shape[0],len(hep_model_keys)))
    
    
    #SVM training and prediction for heart
    for kernel in kernels:
        #Heart
        clf_heart = Svm_Model(kernel=kernel)
        heart_classifiers.append(clf_heart)
        clf_heart.fit(X_train_heart, y_train_heart)
        if kernel == 'linear':
            heart_predictions[:,jh] = clf_heart.predict(X_test_heart)
        else:
            heart_predictions[:,jh] = np.argmax(clf_heart.predict(X_test_heart), axis=1)
        jh = jh+1
        
        #Breast
        clf_breast = Svm_Model(kernel=kernel)
        breast_classifiers.append(clf_breast)
        clf_breast.fit(X_train_breast, y_train_breast)
        if kernel == 'linear':
            breast_predictions[:,jb] = clf_breast.predict(X_test_breast)
        else:
            breast_predictions[:,jb] = np.argmax(clf_breast.predict(X_test_breast), axis=1)
        jb = jb+1
        
        #Hepatitis
        clf_hep = Svm_Model(kernel=kernel)
        hep_classifiers.append(clf_hep)
        clf_hep.fit(X_train_hep, y_train_hep)
        if kernel == 'linear':
            hep_predictions[:,jp] = clf_hep.predict(X_test_hep)
        else:
            hep_predictions[:,jp] = np.argmax(clf_hep.predict(X_test_hep), axis=1)
        jp = jp+1


        
    #Random forest training and prediction
    #Heart
    clf = Random_Forest()
    clf.fit(X_train_heart, y_train_heart)
    heart_classifiers.append(clf)
    heart_predictions[:,jh] = clf.predict(X_test_heart)
    jh = jh+1

    #Breast
    clf = Random_Forest()
    clf.fit(X_train_breast, y_train_breast)
    breast_classifiers.append(clf)
    breast_predictions[:,jb] = clf.predict(X_test_breast)
    jb = jb+1

    #Hepatitis
    clf = Random_Forest()
    clf.fit(X_train_hep, y_train_hep)
    hep_classifiers.append(clf) 
    hep_predictions[:,jp] = clf.predict(X_test_hep)
    jp = jp+1


    #KNN training and prediction
    #Heart
    clf = Knn(k=3)
    clf.fit(X_train_heart, y_train_heart)
    heart_classifiers.append(clf)
    heart_predictions[:,jh] = clf.predict(X_test_heart)
    jh = jh+1

    #Breast
    clf = Knn(k=3)
    clf.fit(X_train_breast, y_train_breast)
    breast_classifiers.append(clf)
    breast_predictions[:,jb] = clf.predict(X_test_breast)
    jb = jb+1

    #Hepatitis
    clf = Knn(k=3)
    clf.fit(X_train_hep, y_train_hep)
    hep_classifiers.append(clf) 
    hep_predictions[:,jp] = clf.predict(X_test_hep)
    jp = jp+1

    heart_model_dict = dict(zip(heart_model_keys, heart_classifiers))
    breast_model_dict = dict(zip(breast_model_keys, breast_classifiers))
    hep_model_dict = dict(zip(hep_model_keys, hep_classifiers))
    
    all_classifiers = [heart_model_dict, breast_model_dict, hep_model_dict]
    all_xtrain = [X_train_heart, X_train_breast, X_train_hep]
    all_ytrain = [y_train_heart, y_train_breast, y_train_hep]
    
    #Obtain metric values
    metrics_values = list([
        calculate_metrics(y_test_heart, heart_predictions, models),
        calculate_metrics(y_test_breast, breast_predictions, models),
        calculate_metrics(y_test_hep, hep_predictions, models)])
    
    
    csv_file = 'eval_metrics.csv'
    with open(csv_file, 'w', newline='') as csvfile:
        # Extract the keys (column headers) from the first dictionary
        fieldnames = metrics_values[0].keys()
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(metrics_values)

    print(f'The data has been written to {csv_file}')

    print(1)
    metric_bars(dataset_types, models, metrics_values),
 
    decision_boundary_visualization(all_xtrain, all_ytrain, all_classifiers, models)
    # print(metrics_values)
    # # Reshape the metrics_values to have the shape (num_datasets, num_models, num_metrics)
    # metrics_values = metrics_values.reshape((len(dataset_types), len(models), -1))
    # # Visualize metrics
    # visualize_metrics_bar_chart(metrics_values, models, dataset_types)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
