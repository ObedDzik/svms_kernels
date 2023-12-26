import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.manifold import TSNE


def calculate_metrics(y_true, y_pred,  models, pos_label=1, average='binary'):

    score_vals = []
    score_dict = {}
    for i in range(len(models)):
        accuracy = accuracy_score(y_true, y_pred[:,i])
        score_vals.append(accuracy)
        
        precision = precision_score(y_true, y_pred[:,i], pos_label=pos_label, average=average)
        score_vals.append(precision)
        
        recall = recall_score(y_true, y_pred[:,i], pos_label=pos_label, average=average)
        score_vals.append(recall)

        score_dict[f'{models[i]}'] = score_vals
        score_vals = []
    
    return score_dict

def metric_bars(dataset, models, metric_scores):
    for i in range(len(metric_scores)):
        accuracy_values = [metric_scores[i][f'{models[0]}'][0], metric_scores[i][f'{models[1]}'][0], \
                        metric_scores[i][f'{models[2]}'][0], metric_scores[i][f'{models[3]}'][0],\
                        metric_scores[i][f'{models[4]}'][0], metric_scores[i][f'{models[5]}'][0],\
                        metric_scores[i][f'{models[6]}'][0], metric_scores[i][f'{models[7]}'][0]]
        precision_values = [metric_scores[i][f'{models[0]}'][1], metric_scores[i][f'{models[1]}'][1],\
                        metric_scores[i][f'{models[2]}'][1], metric_scores[i][f'{models[3]}'][1],\
                        metric_scores[i][f'{models[4]}'][1], metric_scores[i][f'{models[5]}'][1],\
                        metric_scores[i][f'{models[6]}'][1], metric_scores[i][f'{models[7]}'][1]]
        recall_values = [metric_scores[i][f'{models[0]}'][2], metric_scores[i][f'{models[1]}'][2],\
                        metric_scores[i][f'{models[2]}'][2], metric_scores[i][f'{models[3]}'][2],\
                        metric_scores[i][f'{models[4]}'][2], metric_scores[i][f'{models[5]}'][2],\
                        metric_scores[i][f'{models[6]}'][2], metric_scores[i][f'{models[7]}'][2]]

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        bar_width = 0.25
        index = np.arange(len(models))

        ax.barh(index, accuracy_values, color='blue', label='Accuracy', height=bar_width)
        ax.barh(index + bar_width, precision_values, color='green', label='Precision', height=bar_width, alpha=0.7)
        ax.barh(index + 2 * bar_width, recall_values, color='orange', label='Recall', height=bar_width, alpha=0.7)

        ax.set_yticks(index + bar_width)
        ax.set_yticklabels(models)

        # Add labels and legend
        ax.set_xlabel('Metric Score')
        ax.set_title(f'Model performance on the {dataset[i]} dataset')
        ax.legend(loc='upper left')

        savefig(f"{dataset[i]}_graph.jpg", fig=fig)
        plt.close(fig)


def decision_boundary_visualization(Xtrain, ytrain, classifier, models):

    for X_train, y_train, model_dict in zip(Xtrain, ytrain, classifier):
        model_names = list(model_dict.keys())
        
        for i in range(len(model_names)):

            # Apply t-SNE for dimensionality reduction to 2D
            X_train_embedded = TSNE(n_components=2).fit_transform(X_train)

            clf = model_dict[model_names[i]]
            clf.fit(X_train_embedded, y_train)

            # create meshgrid
            resolution = 100  # 100x100 background pixels
            X2d_xmin, X2d_xmax = np.min(X_train_embedded[:, 0]), np.max(X_train_embedded[:, 0])
            X2d_ymin, X2d_ymax = np.min(X_train_embedded[:, 1]), np.max(X_train_embedded[:, 1])
            xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

            # predict the class for each point in the meshgrid using the trained SVM classifier
            voronoiBackground = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            if model_names[i] not in ['linear_heart','linear_breast','linear_hep','r_forest_heart',\
                                      'k_heart','r_forest_breast', 'k_breast','r_forest_hep','k_hep']:
                voronoiBackground = np.argmax(voronoiBackground, axis=1)
            else:
                pass
            
            voronoiBackground = voronoiBackground.reshape(xx.shape)
            # plot
            plt.figure()
            plt.contourf(xx, yy, voronoiBackground, alpha=0.5, cmap=plt.cm.Paired)
            plt.scatter(X_train_embedded[:, 0], X_train_embedded[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired, marker='o', s=50)
            plt.title(f'{models[i]}')
            # plt.xlabel('Principal Component 1')
            # plt.ylabel('Principal Component 2')
            savefig(f"{model_names[i]}_boundary.jpg")
            plt.close()


def savefig(fname, fig=None, verbose=True):
    path = Path(".", fname)
    (plt if fig is None else fig).savefig(path, bbox_inches="tight", pad_inches=1)
    if verbose:
        print(f"Figure saved as '{path}'")

