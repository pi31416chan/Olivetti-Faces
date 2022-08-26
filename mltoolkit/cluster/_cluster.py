import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



def kmeans_find_k(X,k_list:list,annotation:str=None):
    '''
    Plot the inertia and silhouette scores against the n number of clusters.
    
    Parameters:
    ----------
    X: 2d-array
        Input dataset in 2 dimensional array or pandas.DataFrame.
    k_list: list or array
        The list of k number of clusters to try.
    annotation: str
        The text to show as a part of the title.
    
    Returns:
    ----------
    1d-array: List of inertias, List of silhoutte scores.
    '''
    inertias = []
    silhous = []
    if annotation: title = f"Inertia and silhouette score ({annotation})"
    else: title = "Inertia and silhouette score"

    for k in k_list:
        k_model = KMeans(n_clusters=k).fit(X)
        inertias.append(k_model.inertia_)
        silhous.append(silhouette_score(X,k_model.labels_))

    fig = plt.figure(figsize=(10,6))
    ax1 = fig.gca()
    ax2 = ax1.twinx()
    ax1.plot(k_list,inertias,c='b',label='inertia')
    ax2.plot(k_list,silhous,c='r',label='silhouette')
    ax1.set(title=title,xlabel="n clusters",ylabel="inertia")
    ax2.set(ylabel="silhouette score")
    fig.legend(loc=1)
    plt.show()
    
    return np.array(inertias),np.array(silhous)

def kmeans_translate_label(model,y_true,y_pred=None):
    '''
    Translate the labels for clustering model for supervised condition.
    
    Parameters:
    ----------
    model: estimator
        Trained sklearn estimator.
    y_true: list or array
        The list of the actual labels.
    y_pred: list or array (optional)
        The list of the predicted labels.
        If this is passed, the labels in y_pred will be translated, 
        else the model's "labels_" attribute will be translated
    
    Returns:
    ----------
    1d-array: Array of translated labels
    '''
    if y_true is not np.ndarray:
        np.array(y_true)
    if y_pred is None:
        output = np.full_like(knn_0.labels_,-1)
        labels = np.unique(model.labels_)
        target_labels = model.labels_
    else:
        output = np.full_like(y_pred,-1)
        labels = np.unique(y_pred)
        target_labels = y_pred
    
    for label in labels:
        mask = (target_labels == label)
        output[mask] = mode(y_true[mask])[0]
    return output