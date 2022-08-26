# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score,precision_score,recall_score,\
                            f1_score,ConfusionMatrixDisplay



# Classes



# Functions
def test_water(estimators:list,X:np.ndarray,y:list|np.ndarray,cv:int=4,
               scoring:str|list='accuracy',n_jobs:int=2,
               return_train_score=False) -> pd.DataFrame:
    '''
    Doing cross-validation on the passed list of estimators. Compile and output
    the test results as a pandas.DataFrame

    Parameters:
    ----------
    estimators: list or array of estimator
        The list of estimators to be evaluated.
    X: 2D-ndarray
        Refers to documentations on sklearn.model_selection.cross_validate.
    y: 1D-ndarray
        Refers to documentations on sklearn.model_selection.cross_validate.
    cv: int
        Refers to documentations on sklearn.model_selection.cross_validate.
    scoring: str
        Refers to documentations on sklearn.model_selection.cross_validate.
    n_jobs: int
        Refers to documentations on sklearn.model_selection.cross_validate.
    return_train_score: bool
        Refers to documentations on sklearn.model_selection.cross_validate.
    
    Returns:
    ----------
    DataFrame: pandas.DataFrame
    '''
    results = None
    for estimator in estimators:
        model = estimator()
        name = str(model)[:str(model).find('(')]
        score_dict = cross_validate(model,X,y,cv=cv,scoring=scoring,n_jobs=n_jobs,
                                    return_train_score=return_train_score)
        
        if results is None:
            results = pd.concat((pd.DataFrame([name]*cv,columns=['estimator']),
                                 pd.DataFrame(score_dict)),axis=1,join='inner')
        else:
            results = pd.concat((results,
                                 pd.concat((pd.DataFrame([name]*cv,columns=['estimator']),
                                            pd.DataFrame(score_dict)),axis=1,join='inner')),
                                 axis=0)
    else:
        return results.reset_index().rename({'index':'run'},axis=1)

def measure_classifier(estimator,trainortest:str,y_true,y_pred,
                       labels=None,average='binary') -> dict:
    '''
    Quickly measure the performance of an estimator using common scoring method
    for classification.
    
    Parameters:
    ----------
    estimator: Estimator object
        Mainly for the use of extracting the name and labeling the output dict.
    trainortest: str
        To label the results as train or test measurement.
    y_true: 1D-ndarray
        True labels.
    y_pred: 1D-ndarray
        Predicted labels.
    labels: list of str
        The labels to replace the default labels.
    average: {'micro', 'macro', 'samples', 'weighted', 'binary'}, Default: 'binary'
        This parameter is required for multiclass/multilabel targets.
        Relevant to Precision, Recall and F1 score only.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    Returns:
    ----------
    dict: Dictionary
    '''
    acc = accuracy_score(y_true,y_pred)
    prec = precision_score(y_true,y_pred,average=average)
    rec = recall_score(y_true,y_pred,average=average)
    f1 = f1_score(y_true,y_pred,average=average)
    
    name = str(estimator)
    # [:str(estimator).find('(')]
    print("Estimator:",name)
    print("Train/Test:",trainortest.title())
    print("Accuracy:",acc)
    print("Precision:",prec)
    print("Recall:",rec) 
    print("F1:",f1)
    row_num = min((np.unique(y_true).size,9))
    f = plt.figure(figsize=(row_num*2,row_num*2))
    ax = f.gca()
    cmdisp = ConfusionMatrixDisplay.from_predictions(y_true,y_pred,
                                                     display_labels=labels if labels is not None else np.unique(y_true),
                                                     cmap='Greys',ax=ax)
    plt.show()
    results = {
        "Estimator":name,
        "Train/Test":trainortest.title(),
        "Accuracy":acc,
        "Precision":prec,
        "Recall":rec,
        "F1":f1,
        "Confusion Matrix":cmdisp
    }
    return results