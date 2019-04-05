# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:12:45 2018

@author: 60236
"""

import numpy as np
from sklearn import metrics
import torch
import os
import pickle
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def eval_model(y_preds, y_true, n):
    '''
    input:
        y_preds: ?,3,x,x  Float tensor
        y_true : ?,3,x,x  Float tensor
        
    return:
        accuracy, precision, recall
    
    '''
    
    preds_np = y_preds.cpu().detach().numpy()
    true_np = y_true.cpu().numpy()
    
    preds_np[preds_np>=0.5] = 1.
    preds_np[preds_np <0.5] = 0.
    
   # false_positives = get_false_positives(preds_np, true_np)
    
    y_true = true_np.flatten()
    y_predicted = preds_np.flatten()
    
    accuracy = metrics.accuracy_score(y_true, y_predicted)
    precision = metrics.precision_score(y_true, y_predicted)
    recall = metrics.recall_score(y_true, y_predicted)
    
    os.makedirs('./output', exist_ok=True)
    precision_recall_curve(y_true, y_predicted, n, './output')
    
    return accuracy, precision, recall

def get_false_positives(predictions, labels):
    """Get false positives for the given predictions and labels."""
    
    FP = np.logical_and(predictions == 1, labels == 0)
    false_positives = np.copy(predictions)
    false_positives[FP] = 1
    false_positives[np.logical_not(FP)] = 0
    return false_positives
    
    
    
def precision_recall_curve(y_true, y_predicted, n, out_path):
    """Create a PNG with the precision-recall curve for our predictions."""

    print("Calculate precision recall curve.")
    precision, recall, thresholds = metrics.precision_recall_curve(y_true,
                                                                   y_predicted)

    # Save the raw precision and recall results to a pickle since we might want
    # to analyse them later.
    out_file = os.path.join(out_path, "precision_recall.pickle")
    with open(out_file, "wb") as out:
        pickle.dump({
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds
        }, out)

    # Create the precision-recall curve.
    out_file = os.path.join(out_path, "precision_recall_{}.png".format(n))
    plt.clf()
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(out_file)