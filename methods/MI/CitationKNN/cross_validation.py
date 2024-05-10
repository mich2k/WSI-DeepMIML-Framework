from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils import compute_metrics
import numpy as np

def cross_validation(bags, labels, model, folds, parameters = {}, test = False):  
    skf = StratifiedKFold(n_splits = folds)
    results_accuracy_model = []
    results_precision_model = []
    results_recall_model = []
    results_f1_model = []
    fold = 0
    
    
    for train_index, test_index in skf.split(bags, labels):
        X_train = [bags[i] for i in train_index]
        Y_train = labels[train_index]
        X_test  = [bags[i] for i in test_index]
        Y_test  = labels[test_index]
        
        if not test:
            if len(parameters) > 0: 
                model.fit(X_train, Y_train, **parameters)
            else: 
                model.fit(bags, labels)
                
        #print("Starting")
        predictions = model.predict(X_test)
        #print("End prediction in mil_cross_val. Prediction result from model 1:")
        #print(predictions_1)

        if (isinstance(predictions, tuple)):
            predictions = predictions[1]	
			
        #Calculation of metrics
        accuracy_model, precision_model, recall_model, f1_model = compute_metrics(predictions, Y_test)
        results_accuracy_model.append(100 * accuracy_model)
        results_precision_model.append(100 * precision_model)
        results_recall_model.append(100 * recall_model)
        results_f1_model.append(100 * f1_model)
        print(f'Fold {fold} - Accuracy: {100 * accuracy_model}, Precision: {100 * precision_model}, Recall: {100 * recall_model}, F1: {100 * f1_model}')
        fold = fold + 1
        
    return np.mean(results_accuracy_model), np.mean(results_precision_model), np.mean(results_recall_model), np.mean(results_f1_model)