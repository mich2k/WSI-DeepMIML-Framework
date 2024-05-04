from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

def cross_validation(bags, labels, model, folds, parameters = {}):  
    skf = StratifiedKFold(n_splits = folds)
    results_accuracy_model = []
    fold = 0
    
    
    for train_index, test_index in skf.split(bags, labels):
        X_train = [bags[i] for i in train_index]
        Y_train = labels[train_index]
        X_test  = [bags[i] for i in test_index]
        Y_test  = labels[test_index]
        
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
			
        #Calculation of Accuracy
        accuracy_model = np.average(Y_test.T == np.sign(predictions)) 
        results_accuracy_model.append(100 * accuracy_model)
        fold = fold + 1
        
    return np.mean(results_accuracy_model)