from aggregators.models.aggregator.aggregator import Aggregator
from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

class BRkNN(Aggregator):
    def __init__(self, k=5, is_pytorch_model=False):
        super(BRkNN, self).__init__(is_pytorch_model)
        self._classifier_a = BRkNNaClassifier(k=k)
        self._classifier_b = BRkNNbClassifier(k=k)
        self.parameters = {'k': range(1,k)}
        
    def train(self, inputs, labels):
        
        # Flattening input data
        inputs = inputs.reshape(inputs.shape[0], -1)
        
        X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.33)
        
        # Train classifier
        print('Training BRkNNaClassifier')
        clf_a = GridSearchCV(self._classifier_a, self.parameters, refit='accuracy', return_train_score=True, cv=5)
        clf_a.fit(X_train, y_train)        
        predictions_a = clf_a.predict(X_test)
        
        print('Training BRkNNbClassifier')
        clf_b = GridSearchCV(self._classifier_b, self.parameters, refit='accuracy', return_train_score=True, cv=5)
        clf_b.fit(X_train, y_train)
        predictions_b = clf_b.predict(X_test) 
        
        accuracy_score_a = accuracy_score(y_test, predictions_a)
        accuracy_score_b = accuracy_score(y_test, predictions_b)
        
        print(f"Best parameters set found on development set: {clf_a.best_params_}, {clf_b.best_params_}")
        print(f'Accuracy score for BRkNNaClassifier: {accuracy_score_a}')
        print(f'Accuracy score for BRkNNbClassifier: {accuracy_score_b}')