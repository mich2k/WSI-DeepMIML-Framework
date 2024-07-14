from methods.baseline import Baseline
from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from utils import compute_metrics

class BRkNN(Baseline):
    def __init__(self, config):
        super(BRkNN, self).__init__(is_pytorch_model=False)
        self.k = config.k
        self.epoch = config.epoch
        self.parameters = {'k': range(1,self.k)}
        self._classifier_a = BRkNNaClassifier(k=self.k)
        self._classifier_b = BRkNNbClassifier(k=self.k)
        self.clf_a = GridSearchCV(self._classifier_a, self.parameters, refit='accuracy', return_train_score=True, cv=5)
        self.clf_b = GridSearchCV(self._classifier_b, self.parameters, refit='accuracy', return_train_score=True, cv=5)

        
    def train(self, bag_set, labels):
        
        X_train, X_test, y_train, y_test = train_test_split(bag_set, labels, test_size=0.33)
        
        # Transform X_train, y_train to array
        X_train = X_train.detach().numpy()
        y_train = y_train.detach().numpy()
        X_test = X_test.detach().numpy()
        y_test = y_test.detach().numpy()
        
        # Reshape        
        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train = y_train.reshape(y_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        y_test = y_test.reshape(y_test.shape[0], -1)
        
        
        # Train classifier
        self.clf_a.fit(X_train, y_train)
        predictions_a = self.clf_a.predict(X_test) 
        
        self.clf_b.fit(X_train, y_train)
        predictions_b = self.clf_b.predict(X_test) 
        
        accuracy_a, precision_a, recall_a, f1_a = compute_metrics(predictions_a, y_test) 
        accuracy_b, precision_b, recall_b, f1_b = compute_metrics(predictions_b, y_test)
        
        
        print(f"Best parameters set found on development set: {self.clf_a.best_params_}, {self.clf_b.best_params_}")
        print(f"BRkNNaClassifier: Accuracy: {accuracy_a}, Precision: {precision_a}, Recall: {recall_a}, F1: {f1_a}")
        print(f"BRkNNbClassifier: Accuracy: {accuracy_b}, Precision: {precision_b}, Recall: {recall_b}, F1: {f1_b}")
        
    def test(self, bag_set, labels):        
        _, X_test, _, y_test = train_test_split(bag_set, labels, test_size=0.66)
        
        # Transform X_train, y_train to array
        X_test = X_test.detach().numpy()
        y_test = y_test.detach().numpy()
        
        # Reshape        
        X_test = X_test.reshape(X_test.shape[0], -1)
        y_test = y_test.reshape(y_test.shape[0], -1)
        
        predictions_a = self.clf_a.predict(X_test)
        predictions_b = self.clf_b.predict(X_test) 
        
        accuracy_a, precision_a, recall_a, f1_a = compute_metrics(predictions_a, y_test) 
        accuracy_b, precision_b, recall_b, f1_b = compute_metrics(predictions_b, y_test)
        
        print(f"BRkNNaClassifier: Accuracy: {accuracy_a}, Precision: {precision_a}, Recall: {recall_a}, F1: {f1_a}")
        print(f"BRkNNbClassifier: Accuracy: {accuracy_b}, Precision: {precision_b}, Recall: {recall_b}, F1: {f1_b}")   
        
        
    def run(self, trainset, testset):
        
        trainloader = DataLoader(trainset, batch_size=20, shuffle=True, pin_memory=True)
        testloader = DataLoader(testset, batch_size=20, shuffle=True, pin_memory=True)
        
        for _ in range(self.epoch):
            print(f"Train Epoch: {_}")
            for inputs, labels in trainloader:
                self.train(inputs, labels)
        for _ in range(self.epoch):        
            print(f"Test Epoch: {_}")
            for inputs, labels in testloader:
                self.test(inputs, labels)