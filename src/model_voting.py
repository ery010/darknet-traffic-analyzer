# Module for voting ensemble

import numpy as np
import pickle
import os
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
    
class VotingEnsemble(object):
    
    def __init__(self, models):
        self.models = models
        self.voting_clf = None
        self.accuracy = -np.inf
        self.classification_report = None
        
    def fit_voting_classifier(self, X_train, X_test, y_train, y_test, voting="hard"):
        """
        params:
            X_train: train data
            X_test: test data
            y_train: train labels
            y_test: test labels
            voting: vote type
        """
        estimators = []
        for model in self.models:
            estimators.append((str(model), model))
            
        self.voting_clf = VotingClassifier(estimators=estimators)
        self.voting_clf.fit(X_train, y_train)
        
        y_pred = self.voting_clf.predict(X_test)
        self.classification_report = classification_report(y_test, y_pred)
        
        self.accuracy = self.voting_clf.score(X_test, y_test)
        
    def get_accuracy(self):
        """
        returns: Voting classifier accuracy
        """
        assert self.accuracy != -np.inf, "Must fit voting classifier."
        
        return self.accuracy
    
    def get_classification_report(self):
        """
        returns: Model classification report
        """
        assert self.classification_report, "Must fit voting classifier."
        
        return self.classification_report
        
    def save_model(self, name, directory):
        """
        params:
            name: Model name
            directory: File directory to save model
        """
        assert self.voting_clf, "Must fit voting classifier."
        
        filepath = os.path.join(directory, name)
        with open(filepath, 'wb') as file:
            pickle.dump(self.voting_clf, file)
        
        
        
            