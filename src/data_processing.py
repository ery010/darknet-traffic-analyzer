# Module for processing data and feature engineering

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor(object):
    
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath, on_bad_lines='skip')
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data_label = None
        
    def preprocess(self):
        self.data = self.data.dropna()
        
    def feature_engineering(self, label, corr_threshold=0.9, dropped=None, label_map=None, one_hot_columns=None):
        """
        params:
            label: Assigned label
            corr_threshold: Correlation threshold for dropping features
            dropped: List of columns to drop
            label_map: Mapping for encoding labels
            one_hot_columns: Columns to one-hot encode
        """
        self.data_label = label
        
        # Map labels, if not None
        if label_map:
            assert type(label_map) == dict, "Label map must be a dictionary."
            self.data[label] = self.data[label].map(label_map)
        
        # Drop columns, if not None
        if dropped:
            self.data = self.data.drop(columns=list(dropped))
        
        # One-hot encode, if not None
        if one_hot_columns:
            one_hot = pd.get_dummies(self.data, columns=list(one_hot_columns))
            self.data = one_hot.iloc[:len(self.data)]
        
        # Drop features based on correlation matrix
        corr_matrix = self.data.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        dropped_features = [feature for feature in upper_tri.columns if any(upper_tri[feature] > corr_threshold)]
        self.data = self.data.drop(columns=dropped_features)
        
    def create_train_test(self, test_size=0.2, random_state=0):
        """
        params:
            test_size: Proportion of data assigned as test data
            random_state: Random state variable assigned to train_test_split
        """
        assert self.data_label, "Label must be assigned, call feature_engineering before create_train_test."
        
        # Split data into train and test sets
        X = self.data.drop(columns=[self.data_label])
        y = self.data[self.data_label]

        # Replace inf and -inf values with max/min float values
        X[X == np.inf] = np.finfo(np.float32).max
        X[X == -np.inf] = np.finfo(np.float32).min
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        
    def get_X_train(self):
        """
        returns: X_train data
        """
        return self.X_train
    
    def get_X_test(self):
        """
        returns: X_test data
        """
        return self.X_test
    
    def get_y_train(self):
        """
        returns: y_train labels
        """
        return self.y_train
    
    def get_y_test(self):
        """
        returns: y_test labels
        """
        return self.y_test
    
    def save_data(self, name, directory):
        """
        params:
            directory: File directory to save data.
        """
        X_train_fp = os.path.join(directory, name + "_X_train.csv")
        X_test_fp = os.path.join(directory, name + "_X_test.csv")
        y_train_fp = os.path.join(directory, name + "_y_train.csv")
        y_test_fp = os.path.join(directory, name + "_y_test.csv")
        
        self.X_train.to_csv(X_train_fp, index=False)
        self.X_test.to_csv(X_test_fp, index=False)
        self.y_train.to_csv(y_train_fp, index=False)
        self.y_test.to_csv(y_test_fp, index=False)
        
        
        