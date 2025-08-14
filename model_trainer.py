"""
Model Training and Evaluation Module
This module implements a ModelTrainer class for training and evaluating both classification and regression models
for movie success prediction. It includes cross-validation, hyperparameter tuning, and performance metrics calculation.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                           mean_absolute_error, mean_squared_error, r2_score)
import pandas as pd

class ModelTrainer:
    """
    A class for training and evaluating machine learning models
    Handles both classification and regression tasks with cross-validation
    """
    def __init__(self, n_splits=10):
        """
        Initialize the ModelTrainer with number of cross-validation splits
        Args:
            n_splits (int): Number of folds for cross-validation
        """
        self.n_splits = n_splits
        self.scaler = StandardScaler()
        
    def prepare_data(self, X, y):
        """
        Prepare data by scaling features
        Args:
            X: Feature matrix
            y: Target variable
        Returns:
            Scaled feature matrix and target variable
        """
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
    
    def train_classification_models(self, X, y):
        """
        Train and evaluate classification models using stratified k-fold cross-validation
        Implements Random Forest, Logistic Regression, and Gradient Boosting
        Args:
            X: Feature matrix
            y: Binary target variable
        Returns:
            Dictionary containing model performance metrics and trained models
        """
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        
        results = {}
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for name, model in models.items():
            accuracies = []
            roc_aucs = []
            f1_scores = []
            
            # Train the model on the full dataset for final model storage
            model.fit(X, y)
            
            for train_idx, val_idx in skf.split(X, y):
                if isinstance(X, pd.DataFrame):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                else:
                    X_train, X_val = X[train_idx], X[val_idx]
                    
                if isinstance(y, pd.Series):
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                else:
                    y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                accuracies.append(accuracy_score(y_val, y_pred))
                roc_aucs.append(roc_auc_score(y_val, y_pred_proba))
                f1_scores.append(f1_score(y_val, y_pred))
            
            results[name] = {
                'model': model,
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'mean_roc_auc': np.mean(roc_aucs),
                'std_roc_auc': np.std(roc_aucs),
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores)
            }
        
        return results
    
    def train_regression_models(self, X, y):
        """
        Train and evaluate regression models using k-fold cross-validation
        Implements Random Forest and Gradient Boosting for revenue prediction
        Args:
            X: Feature matrix
            y: Continuous target variable (revenue)
        Returns:
            Dictionary containing model performance metrics and trained models
        """
        models = {
            'RandomForest': RandomForestRegressor(random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42)
        }
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        results = {}
        
        for name, model in models.items():
            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                if isinstance(X, pd.DataFrame):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                else:
                    X_train, X_val = X[train_idx], X[val_idx]
                    
                if isinstance(y, pd.Series):
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                else:
                    y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                scores = {
                    'fold': fold,
                    'mae': mean_absolute_error(y_val, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                    'r2': r2_score(y_val, y_pred)
                }
                
                fold_scores.append(scores)
            
            results[name] = {
                'model': model,
                'scores': fold_scores,
                'mean_mae': np.mean([s['mae'] for s in fold_scores]),
                'std_mae': np.std([s['mae'] for s in fold_scores]),
                'mean_rmse': np.mean([s['rmse'] for s in fold_scores]),
                'std_rmse': np.std([s['rmse'] for s in fold_scores]),
                'mean_r2': np.mean([s['r2'] for s in fold_scores]),
                'std_r2': np.std([s['r2'] for s in fold_scores])
            }
        
        return results
    
    def tune_random_forest(self, X, y, task='classification'):
        """
        Perform hyperparameter tuning for Random Forest models
        Uses GridSearchCV to find optimal parameters
        Args:
            X: Feature matrix
            y: Target variable
            task: Either 'classification' or 'regression'
        Returns:
            Best parameters and corresponding score
        """
        if task == 'classification':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            scoring = 'accuracy'
        else:  # regression
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            scoring = 'r2'
        
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring=scoring,
            n_jobs=-1
        )
        
        search.fit(X, y)
        return search.best_params_, search.best_score_ 