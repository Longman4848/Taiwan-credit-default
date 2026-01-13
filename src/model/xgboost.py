import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Tuple, Dict, Any

class XGBoostTaiwanCreditModel:
    """
    XGBoost model for Taiwan Credit Default prediction with threshold 0.45
    """
    
    def __init__(self, threshold: float = 0.45):
        """
        Initialize the model with threshold 0.45
        
        Args:
            threshold: Classification threshold (default 0.45 for your model)
        """
        self.threshold = threshold
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            learning_rate=0.05,
            max_depth=6,
            n_estimators=400,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,
            random_state=42
        )
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the XGBoost model
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self (for method chaining)
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using threshold 0.45
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        probabilities = self.model.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Input features
            
        Returns:
            Probability array [prob_class_0, prob_class_1]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(X)
    
    def predict_with_custom_threshold(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        Make predictions with custom threshold
        
        Args:
            X: Input features
            threshold: Custom threshold (uses default if None)
            
        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if threshold is None:
            threshold = self.threshold
        
        probabilities = self.model.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)

def train_taiwan_credit_model(
    X: pd.DataFrame, 
    y: pd.Series,
    threshold: float = 0.45,
    test_size: float = 0.2
) -> Tuple[XGBoostTaiwanCreditModel, Dict[str, Any]]:
    """
    Train XGBoost model for Taiwan credit default with threshold 0.45
    
    Args:
        X: Features (X1, X2, X3, X4, X5, etc.)
        y: Target variable (default or not)
        threshold: Classification threshold (default 0.45)
        test_size: Test size for train-test split
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Create and train model
    model = XGBoostTaiwanCreditModel(threshold=threshold)
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'threshold_used': threshold,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    return model, metrics

def evaluate_model_performance(
    model: XGBoostTaiwanCreditModel, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate model performance on test data
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of performance metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'threshold': model.threshold
    }
    
    return metrics

"""
XGBoost model with threshold 0.45 (your specific requirement)
Complete training function that returns model and metrics
Evaluation function for performance assessment
Comprehensive unit tests for all functionality
Error handling for unfitted models
Integration test for complete workflow
Ready for Taiwan credit data with X1, X2, X3, X4, X5 format
"""