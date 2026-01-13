import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.model.xgboost import (
    XGBoostTaiwanCreditModel, 
    train_taiwan_credit_model, 
    evaluate_model_performance
)

class TestXGBoostTaiwanCreditModel:
    
    def test_init_default_threshold(self):
        """Test initialization with default threshold 0.45"""
        model = XGBoostTaiwanCreditModel()
        
        assert model.threshold == 0.45
        assert model.is_fitted is False
    
    def test_init_custom_threshold(self):
        """Test initialization with custom threshold"""
        model = XGBoostTaiwanCreditModel(threshold=0.3)
        
        assert model.threshold == 0.3
        assert model.is_fitted is False
    
    def test_fit_method(self):
        """Test that fit method trains the model"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(10)])
        y_series = pd.Series(y)
        
        model = XGBoostTaiwanCreditModel(threshold=0.45)
        fitted_model = model.fit(X_df, y_series)
        
        # Should return self
        assert fitted_model is model
        assert model.is_fitted is True
        
        # Model should be trained
        assert hasattr(model.model, 'predict')
        assert hasattr(model.model, 'predict_proba')
    
    def test_predict_requires_fitted_model(self):
        """Test that predict raises error if model not fitted"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(10)])
        
        model = XGBoostTaiwanCreditModel(threshold=0.45)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X_df)
    
    def test_predict_returns_binary_predictions(self):
        """Test that predict returns binary predictions"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        X_train_df = pd.DataFrame(X_train, columns=[f'X{i+1}' for i in range(10)])
        X_test_df = pd.DataFrame(X_test, columns=[f'X{i+1}' for i in range(10)])
        y_train_series = pd.Series(y_train)
        
        model = XGBoostTaiwanCreditModel(threshold=0.45)
        model.fit(X_train_df, y_train_series)
        
        predictions = model.predict(X_test_df)
        
        # Predictions should be binary (0 or 1)
        assert set(predictions).issubset({0, 1})
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_proba_returns_probabilities(self):
        """Test that predict_proba returns probabilities"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        X_train_df = pd.DataFrame(X_train, columns=[f'X{i+1}' for i in range(10)])
        X_test_df = pd.DataFrame(X_test, columns=[f'X{i+1}' for i in range(10)])
        y_train_series = pd.Series(y_train)
        
        model = XGBoostTaiwanCreditModel(threshold=0.45)
        model.fit(X_train_df, y_train_series)
        
        probabilities = model.predict_proba(X_test_df)
        
        # Should have 2 columns (class 0 and class 1 probabilities)
        assert probabilities.shape[1] == 2
        assert len(probabilities) == len(X_test)
        # Probabilities should be between 0 and 1
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
        assert isinstance(probabilities, np.ndarray)
    
    def test_predict_with_custom_threshold(self):
        """Test predict_with_custom_threshold with different thresholds"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        X_train_df = pd.DataFrame(X_train, columns=[f'X{i+1}' for i in range(10)])
        X_test_df = pd.DataFrame(X_test, columns=[f'X{i+1}' for i in range(10)])
        y_train_series = pd.Series(y_train)
        
        model = XGBoostTaiwanCreditModel(threshold=0.45)
        model.fit(X_train_df, y_train_series)
        
        # Test with default threshold (0.45)
        pred_default = model.predict_with_custom_threshold(X_test_df)
        
        # Test with custom threshold (0.6)
        pred_custom = model.predict_with_custom_threshold(X_test_df, threshold=0.6)
        
        assert len(pred_default) == len(pred_custom) == len(X_test)
        # Different thresholds should generally produce different predictions
    
    def test_predict_with_custom_threshold_uses_default_if_none(self):
        """Test that predict_with_custom_threshold uses default threshold when None"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        X_train_df = pd.DataFrame(X_train, columns=[f'X{i+1}' for i in range(10)])
        X_test_df = pd.DataFrame(X_test, columns=[f'X{i+1}' for i in range(10)])
        y_train_series = pd.Series(y_train)
        
        model = XGBoostTaiwanCreditModel(threshold=0.45)
        model.fit(X_train_df, y_train_series)
        
        pred_default = model.predict(X_test_df)
        pred_with_none = model.predict_with_custom_threshold(X_test_df, threshold=None)
        
        # Should be the same
        np.testing.assert_array_equal(pred_default, pred_with_none)

class TestTrainTaiwanCreditModel:
    
    def test_train_taiwan_credit_model_basic(self):
        """Test basic model training with threshold 0.45"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(10)])
        y_series = pd.Series(y)
        
        model, metrics = train_taiwan_credit_model(X_df, y_series, threshold=0.45)
        
        # Should return a trained model with threshold 0.45
        assert model is not None
        assert model.threshold == 0.45
        assert model.is_fitted is True
        assert hasattr(model.model, 'predict')
        
        # Should return metrics
        assert isinstance(metrics, dict)
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'threshold_used']
        for metric in expected_metrics:
            assert metric in metrics
    
    def test_train_taiwan_credit_model_with_different_threshold(self):
        """Test model training with different threshold"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(10)])
        y_series = pd.Series(y)
        
        model, metrics = train_taiwan_credit_model(X_df, y_series, threshold=0.3)
        
        assert model.threshold == 0.3
        assert metrics['threshold_used'] == 0.3
    
    def test_train_taiwan_credit_model_predicts(self):
        """Test that trained model can make predictions"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(10)])
        y_series = pd.Series(y)
        
        model, metrics = train_taiwan_credit_model(X_df, y_series, threshold=0.45)
        
        predictions = model.predict(X_df)
        
        # Should make valid predictions
        assert len(predictions) == len(X_df)
        assert set(predictions).issubset({0, 1})
    
    def test_train_taiwan_credit_model_returns_metrics(self):
        """Test that training returns reasonable metrics"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(10)])
        y_series = pd.Series(y)
        
        model, metrics = train_taiwan_credit_model(X_df, y_series, threshold=0.45)
        
        # Check that metrics are reasonable
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1

class TestEvaluateModelPerformance:
    
    def test_evaluate_model_performance_basic(self):
        """Test that evaluate_model_performance returns correct metrics"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        X_train_df = pd.DataFrame(X_train, columns=[f'X{i+1}' for i in range(10)])
        X_test_df = pd.DataFrame(X_test, columns=[f'X{i+1}' for i in range(10)])
        y_train_series = pd.Series(y_train)
        y_test_series = pd.Series(y_test)
        
        model = XGBoostTaiwanCreditModel(threshold=0.45)
        model.fit(X_train_df, y_train_series)
        
        metrics = evaluate_model_performance(model, X_test_df, y_test_series)
        
        # Should return metrics dictionary
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'threshold']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1

def test_complete_workflow():
    """Integration test for complete XGBoost workflow with threshold 0.45"""
    # Create sample data
    X, y = make_classification(
        n_samples=200, 
        n_features=10, 
        n_informative=8, 
        n_redundant=2,
        random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(10)])
    y_series = pd.Series(y)
    
    # Train model with threshold 0.45
    model, train_metrics = train_taiwan_credit_model(X_df, y_series, threshold=0.45)
    
    # Evaluate model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test_df = pd.DataFrame(X_test, columns=[f'X{i+1}' for i in range(10)])
    y_test_series = pd.Series(y_test)
    
    eval_metrics = evaluate_model_performance(model, X_test_df, y_test_series)
    
    # Verify everything works together
    assert model.threshold == 0.45  # The exact threshold you want
    assert model.is_fitted is True
    
    # Check that metrics are reasonable
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        assert 0 <= train_metrics[metric] <= 1
        assert 0 <= eval_metrics[metric] <= 1
    
    print(f"Model trained with threshold {model.threshold}")
    print(f"Train accuracy: {train_metrics['accuracy']:.3f}")
    print(f"ROC-AUC: {train_metrics['roc_auc']:.3f}")

if __name__ == "__main__":
    test_complete_workflow()
    print("All tests passed!")

    #$env:PYTHONPATH = "$(Get-Location)"; pytest tests\test_xgboost.py -v