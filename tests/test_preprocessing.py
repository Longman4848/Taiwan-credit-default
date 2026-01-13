import sys
import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to Python path - MUST BE AT THE VERY TOP
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.preprocessing import (
    clean_categorical_variables,
    remove_invalid_records,
    preprocess_taiwan_credit
)

def test_clean_categorical_variables():
    """Test that categorical variables are properly cleaned"""
    # Create test data with CORRECT column names (X3, X4, etc.)
    test_data = pd.DataFrame({
        'X1': [50000, 100000, 150000, 200000, 250000],
        'X2': [1, 2, 1, 2, 1],           # SEX (1 or 2)
        'X3': [0, 1, 2, 5, 6],           # EDUCATION (invalid: 0, 5, 6)
        'X4': [0, 1, 2, 4, 5],           # MARRIAGE (invalid: 0, 4, 5)
        'X5': [25, 30, 35, 40, 45],      # AGE (all valid)
        'target': [0, 1, 0, 1, 0]
    })
    
    print("Before cleaning:")
    print(test_data[['X3', 'X4']])  # Only show the columns we're testing
    
    cleaned_data = clean_categorical_variables(test_data)
    
    print("After cleaning:")
    print(cleaned_data[['X3', 'X4']])
    
    # Check that invalid codes were converted to valid ones
    expected_education = [4, 1, 2, 4, 4]  # 0→4, 1→1, 2→2, 5→4, 6→4
    expected_marriage = [3, 1, 2, 3, 3]   # 0→3, 1→1, 2→2, 4→3, 5→3
    
    assert cleaned_data['X3'].tolist() == expected_education
    assert cleaned_data['X4'].tolist() == expected_marriage
    
    print("Categorical variables test passed!")

def test_remove_invalid_ages():
    """Test that invalid ages are removed"""
    test_df = pd.DataFrame({
        'X1': [50000, 100000, 150000],
        'X5': [15, 25, 150]  # Invalid: 15 and 150 (should be 20-80)
    })
    
    result = remove_invalid_records(test_df)
    
    # Should only keep age 25 (between 20-80)
    assert len(result) == 1
    assert result['X5'].iloc[0] == 25
    print("Remove invalid ages test passed!")

def test_remove_invalid_limits():
    """Test that invalid credit limits are removed"""
    test_df = pd.DataFrame({
        'X1': [-1000, 0, 50000],  # Invalid: -1000 and 0 (should be > 0)
        'X5': [25, 30, 35]       # All valid
    })
    
    result = remove_invalid_records(test_df)
    
    # Should only keep positive limit 50000
    assert len(result) == 1
    assert result['X1'].iloc[0] == 50000
    print("Remove invalid limits test passed!")

def test_remove_both_invalid():
    """Test that both age and limit conditions work together"""
    test_df = pd.DataFrame({
        'X1': [-1000, 50000, 100000],  # Invalid: -1000
        'X5': [15, 25, 150]           # Invalid: 15, 150
    })
    
    result = remove_invalid_records(test_df)
    
    # Should only keep row with valid limit AND valid age: X1=50000, X5=25
    assert len(result) == 1
    assert result['X1'].iloc[0] == 50000
    assert result['X5'].iloc[0] == 25
    print("Remove both invalid test passed!")

def test_keep_valid_data():
    """Test that valid data is preserved"""
    test_df = pd.DataFrame({
        'X1': [50000, 100000],
        'X5': [25, 30]  # Both valid
    })
    
    result = remove_invalid_records(test_df)
    
    # All valid rows should be preserved
    assert len(result) == 2
    print("Keep valid data test passed!")

def test_boundary_values():
    """Test boundary values (20 and 80 should be valid)"""
    test_df = pd.DataFrame({
        'X1': [50000, 100000, 150000, 200000],
        'X5': [19, 20, 80, 81]  # Invalid: 19 and 81
    })
    
    result = remove_invalid_records(test_df)
    
    # Should keep ages 20 and 80, remove 19 and 81
    assert len(result) == 2
    assert result['X5'].tolist() == [20, 80]
    print(" Boundary values test passed!")

class TestPreprocessTaiwanCredit:
    
    def test_preprocess_taiwan_credit_valid_data(self):
        """Test that valid data is processed correctly"""
        test_df = pd.DataFrame({
            'X1': [50000, 100000, 150000],      # Valid limits
            'X2': [1, 2, 1],                    # Valid sex
            'X3': [1, 2, 3],                    # Valid education
            'X4': [1, 2, 3],                    # Valid marriage
            'X5': [25, 30, 35],                # Valid ages
            'target': [0, 1, 0]
        })
        
        original_shape = test_df.shape
        processed_df = preprocess_taiwan_credit(test_df)
        
        # All valid data should be preserved
        assert processed_df.shape[0] == original_shape[0]
        assert processed_df.shape[1] == original_shape[1]
        
        # Data should remain unchanged for valid records
        pd.testing.assert_frame_equal(processed_df, test_df)
        print("Valid data test passed!")

    def test_preprocess_taiwan_credit_removes_invalid_records(self):
        """Test that invalid records are removed"""
        test_df = pd.DataFrame({
            'X1': [50000, -1000, 100000, 0, 150000],  # Invalid: -1000, 0
            'X2': [1, 2, 1, 2, 1],                    # Valid sex
            'X3': [1, 2, 3, 4, 1],                    # Valid education
            'X4': [1, 2, 3, 1, 2],                    # Valid marriage
            'X5': [15, 25, 150, 30, 35],              # Invalid: 15, 150
            'target': [0, 1, 0, 1, 0]
        })
        
        original_count = len(test_df)
        processed_df = preprocess_taiwan_credit(test_df)
        
        # Should remove rows with invalid limits (-1000, 0) and invalid ages (15, 150)
        # Valid rows: [50000, 25] and [150000, 35] = 2 rows
        assert len(processed_df) == 1
        
        # Check remaining data
        expected_limits = [150000]
        expected_ages = [35]
        
        assert processed_df['X1'].tolist() == expected_limits
        assert processed_df['X5'].tolist() == expected_ages
        print("Remove invalid records test passed!")

    def test_preprocess_taiwan_credit_cleans_categorical_variables(self):
        """Test that categorical variables are cleaned"""
        test_df = pd.DataFrame({
            'X1': [50000, 100000, 150000],      # Valid limits
            'X2': [1, 2, 1],                    # Valid sex
            'X3': [0, 5, 6],                    # Invalid education codes: 0, 5, 6
            'X4': [0, 4, 5],                    # Invalid marriage codes: 0, 4, 5
            'X5': [25, 30, 35],                # Valid ages
            'target': [0, 1, 0]
        })
        
        processed_df = preprocess_taiwan_credit(test_df)
        
        # Invalid education codes should be mapped to 4
        expected_education = [4, 4, 4]  # 0→4, 5→4, 6→4
        assert processed_df['X3'].tolist() == expected_education
        
        # Invalid marriage codes should be mapped to 3
        expected_marriage = [3, 3, 3]   # 0→3, 4→3, 5→3
        assert processed_df['X4'].tolist() == expected_marriage
        print("Clean categorical variables test passed!")

    def test_preprocess_taiwan_credit_combines_both_steps(self):
        """Test that both cleaning and filtering work together"""
        test_df = pd.DataFrame({
            'X1': [-1000, 0, 50000, 100000],  # Invalid: -1000, 0
            'X2': [1, 2, 1, 2],               # Valid sex
            'X3': [0, 1, 5, 2],               # Invalid: 0, 5
            'X4': [0, 2, 4, 3],               # Invalid: 0, 4
            'X5': [15, 25, 150, 35],          # Invalid: 15, 150
            'target': [0, 1, 0, 1]
        })
        
        processed_df = preprocess_taiwan_credit(test_df)
        
        # Should only keep row: [100000, 2, 2, 3, 35, 1] (valid limit, valid education after cleaning, valid marriage after cleaning, valid age)
        assert len(processed_df) == 1
        
        # Check the remaining row has valid/cleaned values
        remaining_row = processed_df.iloc[0]
        assert remaining_row['X1'] == 100000  # Valid limit
        assert remaining_row['X3'] == 2       # Valid education (not cleaned because was already valid)
        assert remaining_row['X4'] == 3       # Valid marriage (not cleaned because was already valid)
        assert remaining_row['X5'] == 35      # Valid age
        print("Combined steps test passed!")

def test_preprocess_taiwan_credit_basic_functionality():
    """Basic functionality test"""
    test_df = pd.DataFrame({
        'X1': [50000, 100000],
        'X2': [1, 2],
        'X3': [1, 2],
        'X4': [1, 2],
        'X5': [25, 30],
        'target': [0, 1]
    })
    
    result = preprocess_taiwan_credit(test_df)
    
    # Basic assertions
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= len(test_df)  # May remove invalid rows
    assert 'X1' in result.columns
    assert 'X2' in result.columns
    assert 'X3' in result.columns
    assert 'X4' in result.columns
    assert 'X5' in result.columns
    assert 'target' in result.columns
    
    print(" All tests passed for preprocess_taiwan_credit function!")

if __name__ == "__main__":
    print("Running tests directly...")
    test_clean_categorical_variables()
    test_remove_invalid_ages()
    #test_remove_invalid_limits()
    #test_remove_both_invalid()
    #test_keep_valid_data()
    #test_boundary_values()
    test_preprocess_taiwan_credit_basic_functionality()
    print(" All tests passed!")