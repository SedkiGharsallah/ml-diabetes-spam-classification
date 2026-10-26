import unittest
import numpy as np
from functions import balance_data_smote


class TestBalanceDataSmote(unittest.TestCase):
    """Unit tests for the balance_data_smote function."""
    
    def test_balances_imbalanced_dataset(self):
        """Test that SMOTE increases the number of samples for minority class."""
        # Create imbalanced dataset with enough samples for SMOTE (needs at least 6 for minority class)
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                      [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16]])
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 9 class 0, 6 class 1
        
        X_resampled, y_resampled = balance_data_smote(X, y, random_state=42)
        
        # Check that both classes have equal representation
        unique, counts = np.unique(y_resampled, return_counts=True)
        self.assertEqual(counts[0], counts[1], "Classes should be balanced after SMOTE")
        
    def test_output_shape(self):
        """Test that output arrays have correct shapes."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                      [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16]])
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        
        X_resampled, y_resampled = balance_data_smote(X, y, random_state=42)
        
        # Check shapes match
        self.assertEqual(X_resampled.shape[0], y_resampled.shape[0], 
                        "X and y should have same number of samples")
        self.assertEqual(X_resampled.shape[1], X.shape[1], 
                        "Number of features should remain the same")
        
    def test_increases_sample_count(self):
        """Test that SMOTE increases the total number of samples."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                      [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16]])
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        
        X_resampled, y_resampled = balance_data_smote(X, y, random_state=42)
        
        self.assertGreater(len(X_resampled), len(X), 
                          "SMOTE should increase the number of samples")


if __name__ == '__main__':
    unittest.main()
