import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from functions import permutation_importance_numpy


class TestPermutationImportanceNumpy(unittest.TestCase):
    """Unit tests for the permutation_importance_numpy function."""
    
    def setUp(self):
        """Set up test data and fitted model for all tests."""
        # Create simple dataset
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        # Make y dependent on first two features
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)
        
        # Fit a simple model
        self.model = LogisticRegression(random_state=42)
        self.model.fit(self.X, self.y)
    
    def test_returns_correct_shape(self):
        """Test that function returns arrays with correct shape."""
        mean, std = permutation_importance_numpy(self.model, self.X, self.y, 
                                                  n_repeats=10, random_state=42)
        
        # Should return importance for each feature
        self.assertEqual(len(mean), self.X.shape[1], 
                        "Mean importance should have one value per feature")
        self.assertEqual(len(std), self.X.shape[1], 
                        "Std importance should have one value per feature")
    
    def test_important_features_have_higher_scores(self):
        """Test that important features have higher importance scores."""
        mean, std = permutation_importance_numpy(self.model, self.X, self.y, 
                                                  n_repeats=20, random_state=42)
        
        # First two features should be more important (we constructed y from them)
        # Average of first two features should be greater than last three
        important_avg = np.mean(mean[:2])
        unimportant_avg = np.mean(mean[2:])
        
        self.assertGreater(important_avg, unimportant_avg,
                          "Important features should have higher importance scores")
    
    def test_std_is_non_negative(self):
        """Test that standard deviation values are non-negative."""
        mean, std = permutation_importance_numpy(self.model, self.X, self.y, 
                                                  n_repeats=10, random_state=42)
        
        self.assertTrue(np.all(std >= 0), 
                       "Standard deviation should be non-negative")
    
    def test_random_state_reproducibility(self):
        """Test that same random_state gives same results."""
        mean1, std1 = permutation_importance_numpy(self.model, self.X, self.y, 
                                                     n_repeats=10, random_state=42)
        mean2, std2 = permutation_importance_numpy(self.model, self.X, self.y, 
                                                     n_repeats=10, random_state=42)
        
        np.testing.assert_array_almost_equal(mean1, mean2, decimal=10,
                                             err_msg="Same random_state should give same mean")
        np.testing.assert_array_almost_equal(std1, std2, decimal=10,
                                             err_msg="Same random_state should give same std")
    
    def test_works_with_different_model(self):
        """Test that function works with different sklearn models."""
        # Try with a different model
        tree_model = DecisionTreeClassifier(random_state=42)
        tree_model.fit(self.X, self.y)
        
        mean, std = permutation_importance_numpy(tree_model, self.X, self.y, 
                                                  n_repeats=10, random_state=42)
        
        self.assertEqual(len(mean), self.X.shape[1])
        self.assertEqual(len(std), self.X.shape[1])
        self.assertTrue(np.all(std >= 0))


if __name__ == '__main__':
    unittest.main()
