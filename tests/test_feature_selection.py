import unittest
import numpy as np
from functions import select_k_best_important_features

class TestFeatureSelection(unittest.TestCase):

    def test_select_k_best_important_features(self):
        # Fake dataset : 4 features
        X = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ])

        feature_names = ["A", "B", "C", "D"]

        # Importances volontairement ordonnées
        importances = np.array([0.1, 0.5, 0.2, 0.8]) 
        # -> D (0.8), B (0.5), C (0.2), A (0.1)

        # Select top-2
        X_reduced, selected = select_k_best_important_features(X, feature_names, importances, k=2)

        # Vérifications
        expected_features = ["D", "B"]  # dans cet ordre
        self.assertEqual(selected, expected_features)

        # Vérifie les dimensions (3 échantillons, 2 features)
        self.assertEqual(X_reduced.shape, (3, 2))

        # Vérifie que les colonnes sélectionnées correspondent bien aux bonnes colonnes
        np.testing.assert_array_equal(X_reduced, X[:, [3, 1]])

if __name__ == "__main__":
    unittest.main()



