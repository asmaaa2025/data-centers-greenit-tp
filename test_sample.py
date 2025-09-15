import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def test_data_loading():
    """Test que le dataset se charge correctement"""
    df = pd.read_csv('dataset_tp/Book1 (1).csv')
    assert df.shape[0] > 0, "Le dataset ne contient aucune ligne"
    assert 'country' in df.columns, "La colonne 'country' est manquante"
    assert 'total_data_centers' in df.columns, "La colonne 'total_data_centers' est manquante"
    print("✅ Test de chargement des données réussi")

def test_data_cleaning():
    """Test que le nettoyage des données fonctionne"""
    df = pd.read_csv('dataset_tp/Book1 (1).csv')

    # Fonction de nettoyage
    def clean_percentage(value):
        if pd.isna(value):
            return None
        value_str = str(value)
        cleaned = value_str.replace('%', '').replace('~', '').replace('+', '').strip()
        try:
            return float(cleaned)
        except:
            return None

    # Test sur quelques valeurs
    test_values = ['50%', '~30%', '40+', '25.5']
    expected_results = [50.0, 30.0, 40.0, 25.5]

    for val, expected in zip(test_values, expected_results):
        result = clean_percentage(val)
        assert result == expected, f"Erreur: {val} -> {result}, attendu {expected}"

    print("✅ Test de nettoyage des données réussi")

def test_regression_model():
    """Test qu'un modèle de régression peut être entraîné"""
    # Données de test
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegression().fit(X, y)
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)

    assert r2 > 0.9, f"R² trop faible: {r2}"
    print("✅ Test de modèle de régression réussi")

if __name__ == "__main__":
    test_data_loading()
    test_data_cleaning()
    test_regression_model()
    print("🎉 Tous les tests sont passés avec succès!")
