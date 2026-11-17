import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


def load_data(data_path='Churn_Modelling.csv'):
    """Charge les données depuis le fichier CSV."""
    # Chargement des données depuis le fichier CSV
    data = pd.read_csv(data_path)
    return data


def drop_useless_columns(data):
    """Supprime les colonnes inutiles pour la prédiction."""
    # Suppression des colonnes inutiles pour la prédiction (le nom et le pays)
    data = data.drop(['Surname', 'Geography'], axis=1)
    return data


def encode_gender(data):
    """Encode la colonne Gender en valeurs numériques."""
    # Importation d'un encodeur pour transformer les données catégoriques
    # en numériques
    encoder = LabelEncoder()
    # Encodage de la colonne "Gender" (par exemple, "Male" -> 1 et "Female" -> 0)
    data['Gender'] = encoder.fit_transform(data['Gender'])
    return data


def remove_missing_values(data):
    """Supprime les lignes contenant des données manquantes."""
    # Suppression des lignes avec des données manquantes
    data = data.dropna()
    return data


def split_features_target(data):
    """Sépare les caractéristiques (X) de la cible (y)."""
    # Séparation des données en caractéristiques (X) et cible (y)
    # Suppression des colonnes inutiles pour X, en gardant les colonnes utiles
    X = data.drop(['Exited', 'RowNumber', 'CustomerId'], axis=1)  # X : Caractéristiques
    y = data['Exited']  # y : Cible (le label à prédire)
    return X, y


def split_train_test(X, y):
    """Divise les données en jeu d'entraînement et jeu de test."""
    # Division des données en jeu d'entraînement (80 %) et jeu de test (20 %)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test


def scale_data(x_train, x_test):
    """Normalise les données et sauvegarde le scaler."""
    # Normalisation des données pour mettre les valeurs sur une même échelle
    scaler = StandardScaler()
    # Ajustement et transformation des données d'entraînement
    x_train_scaled = scaler.fit_transform(x_train)
    # Transformation des données de test en utilisant le même scaler
    x_test_scaled = scaler.transform(x_test)
    # Sauvegarde du scaler pour une utilisation ultérieure
    joblib.dump(scaler, 'scaler.joblib')
    return x_train_scaled, x_test_scaled


def prepare_data(data_path='Churn_Modelling.csv'):
    """Charge et prétraite les données en appelant les fonctions ci-dessus."""
    data = load_data(data_path)  # Charge les données
    data = drop_useless_columns(data)  # Supprime les colonnes inutiles
    data = encode_gender(data)  # Encode Gender
    data = remove_missing_values(data)  # Supprime les valeurs manquantes
    X, y = split_features_target(data)  # Sépare X et y
    x_train, x_test, y_train, y_test = split_train_test(X, y)  # Train/test
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)  # Normalise
    # Retourne les données préparées pour le modèle
    return x_train_scaled, x_test_scaled, y_train, y_test


def train_model(x_train, y_train):
    """Entraîne un modèle avec les données préparées."""
    # Crée une instance du classificateur RandomForest avec 100 arbres
    # (n_estimators) et une valeur aléatoire fixe pour assurer la
    # reproductibilité (random_state=1)
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    # Entraîne le modèle sur les données d'entraînement (x_train) et les
    # étiquettes (y_train)
    model.fit(x_train, y_train)
    # Retourne le modèle entraîné
    return model


def evaluate_model(model, x_test, y_test):
    """Évalue la performance du modèle sur les données de test."""
    # Utilise le modèle entraîné pour prédire les étiquettes des données de test
    y_pred = model.predict(x_test)
    # Calcule la précision du modèle en comparant les prédictions (y_pred)
    # avec les vraies étiquettes (y_test)
    accuracy = accuracy_score(y_test, y_pred)
    # Affiche la précision (accuracy) du modèle
    print(f"Accuracy: {accuracy}")
    # Retourne la précision calculée
    return accuracy


def predict(features):
    model = joblib.load('model.joblib')
    prediction = model.predict(features)
    return prediction


def save_model(model, model_path='model.joblib'):
    """Sauvegarde un modèle entraîné dans un fichier."""
    # Sauvegarde le modèle entraîné dans un fichier spécifié par model_path
    # (par défaut 'model.joblib')
    joblib.dump(model, model_path)
    # Affiche un message confirmant que le modèle a été sauvegardé
    print(f"Model saved to {model_path}")


def load_model(model_path='model.joblib'):
    """Charge le modèle depuis un fichier."""
    # Charge le modèle à partir du fichier spécifié par model_path
    # (par défaut 'model.joblib')
    model = joblib.load(model_path)
    # Affiche un message confirmant que le modèle a été chargé
    print(f"Model loaded from {model_path}")
    # Retourne le modèle chargé
    return model
