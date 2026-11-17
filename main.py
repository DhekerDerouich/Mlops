import argparse
from model_pipeline import (prepare_data, train_model, evaluate_model,
                            save_model, load_model, predict)


def main():
    # Crée un objet ArgumentParser pour gérer les arguments en ligne de commande
    parser = argparse.ArgumentParser()
    # Ajoute l'argument --prepare qui, s'il est spécifié, indique de préparer
    # les données
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    # Ajoute l'argument --train qui, s'il est spécifié, indique d'entraîner
    # le modèle
    parser.add_argument("--train", action="store_true", help="Train the model")
    # Ajoute l'argument --evaluate qui, s'il est spécifié, indique d'évaluer
    # le modèle
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    # Ajoute l'argument --predict (nécessaire car le bloc args.predict plus bas)
    parser.add_argument("--predict", action="store_true", help="Predict with the model")
    # Ajoute l'argument --save qui, s'il est spécifié
    parser.add_argument("--save", action="store_true", help="save the model")
    # Ajoute l'argument --load qui, s'il est spécifié
    parser.add_argument("--load", action="store_true", help="load the model")
    # Ajoute l'argument --all qui, s'il est spécifié, exécute tout le pipeline
    parser.add_argument("--all", action="store_true", help="Run the whole pipeline")
    # Analyse les arguments passés en ligne de commande
    args = parser.parse_args()

    # Si l'argument --all est spécifié, active toutes les étapes du pipeline
    # (les blocs ci-dessous s'exécutent ensuite dans l'ordre)
    if args.all:
            args.prepare = args.train = args.evaluate = args.predict = True
    # Si l'argument --prepare est spécifié, prépare les données
    if args.prepare:
        x_train, x_test, y_train, y_test = prepare_data()
        print("Data prepared.")
    # Si l'argument --train est spécifié, prépare les données et entraîne
    # le modèle
    if args.train:
        x_train, x_test, y_train, y_test = prepare_data()  # Prépare les données
        model = train_model(x_train, y_train)  # Entraîne le modèle
        save_model(model)  # Sauvegarde le modèle entraîné
        print("Model trained and saved.")
    # Si l'argument --evaluate est spécifié, prépare les données et évalue
    # le modèle
    if args.evaluate:
        x_train, x_test, y_train, y_test = prepare_data()  # Prépare les données
        model = load_model()  # Charge le modèle sauvegardé
        evaluate_model(model, x_test, y_test)  # Évalue le modèle
    if args.predict:
        x_train, x_test, y_train, y_test = prepare_data()  # Prépare les données
        model = load_model()  # Charge le modèle sauvegardé
        predictions = predict(x_test)  # Prédit sur les données de test
        print("Predictions (10 premières):", predictions[:10])
    if args.save:
        x_train, x_test, y_train, y_test = prepare_data()  # Prépare les données
        model = train_model(x_train, y_train)  # Entraîne le modèle
        save_model(model)  # Sauvegarde le modèle entraîné
        print("Model trained and saved.")
    if args.load:
        x_train, x_test, y_train, y_test = prepare_data()  # Prépare les données
        model = load_model()  # Charge le modèle sauvegardé


# Exécute la fonction main si ce fichier est exécuté en tant que script
if _name_ == "_main_":
    main()
