import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict,
)


def main():
    # Crée un objet ArgumentParser pour gérer les arguments en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--predict", action="store_true", help="Predict with the model")
    parser.add_argument("--save", action="store_true", help="Save the model")
    parser.add_argument("--load", action="store_true", help="Load the model")
    parser.add_argument("--all", action="store_true", help="Run the whole pipeline")
    # Analyse les arguments passés en ligne de commande
    args = parser.parse_args()

    # --all active toutes les étapes du pipeline
    if args.all:
        args.prepare = args.train = args.evaluate = args.predict = True

    # --prepare : prépare les données
    if args.prepare:
        prepare_data()
        print("Data prepared.")

    # --train : prépare les données, entraîne et sauvegarde le modèle
    if args.train:
        x_train, _, y_train, _ = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)
        print("Model trained and saved.")

    # --evaluate : charge le modèle et l'évalue
    if args.evaluate:
        _, x_test, _, y_test = prepare_data()
        model = load_model()
        evaluate_model(model, x_test, y_test)

    # --predict : prédit sur les données de test (predict() charge le modèle)
    if args.predict:
        _, x_test, _, _ = prepare_data()
        predictions = predict(x_test)
        print("Predictions (10 premières):", predictions[:10])

    # --save : entraîne puis sauvegarde le modèle
    if args.save:
        x_train, _, y_train, _ = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)
        print("Model trained and saved.")

    # --load : charge le modèle sauvegardé
    if args.load:
        load_model()


# Exécute la fonction main si ce fichier est exécuté en tant que script
if __name__ == "__main__":
    main()
