import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, task, get_run_logger

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict,
)

# Uniquement nos fichiers : sinon le scan est très long
FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run_cmd(cmd):
    """Exécute une commande shell et fait échouer la task en cas d'erreur."""
    logger = get_run_logger()
    # Commandes fixes construites par ce fichier, jamais par l'utilisateur
    result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stdout + result.stderr)
        raise RuntimeError(f"Commande échouée : {' '.join(cmd)}")


# ---------- TASKS : environnement et code ----------
@task(name="install-dependencies")
def install_dependencies():
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="format-code")
def format_code():
    run_cmd([sys.executable, "-m", "black", *FILES])


@task(name="code-quality")
def code_quality():
    run_cmd([sys.executable, "-m", "flake8", "--max-line-length=120", *FILES])


@task(name="code-security")
def code_security():
    run_cmd([sys.executable, "-m", "bandit", "-q", *FILES])


@task(name="unit-tests")
def unit_tests():
    run_cmd([sys.executable, "-m", "pytest", "-v", "test_pipeline.py"])


# ---------- TASKS : données et modèle ----------
@task(name="prepare-data")
def prepare_data_task():
    return prepare_data()


@task(name="train-model")
def train_model_task(data):
    x_train, _, y_train, _ = data
    return train_model(x_train, y_train)


@task(name="save-model")
def save_model_task(model):
    save_model(model)


@task(name="load-model")
def load_model_task():
    return load_model()


@task(name="evaluate-model")
def evaluate_model_task(model, data):
    _, x_test, _, y_test = data
    return evaluate_model(model, x_test, y_test)


@task(name="predict")
def predict_task(data):
    _, x_test, _, _ = data
    return predict(x_test[:5])


# ---------- FLOWS ----------
@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    code_quality()
    code_security()
    unit_tests()


@flow(name="train")
def train_flow():
    data = prepare_data_task()
    model = train_model_task(data)
    save_model_task(model)
    return model


@flow(name="evaluate")
def evaluate_flow():
    data = prepare_data_task()
    model = load_model_task()
    return evaluate_model_task(model, data)


@flow(name="predict")
def predict_flow():
    data = prepare_data_task()
    return predict_task(data)


@flow(name="all")
def all_flow():
    code_flow()
    data = prepare_data_task()
    model = train_model_task(data)
    save_model_task(model)
    evaluate_model_task(model, data)


FLOWS = {
    "install": install_flow,
    "code": code_flow,
    "train": train_flow,
    "evaluate": evaluate_flow,
    "predict": predict_flow,
    "all": all_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
