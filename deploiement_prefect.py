from prefect import serve
from prefect.schedules import Cron

from pipeline_prefect import (
    all_flow,
    train_flow,
    evaluate_flow,
    predict_flow,
    install_flow,
    code_flow,
)

TZ = "Africa/Tunis"

if __name__ == "__main__":
    serve(
        # Planifiés
        all_flow.to_deployment(
            name="ml-pipeline-all",
            schedule=Cron("0 2 * * *", timezone=TZ),
            tags=["full-pipeline", "mlops"],
        ),
        train_flow.to_deployment(
            name="ml-pipeline-train",
            schedule=Cron("0 3 * * 1", timezone=TZ),
            tags=["training", "mlops"],
        ),
        predict_flow.to_deployment(
            name="ml-pipeline-predict",
            schedule=Cron("0 * * * *", timezone=TZ),
            tags=["prediction", "mlops"],
        ),
        # Manuels uniquement (pas de schedule)
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]
        ),
        install_flow.to_deployment(
            name="ml-pipeline-install", tags=["setup", "mlops"]
        ),
        code_flow.to_deployment(
            name="ml-pipeline-code", tags=["code-quality", "mlops"]
        ),
    )
