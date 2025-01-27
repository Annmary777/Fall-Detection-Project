# stage_04_model_evaluation.py

import sys
import os
import logging
import json
from pathlib import Path
import subprocess  # Import subprocess to execute DVC commands
import datetime  # Import datetime to timestamp retraining triggers

# Ensure Python can find your src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ChestCancerClassifier.config.configuration import ConfigurationManager
from ChestCancerClassifier.components.model_evaluation_mlflow import Evaluation
from ChestCancerClassifier.utils.common import save_json
from ChestCancerClassifier import logger

# DagsHub + MLflow imports
import dagshub
import mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

STAGE_NAME = "Evaluation"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info("Loading evaluation configuration...")
        # 1) Load evaluation config
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()

        # 2) Instantiate the Evaluation component
        evaluation = Evaluation(eval_config)

        # 3) Initialize DagsHub MLflow integration
        dagshub.init(
            repo_owner="Annmary777",  # Replace with your DagsHub username
            repo_name="Chest-Cancer-Detection-using-MLOps1",  # Replace with your DagsHub repo name
            mlflow=True
        )

        logger.info("Starting MLflow run...")
        # 4) Start an MLflow run
        with mlflow.start_run(run_name=STAGE_NAME):
            logger.info("Logging evaluation parameters...")
            # Log evaluation parameters
            mlflow.log_param("image_size", eval_config.params_image_size)
            mlflow.log_param("batch_size", eval_config.params_batch_size)

            logger.info("Running evaluation...")
            # 5) Run the evaluation (returns a dict of metrics)
            results = evaluation.evaluation()

            logger.info(f"Evaluation results: {results}")

            logger.info("Logging metrics from evaluation results...")
            # 6) Log metrics from evaluation results
            if results and isinstance(results, dict):
                for metric_name, metric_value in results.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                        logger.info(f"Logged metric: {metric_name} = {metric_value}")
                    else:
                        mlflow.log_param(metric_name, str(metric_value))
                        logger.info(f"Logged param: {metric_name} = {metric_value}")
            else:
                logger.warning("No evaluation metrics to log.")

            logger.info("Logging metrics from scores.json...")
            # 7) Log metrics from scores.json
            scores_file = Path("scores.json")
            if scores_file.exists():
                try:
                    with open(scores_file, "r") as f:
                        scores = json.load(f)
                        for metric, value in scores.items():
                            mlflow.log_metric(metric, value)
                            logger.info(f"Logged metric from scores.json: {metric} = {value}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode scores.json: {e}")
            else:
                logger.warning("scores.json not found. Skipping logging from scores.json.")

            logger.info("Logging metrics from incorrect_predictions.json...")
            # 8) Log metrics from incorrect_predictions.json and trigger retraining if necessary
            incorrect_file = Path("incorrect_predictions.json")
            if incorrect_file.exists():
                try:
                    with open(incorrect_file, "r") as f:
                        incorrect_data = json.load(f)
                        num_incorrect = len(incorrect_data)
                        mlflow.log_metric("num_incorrect_predictions", num_incorrect)
                        logger.info(f"Logged number of incorrect predictions: {num_incorrect}")

                        # Optionally log details of incorrect predictions as params
                        for idx, pred in enumerate(incorrect_data):
                            mlflow.log_param(f"incorrect_prediction_{idx}", pred)
                            logger.debug(f"Logged incorrect prediction: {pred}")

                        # Trigger training if incorrect predictions exceed threshold
                        if num_incorrect > 5:
                            logger.info("Incorrect predictions exceed 5. Initiating retraining process...")
                            try:
                                # Update the retrain_trigger.txt file to trigger retraining
                                trigger_file = Path("retrain_trigger.txt")
                                with open(trigger_file, "a") as tf:
                                    tf.write(f"Retraining triggered at {datetime.datetime.now().isoformat()}\n")
                                logger.info(f"Updated {trigger_file} to trigger retraining.")

                                # Execute DVC repro training from the project root directory
                                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
                                result = subprocess.run(
                                    ["dvc", "repro", "training"],
                                    check=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    cwd=project_root  # Ensure the command runs from the project root
                                )

                                # Log the output and error (if any) from the DVC command
                                stdout_decoded = result.stdout.decode().strip()
                                stderr_decoded = result.stderr.decode().strip()
                                if stdout_decoded:
                                    logger.info(f"DVC repro output: {stdout_decoded}")
                                if stderr_decoded:
                                    logger.warning(f"DVC repro stderr: {stderr_decoded}")

                                logger.info("Retraining stage triggered successfully.")
                            except subprocess.CalledProcessError as e:
                                logger.error(f"Failed to trigger retraining: {e.stderr.decode().strip()}")
                                raise e
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode incorrect_predictions.json: {e}")
            else:
                logger.warning("incorrect_predictions.json not found. Skipping logging from incorrect_predictions.json.")

        logger.info("Evaluation run completed and logged successfully.")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

