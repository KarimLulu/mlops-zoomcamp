import argparse
import logging
import mlflow
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {filename} - {funcName} - {levelname} - {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{"
)
logger = logging.getLogger(__name__)


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def fetch_logged_data(run_id):
    """Fetch params, metrics, tags, and artifacts in the specified run"""
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    return data.params, data.metrics


def run(data_path):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_valid)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
    run_id = mlflow.last_active_run().info.run_id
    params, metrics = fetch_logged_data(run_id)
    logger.info(f"Number of parameters: {len(params)}")
    for k, v in params.items():
        print(f"{k} - {v}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    args = parser.parse_args()

    run(args.data_path)
