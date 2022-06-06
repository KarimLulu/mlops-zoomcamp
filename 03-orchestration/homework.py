import datetime
from pathlib import Path
import pickle

import pandas as pd
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


BASE_PATH = Path().parent.resolve()
DATA_PATH = BASE_PATH / 'data'
MODEL_PATH = BASE_PATH / 'models'
MODEL_PATH.mkdir(exist_ok=True)
BASE_FNAME = 'fhv_tripdata_{}-{:02d}.parquet'
DATE_FMT = "%Y-%m-%d"


@task
def read_data(path):
    return pd.read_parquet(path)


def get_path(date):
    return DATA_PATH / BASE_FNAME.format(date.year, date.month)


def get_date(date: str):
    if date is None:
        date = datetime.date.today()
    else:
        date = datetime.datetime.strptime(date, DATE_FMT).date()
    return date


@task
def get_paths(date):
    val_date = date.replace(day=1) - datetime.timedelta(days=1)
    train_date = val_date.replace(day=1) - datetime.timedelta(days=1)
    return get_path(train_date),  get_path(val_date)


@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values
    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")


def serialize_model(date, model, vectorizer):
    logger = get_run_logger()
    with (MODEL_PATH / f"model-{date}.bin").open("wb") as f:
        pickle.dump(model, f)
    vectorizer_path = MODEL_PATH / f"dv-{date}.pkl"
    with vectorizer_path.open("wb") as f:
        pickle.dump(vectorizer, f)
    size = vectorizer_path.stat().st_size
    logger.info(f"Vectorizer size {size:_} bytes")


@flow(task_runner=SequentialTaskRunner())
def main(date=None):
    date = get_date(date)
    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, train=False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    serialize_model(date.strftime(DATE_FMT), lr, dv)


DeploymentSpec(
    flow=main,
    name="homework_model_training",
    schedule=CronSchedule(cron='0 9 15 * *'),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml", "homework", "duration-prediction"]
)
