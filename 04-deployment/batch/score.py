#!/usr/bin/env python
# coding: utf-8

import argparse
from io import BytesIO
import logging
import uuid

import boto3
import mlflow
import pandas as pd


BUCKET_NAME = 'mlflow-artifacts-remote-3'
MODEL_PATH = 's3://mlflow-artifacts-remote-3/2/{run_id}/artifacts/model'


logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {filename} - {funcName} - {levelname} - {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{"
)
logger = logging.getLogger(__name__)
s3_client = boto3.client('s3')


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df['ride_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


def load_model(run_id):
    model = mlflow.pyfunc.load_model(MODEL_PATH.format(run_id=run_id))
    return model


def prepare_s3_key(taxi_type, year, month):
    key = f"output/{taxi_type}/{year:04d}-{month:02d}.parquet"
    return key


def save_predictions(df: pd.DataFrame, output_key: str):
    out_buffer = BytesIO()
    df.to_parquet(out_buffer, index=False)
    s3_client.put_object(
        Body=out_buffer.getvalue(),
        Bucket=BUCKET_NAME,
        Key=output_key
    )


def apply_model(input_file, run_id, output_key):
    logger.info(f'Reading the data from {input_file}')
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    logger.info(f'Loading the model with RUN_ID={run_id}')
    model = load_model(run_id)

    logger.info(f'Applying the model')
    y_pred = model.predict(dicts)

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id

    logger.info(f'Saving the result to S3 {repr(output_key)}')
    save_predictions(df_result, output_key)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxi_type')
    parser.add_argument('--year', type=int)
    parser.add_argument('--month', type=int)
    parser.add_argument('--run_id')
    args = parser.parse_args()
    taxi_type, year, month = args.taxi_type, args.year, args.month

    input_file = f'https://s3.amazonaws.com/nyc-tlc/trip+data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_key = prepare_s3_key(taxi_type, year, month)

    apply_model(
        input_file=input_file,
        run_id=args.run_id,
        output_key=output_key
    )


if __name__ == '__main__':
    run()
