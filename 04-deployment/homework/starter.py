#!/usr/bin/env python
# coding: utf-8

import argparse
from io import BytesIO
import logging
import pickle

import boto3
import pandas as pd


CATEGORICAL = ['PUlocationID', 'DOlocationID']
BUCKET_NAME = 'mlflow-artifacts-remote-3'


logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {filename} - {funcName} - {levelname} - {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{"
)
logger = logging.getLogger(__name__)
s3_client = boto3.client('s3')


def prepare_s3_key(year, month):
    key = f'fhv_tripdata_{year:04d}_{month:02d}_predictions.parquet'
    return key


def load_model(path: str = 'model.bin'):
    with open(path, 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr


def read_data(filename, categorical: list = None):
    if categorical is None:
        categorical = CATEGORICAL
    df = pd.read_parquet(filename)
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


def save_predictions(df: pd.DataFrame, output_key: str):
    out_buffer = BytesIO()
    df.to_parquet(out_buffer, index=False)
    s3_client.put_object(
        Body=out_buffer.getvalue(),
        Bucket=BUCKET_NAME,
        Key=output_key
    )


def apply_model(input_file, output_key, date_part_ride_id: str, categorical: list = None):
    if categorical is None:
        categorical = CATEGORICAL
    dv, lr = load_model()
    logger.info("Reading the data")
    df = read_data(input_file)
    dicts = df[categorical].to_dict(orient='records')
    logger.info("Building features")
    X_val = dv.transform(dicts)
    logger.info("Applying the model")
    y_pred = lr.predict(X_val)
    logger.info(f"Mean predicted duration {y_pred.mean():.3f}")
    df['ride_id'] = date_part_ride_id + df.index.astype('str')
    df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predicted_duration': y_pred})
    logger.info(f'Saving the result to S3 {repr(output_key)}')
    save_predictions(df_result, output_key)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int)
    parser.add_argument('--month', type=int)
    args = parser.parse_args()
    year, month = args.year, args.month
    input_file = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_key = prepare_s3_key(year, month)
    date_part_ride_id = f'{year:04d}/{month:02d}_'
    logger.info(input_file)
    apply_model(
        input_file=input_file,
        output_key=output_key,
        date_part_ride_id=date_part_ride_id
    )


if __name__ == '__main__':
    run()
