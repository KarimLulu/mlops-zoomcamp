#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import pickle


import pandas as pd


CATEGORICAL = ['PUlocationID', 'DOlocationID']


logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {filename} - {funcName} - {levelname} - {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{"
)
logger = logging.getLogger(__name__)


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


def apply_model(input_file, output_file, date_part_ride_id: str, categorical: list = None):
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
    logger.info("Saving results")
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int)
    parser.add_argument('--month', type=int)
    args = parser.parse_args()
    year, month = args.year, args.month
    input_file = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'fhv_tripdata_{year:04d}_{month:02d}_predictions.parquet'
    date_part_ride_id = f'{year:04d}/{month:02d}_'
    logger.info(input_file)
    apply_model(
        input_file=input_file,
        output_file=output_file,
        date_part_ride_id=date_part_ride_id
    )


if __name__ == '__main__':
    run()
