#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pickle

import pandas as pd

CATEGORICAL = ["PUlocationID", "DOlocationID"]


def get_input_path(year: int, month: int):
    default_input_pattern = (
        "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/"
        f"fhv_tripdata_{year:04d}-{month:02d}.parquet"
    )
    input_pattern = os.getenv("INPUT_FILE_PATTERN", default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year: int, month: int):
    s3_bucket_name = os.getenv("S3_BUCKET_NAME", "mlflow-artifacts-remote-3")
    default_output_pattern = (
        f"s3://{s3_bucket_name}/taxi_type=fhv/"
        f"year={year:04d}/month={month:02d}/predictions.parquet"
    )
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
    return output_pattern.format(year=year, month=month)


def load_model():
    with open("model.bin", "rb") as f_in:
        vectorizer, model = pickle.load(f_in)
    return vectorizer, model


def read_data(filename: str):
    options = None
    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
    if S3_ENDPOINT_URL:
        options = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}
    df = pd.read_parquet(filename, storage_options=options)
    return df


def save_data(df: pd.DataFrame, filename: str):
    options = None
    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
    if S3_ENDPOINT_URL:
        options = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}
    df.to_parquet(filename, engine="pyarrow", index=False, storage_options=options)


def prepare_data(df: pd.DataFrame, categorical: list = None):
    if categorical is None:
        categorical = CATEGORICAL
    df["duration"] = (df.dropOff_datetime - df.pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    return df


def main(year: int, month: int):
    vectorizer, model = load_model()
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    df = read_data(input_file)
    df = prepare_data(df, categorical=CATEGORICAL)
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    dicts = df[CATEGORICAL].to_dict(orient="records")
    X_val = vectorizer.transform(dicts)
    y_pred = model.predict(X_val)

    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred

    save_data(df_result, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    args = parser.parse_args()
    main(year=args.year, month=args.month)
