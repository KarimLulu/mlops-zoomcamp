import os

import pandas as pd


def run_test():
    predictions = pd.read_parquet(
        os.environ["OUTPUT_FILE_PATTERN"],
        storage_options={
            "client_kwargs": {"endpoint_url": os.environ["S3_ENDPOINT_URL"]}
        },
    )
    assert round(predictions["predicted_duration"].sum(), 2) == 69.29
