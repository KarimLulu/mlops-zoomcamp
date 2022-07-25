import os

import pandas as pd


def run_test():
    predictions = pd.read_parquet(
        os.environ["OUTPUT_FILE_PATTERN"],
        storage_options={
            "client_kwargs": {"endpoint_url": os.environ["S3_ENDPOINT_URL"]}
        },
    )
    print(f"Sum of predictions: {predictions['predicted_duration'].sum():.2f}")
    assert round(predictions["predicted_duration"].sum(), 2) == 69.29


if __name__ == "__main__":
    run_test()
