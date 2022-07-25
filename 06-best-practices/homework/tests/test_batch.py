from datetime import datetime

import pandas as pd

from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = ["PUlocationID", "DOlocationID", "pickup_datetime", "dropOff_datetime"]
    df = pd.DataFrame(data, columns=columns)
    expected = pd.DataFrame(
        {
            "PUlocationID": ["-1", "1"],
            "DOlocationID": ["-1", "1"],
            "pickup_datetime": df["pickup_datetime"].iloc[:2],
            "dropOff_datetime": df["dropOff_datetime"].iloc[:2],
            "duration": [8.0, 8.0],
        }
    )
    output = prepare_data(df)
    assert len(output) == 2
    pd.testing.assert_frame_equal(output, expected)
