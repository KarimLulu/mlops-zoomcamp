import os

import mlflow
from flask import Flask, request, jsonify


RUN_ID = os.getenv('RUN_ID')

logged_model = f's3://mlflow-artifacts-remote-3/2/{RUN_ID}/artifacts/model'
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    features = {
        'PU_DO': f"{ride['PULocationID']}_{ride['DOLocationID']}",
        'trip_distance': ride['trip_distance']
    }
    return features


def predict(features):
    preds = model.predict(features)
    return preds[-1]


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
