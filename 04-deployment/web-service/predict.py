import pickle

from flask import Flask, request, jsonify


FLASK_PORT = 9696
MODEL_PATH = 'lin_reg.bin'


with open(MODEL_PATH, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


def prepare_features(ride):
    features = {
        'PU_DO': f"{ride['PULocationID']}_{ride['DOLocationID']}",
        'trip_distance': ride['trip_distance']
    }
    return features


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds[-1]


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    predicted_duration = predict(features)

    result = {
        'duration': predicted_duration
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=FLASK_PORT)
