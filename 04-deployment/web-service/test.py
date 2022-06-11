import requests

from predict import FLASK_PORT


ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

url = f'http://localhost:{FLASK_PORT}/predict'
response = requests.post(url, json=ride)
print(response.json())
