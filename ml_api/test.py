import requests
import json

# Patient avec historique de DÉTÉRIORATION
critical_patient = {
    "hr": 140,
    "spo2": 88,
    "bp": 160,
    "temp": 39.5,
    "rr": 28,
    "age": 78,
    "history": [
        {"hr": 100, "spo2": 95, "bp": 130, "temp": 37.5, "rr": 18},
        {"hr": 115, "spo2": 92, "bp": 140, "temp": 38.0, "rr": 22},
        {"hr": 125, "spo2": 90, "bp": 150, "temp": 38.8, "rr": 25},
        {"hr": 140, "spo2": 88, "bp": 160, "temp": 39.5, "rr": 28}
    ]
}

response = requests.post("http://localhost:5000/predict", json=critical_patient)
print(json.dumps(response.json(), indent=2))