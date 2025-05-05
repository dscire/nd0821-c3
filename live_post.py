import requests

url = 'http://127.0.0.1:8000/inference'

data = {
    "age": 42,
    "workclass": "Private",
    "fnlgt": 159449,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 5178,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

print(f'Sending request to url: {url}')
print(f"POST data: {data}")

r = requests.post(url, json=data)

print(r.text)
