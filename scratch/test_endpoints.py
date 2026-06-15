import requests
import json

base_url = "http://localhost:8000"

print("1. Testing Mixed CS generate_quiz...")
try:
    payload = {
        "topic": "Mixed CS",
        "count": 10,
        "difficulty": "mixed"
    }
    r = requests.post(f"{base_url}/api/generate_quiz", json=payload)
    print("Mixed CS Status:", r.status_code)
    print("Mixed CS Response:", r.text[:500])
except Exception as e:
    print("Error:", e)
