import requests
from dotenv import load_dotenv
import os

load_dotenv()


def send_message(message):
    params = {'messages': {'role': 'user', 'content': message}}
    headers = {}
    response_data = requests.post(os.getenv("IA-URL")+"/chat/completions", params=params, headers=headers)
    if response_data.status_code == 200:
        return response_data.json()
    else:
        return response_data.status_code
