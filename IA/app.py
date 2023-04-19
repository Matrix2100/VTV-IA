import openai
import time
from dotenv import load_dotenv
import requests
import os
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")
openai.organization = os.getenv("OPENAI_ORG")
headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.getenv("OPENAI_KEY")}
model = 'gpt-3.5-turbo'

messages = []


@app.get('/home')
def homepage():
    return "IA module is running", 200


@app.get('/models')
def models():
    return jsonify(openai.Model.list()), 200


@app.post('/chat/completions')
def chat_completions():
    data = request.get_json()
    messages.append(data.get('message'))
    response = openai.ChatCompletion.create(
        model=model,
        messages=data.get('messages'),
        max_tokens=7,
        n=1,
        stop=None,
        temperature=0.5,
    )
    # message = response.choices[0].text
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(port=5000)
