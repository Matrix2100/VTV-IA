import base
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)


@app.get('/home')
def homepage():
    return "IA module is running", 200


@app.get('/models')
def models():
    return jsonify(base.models()), 200


@app.post('/chat/completions')
def chat_completions():
    messages = request.get_json().get('messages')
    return jsonify(base.chat_completions(messages)), 200


if __name__ == '__main__':
    app.run(port=5000)
