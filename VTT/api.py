from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


app = Flask(__name__)

@app.get('/models')
def models():
    return jsonify(openai.Model.list()), 200


@app.post('/chat/completions')

if __name__ == '__main__':
    app.run(port=5002)