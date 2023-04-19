import openai
import time
from dotenv import load_dotenv
import requests
import os
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_KEY")
openai.organization = os.getenv("OPENAI_ORG")
headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.getenv("OPENAI_KEY")}
model = 'gpt-3.5-turbo'


@app.route('/')
def homepage():
    return "IA module is running"


@app.route('/chat', methods=['POST'])
def chat_completions():
    data = request.get_json()
    text = data.get('text')
    response = openai.Completion.create(
        engine=model,
        prompt=text,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = response.choices[0].text
    return jsonify(message)


if __name__ == '__main__':
    app.run()

# GET MODEL LIST
# response = requests.get('https://api.openai.com/v1/models', headers=headers)
# print(response.status_code)
# print(response.text)

# GET MODEL
# response = requests.get(f'https://api.openai.com/v1/models/{model}', headers=headers)
# print(response.status_code)
# print(response.text)

# POST COMPLETIONS
# payload = {'model': f'{model}', 'messages': [{'role': 'user', 'content': 'Como posso acender a luz?'}], 'temperature': 0.7}
# response = requests.post('https://api.openai.com/v1/chat/completions', json=payload, headers=headers)
# print(response.status_code)
# print(response.text)


#
# # Define o modelo GPT-3 que você deseja usar
# model_engine = os.getenv("OPENAI_MODEL")


# Define a função de geração de texto
# def generate_text(prompt2):
#     response = openai.Completion.create(
#         engine=model_engine,
#         prompt=prompt2,
#         max_tokens=1024,
#         n=1,
#         stop=None,
#         temperature=0.5,
#     )
#     time.sleep(1)  # espera 1 segundo para evitar limites da API
#     message = response.choices[0].text
#     return message.strip()
#
#
# # Loop principal do chatbot
# while True:
#     # Obtém a entrada do usuário
#     user_input = input("Você: ")
#
#     # Gera uma resposta do modelo GPT-3
#     prompt = "Conversa: {}\nVocê:".format(user_input)
#     response = generate_text(prompt)
#
#     # Imprime a resposta do chatbot
#     print("Chatbot:", response)
# print("Chatbot:")
# print(os.getenv("OPENAI_KEY"))
