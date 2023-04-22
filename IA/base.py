import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")
openai.organization = os.getenv("OPENAI_ORG")
headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.getenv("OPENAI_KEY")}
model = 'gpt-3.5-turbo'

messages_memory = []


def models():
    return openai.Model.list()


def chat_completions(messages):
    messages_memory.append(messages)
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=7,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response
