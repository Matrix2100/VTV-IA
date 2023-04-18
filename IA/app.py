import openai
# import time
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # Configura a chave da API do OpenAI
# openai.api_key = os.getenv("OPENAI_KEY")
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
print("Chatbot:")
# print(os.getenv("OPENAI_KEY"))
