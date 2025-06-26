from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print("AI:", result.content)

print(chat_history)
