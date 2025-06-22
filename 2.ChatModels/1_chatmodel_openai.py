from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model='gpt-4', temperature=1, max_completion_tokens=10)
result = model.invoke("What is capital of India?")
print(result.content)