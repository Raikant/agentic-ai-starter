from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()
model = ChatOpenAI()
# chat template
prompt_template = ChatPromptTemplate([
    ("system", "you are a helpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# load chat history
chat_history = []
with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())

# create prompt
prompt = prompt_template.invoke({"chat_history": chat_history, "query": "where is my refund?"})
result = model.invoke(prompt)
print(result.content)
