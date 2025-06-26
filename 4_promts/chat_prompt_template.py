from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI()

prompt_template = ChatPromptTemplate([
    ("system", "you are expert in {domain}"),
    ("human", "Tell me about {topic}")
])

prompt = prompt_template.invoke({"domain": "cricket", "topic": "dusra"})
result = model.invoke(prompt)
print(result)