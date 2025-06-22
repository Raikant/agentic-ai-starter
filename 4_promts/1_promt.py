from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4")
st.header("Research tool")
user_input = st.text_input(label="User Input", placeholder="Enter your prompt here")

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.write(result.content)
