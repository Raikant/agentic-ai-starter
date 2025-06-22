from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

RESEARCH_PAPERS = [
    "GPT-4 Technical Report",
    "Constitutional AI: A Research Agenda",
    "Language Models are Few-Shot Learners",
    "Training Language Models to Follow Instructions",
    "PaLM: Scaling Language Modeling with Pathways"
]

model = ChatOpenAI(model='gpt-4')
st.header("Research tool")

selected_paper = st.selectbox("Select a research paper:", RESEARCH_PAPERS)
user_input = st.text_input("Enter your prompt about the selected paper")

if st.button("Summarize"):
    combined_prompt = f"Regarding the paper '{selected_paper}': {user_input}"
    result = model.invoke(combined_prompt)
    st.write(result.content)
