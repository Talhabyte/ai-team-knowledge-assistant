import os
import json
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(
    page_title="AI Team Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)


def load_knowledge_base():
    with open("knowledge_base.txt", "r", encoding="utf-8") as file:
        return file.read()


def ask_ai(context: str, question: str) -> str:
    prompt = f"""
You are an AI Team Knowledge Assistant.

Use ONLY the context below to answer the user's question.
If the answer is not in the context, say that the knowledge base does not contain enough information.

Context:
{context}

Question:
{question}

Answer clearly and practically.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful internal team knowledge assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def generate_summary(context: str) -> str:
    prompt = f"""
Summarize the following team knowledge into:
1. Key discussion points
2. Decisions made
3. Action items
4. Risks
5. Future improvements

Context:
{context}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You summarize internal team knowledge clearly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def extract_action_items(context: str) -> str:
    prompt = f"""
Extract only the action items from this team knowledge.
Return owner, task, and priority if possible.

Context:
{context}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You extract action items from meeting notes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


st.title("🧠 AI Team Knowledge Assistant")
st.write("Ask questions, summarize team knowledge, and extract action items from internal notes.")

context = load_knowledge_base()

with st.expander("View Knowledge Base"):
    st.text_area("Knowledge Base Content", context, height=300)

st.subheader("Ask a Question")

question = st.text_input(
    "Question",
    placeholder="Example: What are the main delivery risks?"
)

if st.button("Ask Assistant"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = ask_ai(context, question)
                st.success("Answer generated.")
                st.write(answer)
            except Exception as e:
                st.error(f"AI error: {e}")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Generate Knowledge Summary")

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                summary = generate_summary(context)
                st.write(summary)

                report = {
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "knowledge_summary",
                    "content": summary
                }

                st.download_button(
                    label="Download Summary",
                    data=json.dumps(report, indent=4),
                    file_name="knowledge_summary.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"AI error: {e}")

with col2:
    st.subheader("Extract Action Items")

    if st.button("Extract Actions"):
        with st.spinner("Extracting action items..."):
            try:
                actions = extract_action_items(context)
                st.write(actions)

                report = {
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "action_items",
                    "content": actions
                }

                st.download_button(
                    label="Download Action Items",
                    data=json.dumps(report, indent=4),
                    file_name="action_items.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"AI error: {e}")