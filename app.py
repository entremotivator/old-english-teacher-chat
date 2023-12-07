import google.generativeai as palm
import time
import streamlit as st

st.header('Old English Teacher Chat')
palm.configure(api_key=st.secrets["GENERATIVE_AI_API_KEY"])

if "model" not in st.session_state:
    st.session_state["model"] = "models/chat-bison-001"

context = "You're a native Anglo-Saxon, once living in ancient England. you're teleported to the present era of modern United Kingdom. You're now teaching Old English to the present modern era English speaking."

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.text(f'{message["role"]}: {message["content"]}')

# Accept user input
prompt = st.text_input("What is up?")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response
    st.text("Teacher: (thinking...)")

    teacher_response = palm.chat(
        context=context,
        messages=prompt,
        temperature=0.25,
        candidate_count=1,
        top_k=40,
        top_p=0.95,
    ).last

    # Simulate response delay
    time.sleep(1)

    # Display assistant response in chat history
    st.session_state.messages.append({"role": "teacher", "content": teacher_response})
    st.text(f'Teacher: {teacher_response}')
