import streamlit as st
import time
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
from trulens_eval import Tru
from dotenv import load_dotenv
import google.generativeai as palm

# Trulens setup
tru = Tru()
load_dotenv()

# Configure Palm
palm.configure(api_key=st.secrets.get("GENERATIVE_AI_API_KEY", ""))

# Old English Teacher Chat

context = "You're a native Anglo-Saxon, once living in ancient England. you're teleported to the present era of modern United Kingdom. You're now teaching Old English to the present modern era English speaking."

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.text_area("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("teacher"):
        message_placeholder = st.empty()
        full_response = ""
        teacher_response = palm.chat(
            context=context,
            messages=prompt,
            temperature=0.25,
            candidate_count=1,
            top_k=40,
            top_p=0.95,
        ).last

        if teacher_response:
            # Simulate stream of response with milliseconds delay
            for chunk in teacher_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        else:
            st.warning("Sorry, I couldn't generate a response at the moment. Please try again.")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "teacher", "content": full_response})

# Continue with the rest of your Streamlit app code
st.write(tru.get_leaderboard(app_ids=["RAG v1"]))
tru.run_dashboard()

