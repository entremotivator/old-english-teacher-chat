import streamlit as st
import time
import google.generativeai as palm
import trulens_eval
from trulens_eval import Tru, TruCustomApp, Feedback, Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
import numpy as np
from dotenv import load_dotenv

# Trulens setup
tru = Tru()
load_dotenv()

# Configure Palm
palm.configure(api_key=st.secrets.get("GENERATIVE_AI_API_KEY"))

# Existing code for Old English Teacher Chat

class RAG_from_scratch:
    def __init__(self):
        self.fopenai = fOpenAI()  # Instantiate fopenai here
        self.grounded = Groundedness(groundedness_provider=self.fopenai)
        self.f_groundedness = (
            Feedback(self.grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
            .on(Select.RecordCalls.retrieve.rets.collect())
            .on_output()
            .aggregate(self.grounded.grounded_statements_aggregator)
        )

        self.f_qa_relevance = (
            Feedback(self.fopenai.relevance_with_cot_reasons, name="Answer Relevance")
            .on(Select.RecordCalls.retrieve.args.query)
            .on_output()
        )

        self.f_context_relevance = (
            Feedback(self.fopenai.qs_relevance_with_cot_reasons, name="Context Relevance")
            .on(Select.RecordCalls.retrieve.args.query)
            .on(Select.RecordCalls.retrieve.rets.collect())
            .aggregate(np.mean)
        )

rag_instance = RAG_from_scratch()

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
        # Simulate stream of response with milliseconds delay
        for chunk in teacher_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "teacher", "content": full_response})

# Trulens integration
query = st.text_input("Ask Trulens a question:")
if query:
    rag_instance.grounded.query(query)

# Display Trulens leaderboard
st.write(tru.get_leaderboard(app_ids=["RAG v1"]))

# Trulens dashboard
if st.button("Open TruLens Dashboard", key="trulens_dashboard"):
    tru.run_dashboard()
