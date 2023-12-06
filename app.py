import google.generativeai as palm
import time
import streamlit as st
from trulens_eval import Tru, TruCustomApp, Feedback, Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
import numpy as np

# Sidebar for OpenAI API key
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password", key="openai_api_key")
st.secrets["OPENAI_API_KEY"] = openai_api_key

# Sidebar for GENERATIVE_AI_API_KEY
generative_ai_api_key = st.sidebar.text_input("Enter Generative AI API Key:", type="password", key="generative_ai_api_key")
st.secrets["GENERATIVE_AI_API_KEY"] = generative_ai_api_key

# Trulens setup
tru = Tru()

# Existing code for Old English Teacher Chat
st.header('Old English Teacher Chat')
palm.configure(api_key=st.secrets["AIzaSyAANEPA1UF6WE4O_0GQh2s27iBT4VrN0Ag"])
if "model" not in st.session_state:
    st.session_state["model"] = "models/chat-bison-001"

context = "You're a native Anglo-Saxon, once living in ancient England. you're teleported to the present era of modern United Kingdom. You're now teaching Old English to the present modern era English speaking."

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
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
class RAG_from_scratch:
    # ... (your existing RAG_from_scratch class)
    fopenai = fOpenAI()
grounded = Groundedness(groundedness_provider=fopenai)
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

f_qa_relevance = (
    Feedback(fopenai.relevance_with_cot_reasons, name="Answer Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on_output()
)

f_context_relevance = (
    Feedback(fopenai.qs_relevance_with_cot_reasons, name="Context Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on(Select.RecordCalls.retrieve.rets.collect())
    .aggregate(np.mean)
)

rag = RAG_from_scratch()
tru_rag = TruCustomApp(rag,
                      app_id='RAG v1',
                      feedbacks=[f_groundedness, f_qa_relevance, f_context_relevance])

with tru_rag as recording:
    rag.query("When was the University of Washington founded?")

st.write(tru.get_leaderboard(app_ids=["RAG v1"]))
tru.run_dashboard()
