import streamlit as st
import time
from trulens_eval import Tru
from trulens_eval import Feedback, OpenAI, Tru, TruLlama, Select, OpenAI as fOpenAI
from trulens_eval.feedback import GroundTruthAgreement, Groundedness
from dotenv import load_dotenv
import google.generativeai as palm

tru = Tru()
# tru.reset_database() # if needed

st.header('Old English Teacher Chat')
palm.configure(api_key= st.secrets["GENERATIVE_AI_API_KEY"])
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
# Trulens setup
tru = Tru()
class OpenAI_custom(OpenAI):
    def query_translation_score(self, question1: str, question2: str) -> float:
        return float(chat_completion(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": "Your job is to rate how similar two quesitons are on a scale of 1 to 10. Respond with the number only."},
                    {"role": "user", "content": f"QUESTION 1: {question1}; QUESTION 2: {question2}"}
                ]
        ).choices[0].message.content) / 10

    def ratings_usage(self, last_context: str) -> float:
        return float(chat_completion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Your job is to respond with a '1' if the following statement mentions ratings or reviews, and a '0' if not."},
                {"role": "user", "content": f"STATEMENT: {last_context}"}
            ]
        ).choices[0].message.content)

# unstable: perhaps reduce temperature?

custom = OpenAI_custom()
# Input to tool based on trimmed user input.
f_query_translation = Feedback(
    custom.query_translation_score,
    name="Query Translation") \
.on_input() \
.on(Select.Record.app.query[0].args.str_or_query_bundle)

f_ratings_usage = Feedback(
    custom.ratings_usage,
    name="Ratings Usage") \
.on(Select.Record.app.query[0].rets.response)

# Result of this prompt: Given the context information and not prior knowledge, answer the query.
# Query: address of Gumbo Social
# Answer: "
fopenai = fOpenAI()
# Question/statement (context) relevance between question and last context chunk (i.e. summary)
f_context_relevance = Feedback(
    fopenai.qs_relevance,
    name="Context Relevance") \
.on_input() \
.on(Select.Record.app.query[0].rets.response)

# Groundedness
grounded = Groundedness(groundedness_provider=fopenai)

f_groundedness = Feedback(
    grounded.groundedness_measure,
    name="Groundedness") \
.on(Select.Record.app.query[0].rets.response) \
.on_output().aggregate(grounded.grounded_statements_aggregator)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(
    fopenai.relevance,
    name="Answer Relevance"
).on_input_output()

golden_set = [
    {"query": "Hello there mister AI. What's the vibe like at oprhan andy's in SF?", "response": "welcoming and friendly"},
    {"query": "Is park tavern in San Fran open yet?", "response": "Yes"},
    {"query": "I'm in san francisco for the morning, does Juniper serve pastries?", "response": "Yes"},
    {"query": "What's the address of Gumbo Social in San Francisco?", "response": "5176 3rd St, San Francisco, CA 94124"},
    {"query": "What are the reviews like of Gola in SF?", "response": "Excellent, 4.6/5"},
    {"query": "Where's the best pizza in New York City", "response": "Joe's Pizza"},
    {"query": "What's the best diner in Toronto?", "response": "The George Street Diner"}
]

f_groundtruth = Feedback(
    GroundTruthAgreement(golden_set).agreement_measure,
    name="Ground Truth Eval") \
.on_input_output()

# Continue with the rest of your Streamlit app code
st.write(tru.get_leaderboard(app_ids=["RAG v1"]))
tru.run_dashboard(
#     _dev=trulens_path, force=True  # if running from github
)
