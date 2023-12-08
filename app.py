import streamlit as st
import time
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
from trulens_eval import Feedback, LiteLLM, Tru, TruChain, Huggingface
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
        # Simulate stream of response with milliseconds delay
        for chunk in teacher_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "teacher", "content": full_response})

 # tru.run_dashboard()

    def ask_question(self, user_msg) -> str:
        rec = None
        with self.chain_recorder as recorder:
            resp = self.conversation({"question": user_msg})
            rec = recorder.get()

        pii_detected = False
        conciseness = 0.0
        if rec:
            for feedback_future in  as_completed(rec.feedback_results):
                feedback, feedback_result = feedback_future.result()
                
                print(f"feedback name: {feedback.name}\n result: {feedback_result.result}")

                if feedback.name == "pii_detection" and feedback_result.result != None:
                    pii_detected = True
                
                if feedback.name == "conciseness":
                    conciseness = float(feedback_result.result)
        
        if pii_detected:
            return "I'm sorry but personal information was detected in your question. Please remove any personal information."
        elif conciseness < 0.5:
            return "Please restate your question in a way the AI can understand and give a better answer"
        else:
            return resp["answer"]
# Display Trulens leaderboard
st.write(tru.get_leaderboard(app_ids=["RAG v1"]))

# Trulens dashboard
if st.button("Open TruLens Dashboard", key="trulens_dashboard"):
    tru.run_dashboard()

