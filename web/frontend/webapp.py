import os
import sys
import json
import random

import logging
import pandas as pd
from json import JSONDecodeError
from pathlib import Path
import streamlit as st
#from annotated_text import annotation
from markdown import markdown
from annotated_text import annotation

# streamlit does not support any states out of the box. On every button click, streamlit reload the whole page
# and every value gets lost. To keep track of our feedback state we use the official streamlit gist mentioned
# here https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
from utils import service_is_ready, query, send_feedback, haystack_version, get_backlink, find_corpus


# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP", "Chồng ép buộc vợ sinh con gái thì bị phạt bao nhiêu tiền?")
DEFAULT_ANSWER_AT_STARTUP = os.getenv("DEFAULT_ANSWER_AT_STARTUP", "Paris")

# Labels for the evaluation
EVAL_LABELS = os.getenv("EVAL_FILE", Path(__file__).parent / "private_test_question.json")

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD"))

LEGAL_CORPUS_FILE = "legal_corpus.json"
LEGAL_CORPUS_DATA = json.load(open(LEGAL_CORPUS_FILE, 'r', encoding='utf-8'))

# Load csv into pandas dataframe
with open(EVAL_LABELS, 'r', encoding='utf-8') as fo:
    questions_data_json = json.load(fo)
questions_data_json_lst = [item for item in questions_data_json['items']][:15]


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

def main():
    st.set_page_config(page_title='CS336 Demo', page_icon="https://aiclub.uit.edu.vn/wp-content/uploads/2021/08/cropped-logo-32x32.png")
    st.markdown(
            """
        <style>
        .reportview-container .main {
            background-color: linear-gradient(#ddd6f3, #faaca8);
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: white;
        }
        .Widget>label {
            color: lack;
            font-family: monospace;
        }
        [class^="st-b"]  {
            color: black;
            font-family: monospace;
        }
        .st-bb {
            background-color: transparent;
        }
        .st-at {
            background-color: #0c0080;
        }
        footer {
            font-family: monospace;
        }
        header .decoration {
            background-image: none;
        }

        </style>
        """,
            unsafe_allow_html=True,
        )
    # Persistent state

    set_state_if_absent('question', DEFAULT_QUESTION_AT_STARTUP)
    set_state_if_absent('answer', DEFAULT_ANSWER_AT_STARTUP)
    set_state_if_absent('results', None)
    set_state_if_absent('raw_json', None)
    set_state_if_absent('random_question_requested', False)

    # Small callback to reset the interface in case the text of the question changes

    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None

    # Title
    st.write("# CS336 Demo - Legal Text Retrieval")
    st.markdown("""
    This demo takes its data from <a href="https://challenge.zalo.ai/portal/legal-text-retrieval">Zalo AI Challenge 2021</a> in NLP task in December 2021 on the topic of

    <h3 style='text-align:center;padding: 0 0 1rem;'>Vietnamese Legal Text Retrieval</h3>

    Ask any question on this topic and see if the model can find the correct answer to your query!

    *Note: do not use keywords, but full-fledged questions.* The demo is not optimized to deal with keyword queries and might misunderstand you.
""", unsafe_allow_html=True)

    hs_version = ""
    try:
        hs_version = f" <small>(v{haystack_version()})</small>"
    except Exception:
        pass

    # Search bar
    question = st.text_input("",
                                value=st.session_state.question,
                                max_chars=150,
                                on_change=reset_results
                                )
    col1, col2 = st.columns(2)
    col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
    col2.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)

    # Run button
    run_pressed = col1.button("Run")

    # Get next random question from the json
    if col2.button("Random question"):
        reset_results()
        new_row = random.choice(questions_data_json_lst)
        # Avoid picking the same question twice (the change is not visible on the UI)
        while new_row['question'] == st.session_state.question:
            new_row = random.choice(questions_data_json_lst)
        st.session_state.question = new_row['question']
        st.session_state.random_question_requested = True
        # Re-runs the script setting the random question as the textbox value
        # Unfortunately necessary as the Random Question button is _below_ the textbox
        raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))
    else:
        st.session_state.random_question_requested = False
    
    run_query = (run_pressed or question != st.session_state.question) and not st.session_state.random_question_requested
    
    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; Service is starting..."):
        if not service_is_ready():
            st.error("🚫 &nbsp;&nbsp; Connection Error. Is the API running?")
            run_query = False
            reset_results()

    # Get results for query
    if run_query and question:
        reset_results()
        st.session_state.question = question
        with st.spinner(
            "🧠 &nbsp;&nbsp; Performing neural search on documents... \n "
        ):
            try:
                st.session_state.results, st.session_state.raw_json = query(question)
                
            except JSONDecodeError as je:
                st.error("👓 &nbsp;&nbsp; An error occurred reading the results. Is the collection working?")
                return
            except Exception as e:
                logging.exception(e)
                if "The server is busy processing requests" in str(e) or "503" in str(e):
                    st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
                else:
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
                return

    if st.session_state.results:
        # Show the gold answer if we use a question of the given set
        st.write("## Result:")
        for result in st.session_state.results:
            if result:
                article_title, article_text = result["title"], result["text"]
                
                start_idx, end_idx = result["start_id"], result["end_id"]
                
                if "<br/>" in article_text[start_idx:end_idx]:
                    answer = article_text[start_idx:end_idx - 7]
                else:
                    answer = article_text[start_idx:end_idx]
                
                
                st.write(markdown(f"<p><h5>{article_title}</h5></p>" +
                            "<p>{}</p>".format(article_text[:start_idx] +
                                                str(annotation(answer, "ANSWER", "#8ef")) +
                                                article_text[end_idx:])), unsafe_allow_html=True)
            #st.markdown(f"**Relevance:** {result['relevance']} -  **Source:** {source}")

        # else:
        #     st.info("🤔 &nbsp;&nbsp; The model is unsure whether any of the documents contain an answer to your question. Try to reformulate it!")

        # if eval_mode and result["answer"]:
        #     # Define columns for buttons
        #     is_correct_answer = None
        #     is_correct_document = None

        #     button_col1, button_col2, button_col3, _ = st.columns([1, 1, 1, 6])
        #     if button_col1.button("👍", key=f"{result['context']}{count}1", help="Correct answer"):
        #         is_correct_answer=True
        #         is_correct_document=True

        #     if button_col2.button("👎", key=f"{result['context']}{count}2", help="Wrong answer and wrong passage"):
        #         is_correct_answer=False
        #         is_correct_document=False

        #     if button_col3.button("👎👍", key=f"{result['context']}{count}3", help="Wrong answer, but correct passage"):
        #         is_correct_answer=False
        #         is_correct_document=True

        #     if is_correct_answer is not None and is_correct_document is not None:
        #         try:
        #             send_feedback(
        #                 query=question,
        #                 answer_obj=result["_raw"],
        #                 is_correct_answer=is_correct_answer,
        #                 is_correct_document=is_correct_document,
        #                 document=result["document"]
        #             )
        #             st.success("✨ &nbsp;&nbsp; Thanks for your feedback! &nbsp;&nbsp; ✨")
        #         except Exception as e:
        #             logging.exception(e)
        #             st.error("🐞 &nbsp;&nbsp; An error occurred while submitting your feedback!")

        # st.write("___")
        
    st.markdown(f"""
    <style>
        a {{
            text-decoration: none;
        }}
        .haystack-footer {{
            text-align: center;
        }}
        .haystack-footer h4 {{
            margin: 0.1rem;
            padding:0;
        }}
        footer {{
            opacity: 0;
        }}
    </style>
    <div class="haystack-footer">
        <hr />
        <h4>Built with <a href="https://streamlit.io">streamlit</a>{hs_version}</h4>
        <p>Get it on <a href="https://github.com/deepset-ai/haystack/">GitHub</a> &nbsp;&nbsp; - &nbsp;&nbsp; Read the <a href="https://haystack.deepset.ai/overview/intro">Docs</a></p>
    </div>
    """, unsafe_allow_html=True)

main()
