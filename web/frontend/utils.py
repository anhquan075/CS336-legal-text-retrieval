from typing import List, Dict, Any, Tuple

import os
import json
import logging
import requests
from requests.structures import CaseInsensitiveDict
from time import sleep
from uuid import uuid4
import streamlit as st


API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
API_QUERY_ENDPOINT = "http://192.168.20.151:5002/api/legal_text_retrieval/"
STATUS = "initialized"
HS_VERSION = "hs_version"
DOC_REQUEST = "query"
DOC_FEEDBACK = "feedback"
DOC_UPLOAD = "file-upload"


def service_is_ready():
    """
    Used to show the "Haystack is loading..." message
    """
    url = API_QUERY_ENDPOINT
    try:
        if requests.get(url).status_code < 400:
            return True
    except Exception as e:
        logging.exception(e)
        sleep(2)  # To avoid spamming a non-existing endpoint at startup
    return False


@st.cache
def haystack_version():
    """
    Get the Haystack version from the REST API
    """
    url = f"{API_ENDPOINT}/{HS_VERSION}"
    return requests.get(url, timeout=0.1).json()["hs_version"]


def query(query) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """

    url = API_QUERY_ENDPOINT

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/x-www-form-urlencoded; charset=utf-8"

    data = f"query={query}"
    
    response_raw = requests.post(url, headers=headers, data=data.encode('utf-8'))

    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()

    # Format response
    answers = [json.loads(response["result"])]
    
    return answers, response


def send_feedback(query, answer_obj, is_correct_answer, is_correct_document, document) -> None:
    """
    Send a feedback (label) to the REST API
    """
    url = f"{API_ENDPOINT}/{DOC_FEEDBACK}"
    req = {
        "id": str(uuid4()),
        "query": query,
        "document": document,
        "is_correct_answer": is_correct_answer,
        "is_correct_document": is_correct_document,
        "origin": "user-feedback",
        "answer": answer_obj
        }
    response_raw = requests.post(url, json=req)
    if response_raw.status_code >= 400:
        raise ValueError(f"An error was returned [code {response_raw.status_code}]: {response_raw.json()}")


def upload_doc(file):
    url = f"{API_ENDPOINT}/{DOC_UPLOAD}"
    files = [("files", file)]
    response = requests.post(url, files=files).json()
    return response


def get_backlink(result) -> Tuple[str, str]:
    if result.get("document", None):
        doc = result["document"]
        if isinstance(doc, dict):
            if doc.get("meta", None):
                if isinstance(doc["meta"], dict):
                    if doc["meta"].get("url", None) and doc["meta"].get("title", None):
                        return doc["meta"]["url"], doc["meta"]["title"]
    return None, None


def find_corpus(corpus_data, law_id, article_id):
    for corpus in corpus_data:
        if corpus['law_id'] == law_id:
            return corpus['law_id'], corpus['articles'][int(article_id) - 1]
    
    return None, None
