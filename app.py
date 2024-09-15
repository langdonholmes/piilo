"""Streamlit app for Student Name Detection models."""

import json
import os
import warnings
from json import JSONEncoder

import pandas as pd
import streamlit as st
from annotated_text import annotated_text

from piilo.engines.analyzer import CustomAnalyzer
from piilo.engines.anonymizer import SurrogateAnonymizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Student Name Detector (English)", layout="wide")

# Pickled vectorizer freaks out if this isn't here
def identity(x):
    return x

# Helper methods
@st.cache_resource()
def analyzer_engine():
    """Return AnalyzerEngine and cache with Streamlit."""

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }

    return CustomAnalyzer(configuration=configuration)


@st.cache_resource()
def anonymizer_engine():
    """Return generate surrogate anonymizer."""
    return SurrogateAnonymizer()


def annotate(text, st_analyze_results, st_entities):
    tokens = []
    # sort by start index
    results = sorted(st_analyze_results, key=lambda x: x.start)
    for i, res in enumerate(results):
        if i == 0:
            tokens.append(text[: res.start])

        # append entity text and entity type
        tokens.append((text[res.start : res.end], res.entity_type))

        # if another entity coming i.e. we're not at the last results element, add text up to next entity
        if i != len(results) - 1:
            tokens.append(text[res.end : results[i + 1].start])
        # if no more entities coming, add all remaining text
        else:
            tokens.append(text[res.end :])
    return tokens


# Side bar
st.sidebar.image("logo.png")

st.sidebar.markdown(
    """Detect and anonymize PII in text using an [NLP model](https://huggingface.co/langdonholmes/en_student_name_detector) [trained](https://github.com/aialoe/deidentification-pipeline) on student-generated text collected from a massive online open-enrollment course.
"""
)

st_entities = st.sidebar.multiselect(
    label="Which entities to look for?",
    options=analyzer_engine().get_supported_entities(),
    default=list(analyzer_engine().get_supported_entities()),
)

st_threshold = st.sidebar.slider(
    label="Acceptance threshold", min_value=0.0, max_value=1.0, value=0.35
)

st_return_decision_process = st.sidebar.checkbox("Add analysis explanations in json")

st.sidebar.info(
    "This is part of a project to develop new anonymization systems that are appropriate for student-generated text."
)

# Main panel
analyzer_load_state = st.info(
    "Starting Presidio analyzer and loading Longformer-based model..."
)
engine = analyzer_engine()
analyzer_load_state.empty()


st_text = st.text_area(
    label="Type in some text",
    value="""   John Smith, living at 123 Elm Street, Springfield, IL 62704, was born on July 5, 1985. His Social Security Number is 123-45-6789. You can contact him via email at john.smith@example.com or by phone at (312) 555-1234.
    His credit card number is 4111 1111 1111 1111, which expires on 08/25, and the CVV is 123. He also has a driver's license number D1234567 issued by the state of Illinois.
    John's wife, Mary Ann Smith, was born on February 14, 1988. Her email address is mary.ann.smith@samplemail.com, and her phone number is (217) 555-9876. Mary's SSN is 987-65-4321. They have two children, Emma Smith (born April 2, 2010) and Michael Smith (born September 12, 2013).
    John's employer, Acme Corp, is located at 456 Maple Ave, Chicago, IL 60601. You can reach the company at (312) 555-7890, or by email at hr@acmecorp.com. His employee ID is JSMITH1234.
    John's bank account number with Big Bank is 000123456789, and the routing number is 026009593.""",
    height=200,
)

button = st.button("Detect PII")

if "first_load" not in st.session_state:
    st.session_state["first_load"] = True

# After
st.subheader("Analyzed")
with st.spinner("Analyzing..."):
    if button or st.session_state.first_load:
        st_analyze_results = engine.analyze(
            text=st_text,
            entities=st_entities,
            language="en",
            score_threshold=st_threshold,
            return_decision_process=st_return_decision_process,
        )
        st_analyze_results = engine.prune_results(st_analyze_results)
        annotated_tokens = annotate(st_text, st_analyze_results, st_entities)
        # annotated_tokens
        annotated_text(*annotated_tokens)

# vertical space
st.text("")

st.subheader("Anonymized")
with st.spinner("Anonymizing..."):
    if button or st.session_state.first_load:
        st_anonymize_results = anonymizer_engine().anonymize(
            st_text, st_analyze_results
        )
        st_anonymize_results.text

# table result
st.subheader("Detailed Findings")
if st_analyze_results:
    res_dicts = [r.to_dict() for r in st_analyze_results]
    for d in res_dicts:
        d["Value"] = st_text[d["start"] : d["end"]]
    df = pd.DataFrame.from_records(res_dicts)
    df = df[["entity_type", "Value", "score", "start", "end"]].rename(
        {
            "entity_type": "Entity type",
            "start": "Start",
            "end": "End",
            "score": "Confidence",
        },
        axis=1,
    )

    st.dataframe(df, width=1000)
else:
    st.text("No findings")

st.session_state["first_load"] = True


# json result
class ToDictListEncoder(JSONEncoder):
    """Encode dict to json."""

    def default(self, o):
        """Encode to JSON using to_dict."""
        if o:
            return o.to_dict()
        return []


if st_return_decision_process:
    st.json(json.dumps(st_analyze_results, cls=ToDictListEncoder))
