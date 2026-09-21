"""Streamlit interface for an LSTM/RNN next-word prediction model."""

from pathlib import Path
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MAX_LEN_PATH = BASE_DIR / "max_len.pkl"
MODEL_CANDIDATES = (BASE_DIR / "lstm_model.h5", BASE_DIR / "LSTM_model.h5", BASE_DIR / "RNN_model.h5", BASE_DIR / "model.h5")


@st.cache_resource(show_spinner="Loading prediction model...")
def load_artifacts():
    """Load and cache the tokenizer, maximum sequence length, and trained model."""
    model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        expected = ", ".join(path.name for path in MODEL_CANDIDATES)
        raise FileNotFoundError(f"Model file not found. Expected one of: {expected}")

    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)
    with MAX_LEN_PATH.open("rb") as file:
        max_len = pickle.load(file)

    return tokenizer, int(max_len), load_model(model_path), model_path.name


def predict_next_word(text, tokenizer, max_len, model):
    """Return the most likely next word for the text entered by the user."""
    sequence = tokenizer.texts_to_sequences([text])[0]
    if not sequence:
        return None

    padded_sequence = pad_sequences([sequence], maxlen=max_len - 1, padding="pre")
    probabilities = model.predict(padded_sequence, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    return tokenizer.index_word.get(predicted_index)


st.set_page_config(page_title="Next Word Predictor", page_icon="✍️", layout="centered")
st.title("✍️ Next Word Predictor")
st.caption("Enter a phrase and let the trained LSTM/RNN model predict its next word.")

try:
    tokenizer, max_len, model, model_name = load_artifacts()
except (FileNotFoundError, OSError, pickle.UnpicklingError, ValueError) as error:
    st.error(f"Could not load the saved model artifacts: {error}")
    st.stop()

with st.form("prediction_form"):
    prompt = st.text_input(
        "Start typing",
        placeholder="For example: The weather is",
        help="Use words that appeared in the data used to train the model.",
    )
    submitted = st.form_submit_button("Predict next word", use_container_width=True)

if submitted:
    if not prompt.strip():
        st.warning("Please enter a few words first.")
    else:
        next_word = predict_next_word(prompt, tokenizer, max_len, model)
        if next_word:
            st.success(f"Predicted next word: **{next_word}**")
            st.info(f"Complete suggestion: {prompt.strip()} **{next_word}**")
        else:
            st.warning("I couldn't recognize those words. Try a phrase closer to the training data.")

with st.expander("Model details"):
    st.write(f"**Model file:** `{model_name}`")
    st.write(f"**Maximum sequence length:** {max_len}")
    st.write(f"**Vocabulary size:** {len(tokenizer.word_index):,} words")
