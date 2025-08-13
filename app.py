import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# -----------------------------
# Load and preprocess dataset
# -----------------------------
nltk.download('punkt')
nltk.download('punkt_tab')

data = pd.read_csv("chatbot_dataset.csv")

data['Question'] = data['Question'].apply(
    lambda x: ' '.join(nltk.word_tokenize(str(x).lower()))
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    data['Question'], data['Answer'], test_size=0.2, random_state=42
)

# Create model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)


# -----------------------------
# Get bot response
# -----------------------------
def get_response(question):
    question = ' '.join(nltk.word_tokenize(question.lower()))
    return model.predict([question])[0]


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Smart Chatbot", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
    .user-msg {
        background-color: #d4f8d4;
        color: #000;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .bot-msg {
        background-color: #f1f0f0;
        color: #000;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💬 AI Chatbot")
st.markdown("Ask me anything from my knowledge base!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Type your message:", placeholder="Ask something...")

if st.button("Send") and user_input.strip() != "":
    response = get_response(user_input)
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    st.session_state.chat_history.append({"role": "bot", "text": response})

for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"<div class='user-msg'>{chat['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{chat['text']}</div>", unsafe_allow_html=True)
