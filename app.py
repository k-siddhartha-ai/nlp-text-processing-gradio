import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
# Page Configuration
st.set_page_config(
    page_title="NLP Text Processing Playground",
    page_icon="🧠",
    layout="centered"
)

# NLTK Setup (Hugging Face safe)
@st.cache_resource
def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

setup_nltk()

# Title & Description
st.title("🧠 NLP Text Processing Playground")
st.subheader("Tokenization • Stemming • Lemmatization")

st.markdown("""
This interactive app demonstrates **core NLP concepts**
used in **Machine Learning and Artificial Intelligence**.
""")

st.markdown("**Author:** Karne Siddhartha")
st.markdown("---")

# User Input
text = st.text_area(
    "✍️ Enter text:",
    "I love NLP with Python. It is the future of Artificial Intelligence!",
    height=120
)

# Word Tokenization
st.header("🔹 Word Tokenization")
words = word_tokenize(text)
st.code(words)

# Sentence Tokenization

st.header("🔹 Sentence Tokenization")
sentences = sent_tokenize(text)

for i, s in enumerate(sentences, start=1):
    st.write(f"{i}. {s}")

st.write(f"**Total sentences:** {len(sentences)}")

# Stemming
st.header("🔹 Stemming (Porter Stemmer)")
porter = PorterStemmer()

stemmed = [(w, porter.stem(w)) for w in words if w.isalpha()]

st.table({
    "Original Word": [w for w, _ in stemmed],
    "Stemmed Word": [s for _, s in stemmed]
})

# Lemmatization
st.header("🔹 Lemmatization (WordNet)")
lemmatizer = WordNetLemmatizer()

lemmatized = [(w, lemmatizer.lemmatize(w)) for w in words if w.isalpha()]

st.table({
    "Original Word": [w for w, _ in lemmatized],
    "Lemmatized Word": [l for _, l in lemmatized]
})

# Footer
st.markdown("---")
st.markdown("Built with **Streamlit** and **NLTK**")



