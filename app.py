import gradio as gr
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ------------------ NLTK SETUP (HF SAFE) ------------------
def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

setup_nltk()

# ------------------ NLP LOGIC ------------------
def process_text(text):
    try:
        if not text or not text.strip():
            return [], [], [], []

        words = word_tokenize(text)
        sentences = sent_tokenize(text)

        porter = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        stemmed = [[w, porter.stem(w)] for w in words if w.isalpha()]
        lemmatized = [[w, lemmatizer.lemmatize(w)] for w in words if w.isalpha()]

        return (
            words,            # list[str]
            sentences,        # list[str]
            stemmed,          # list[list]
            lemmatized        # list[list]
        )

    except Exception as e:
        return (
            ["Error occurred"],
            [str(e)],
            [],
            []
        )

# ------------------ UI ------------------
with gr.Blocks(title="NLP Text Processing Playground") as demo:
    gr.Markdown("## 🧠 NLP Text Processing Playground")
    gr.Markdown(
        "Tokenization • Stemming • Lemmatization  \n"
        "**Author:** Karne Siddhartha"
    )

    text_input = gr.Textbox(
        lines=4,
        label="✍️ Enter text",
        value="I love NLP with Python. It is the future of Artificial Intelligence!"
    )

    run_btn = gr.Button("Run NLP Processing")

    word_out = gr.JSON(label="🔹 Word Tokens")
    sent_out = gr.JSON(label="🔹 Sentence Tokens")
    stem_out = gr.Dataframe(
        headers=["Original Word", "Stemmed Word"],
        label="🔹 Stemming",
        wrap=True
    )
    lemma_out = gr.Dataframe(
        headers=["Original Word", "Lemmatized Word"],
        label="🔹 Lemmatization",
        wrap=True
    )

    run_btn.click(
        process_text,
        inputs=text_input,
        outputs=[word_out, sent_out, stem_out, lemma_out]
    )

demo.launch()





