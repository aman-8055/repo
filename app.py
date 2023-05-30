import streamlit as st
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def download_nltk_resources():
    nltk.download("punkt")

def summarize_text(text, num_sentences):
    # Create a parser and tokenizer
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    # Create an LsaSummarizer
    summarizer = LsaSummarizer()
    
    # Set the number of sentences for the summary
    summarizer = LsaSummarizer()
    summarizer.stop_words = ["."]

    # Summarize the text
    summary = summarizer(parser.document, num_sentences)
    
    # Join the summary sentences into a single string
    summary_text = " ".join(str(sentence) for sentence in summary)
    
    return summary_text

def main():
    st.title("Text Summarizer")
    
    # Download the necessary NLTK resources
    download_nltk_resources()
    
    # Get user input text
    text = st.text_area("Enter the text to summarize:")
    
    # Get the number of sentences for the summary
    num_sentences = st.slider("Select the number of sentences for the summary:", 1, 10, 3)
    
    # Summarize the text when the user clicks the button
    if st.button("Summarize"):
        summary = summarize_text(text, num_sentences)
        st.subheader("Summary:")
        st.write(summary)

if __name__ == "__main__":
    main()
