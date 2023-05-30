import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the pre-trained BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

@st.cache
def summarize_text(text):
    # Tokenize the input text
    inputs = tokenizer([text], truncation=True, max_length=1024, return_tensors='pt')

    # Generate the summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Print the summary
    print("Summary:", summary)

    return summary

def main():
    st.title("Text Summarizer")
    
    # Get user input text
    with st.form("text_form"):
        text = st.text_area("Enter the text to summarize:")
        submit_button = st.form_submit_button(label="Summarize")
    
    # Summarize the text when the form is submitted
    if submit_button:
        summary = summarize_text(text)
        st.subheader("Summary:")
        st.write(summary)

if __name__ == "__main__":
    main()
