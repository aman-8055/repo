import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

def summarize_text(text):
    # Load the pre-trained BART model
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    # Tokenize the input text
    input_ids = tokenizer.encode(text, truncation=True, return_tensors='pt')

    # Generate the summary
    summary_ids = model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    return summary

def main():
    st.title("Text Summarizer")
    
    # Get user input text
    text = st.text_area("Enter the text to summarize:")
    
    # Summarize the text when the user clicks the button
    if st.button("Summarize"):
        summary = summarize_text(text)
        st.subheader("Summary:")
        st.write(summary)

if __name__ == "__main__":
    main()

