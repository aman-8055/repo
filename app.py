import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def summarize_text(text):
    # Load the Pegasus model and tokenizer
    model_name = "google/pegasus-xsum"
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    
    # Tokenize the input text
    inputs = tokenizer.encode(text, truncation=True, max_length=512, return_tensors="pt")
    
    # Generate the summary
    summary_ids = model.generate(inputs, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    
    # Decode the summary tokens into text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Streamlit app
def main():
    st.title("Paragraph Summarizer")
    
    # Text input
    text = st.text_area("Enter the paragraph you want to summarize", height=200)
    
    # Summarize button
    if st.button("Summarize"):
        if text:
            # Call the summarize_text function
            summary = summarize_text(text)
            st.success(summary)
        else:
            st.warning("Please enter a paragraph to summarize.")

if __name__ == "__main__":
    main()
