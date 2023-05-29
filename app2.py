import streamlit as st
from transformers import DistilBartTokenizer, DistilBartForConditionalGeneration

def summarize_text(text):
    # Load the pre-trained DistilBART model
    model = DistilBartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
    tokenizer = DistilBartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

    # Tokenize the input text
    inputs = tokenizer([text], truncation=True, padding='longest', return_tensors='pt')

    # Generate the summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
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
        print("Summary:", summary)  # Print the summary in the console

if __name__ == "__main__":
    main()
