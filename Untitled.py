from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def summarize_text(text, num_sentences=3):
    # Initialize the parser with the text
    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    # Initialize the LSA Summarizer
    summarizer = LsaSummarizer()

    # Summarize the text
    summary = summarizer(parser.document, num_sentences)

    # Create the summary text
    summary_text = ""
    for sentence in summary:
        summary_text += str(sentence) + " "

    return summary_text

# Example usage
text = "Your text goes here. This is a sample text that you want to summarize. You can modify this text with your own content."

summary = summarize_text(text)
print(summary)
