import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

# Function for text summarization using spaCy
def summarizer(rawdocs):
    stopwords = list(STOP_WORDS)

    # Load spaCy model for English
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdocs)

     # Extract individual tokens from the document
    tokens = [token.text for token in doc]

    # Calculate word frequencies
    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1

    # Normalize word frequencies
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq

    # Tokenize the document into sentences
    sent_tokens = [sent for sent in doc.sents]

    # Calculate sentence scores based on word frequencies
    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]

    # Select the top 30% of sentences based on scores
    select_len = int(len(sent_tokens) * 0.3)

    summary = nlargest(select_len, sent_scores, key=sent_scores.get)

    # Convert the summary sentences back to text
    final_summary = [word.text for word in summary]
    summary_text = ' '.join(final_summary)

    return summary_text, doc, len(rawdocs.split(' ')), len(summary_text.split(' '))


# Main Streamlit application
def main():
    
    st.title("Text Summarizer App")
    st.markdown("Welcome to Text Summarizer App! This application helps a traditional Natural Language Processing (NLP) \
                approach to provide a summaries of your text. Whether you have lengthy articles, \
                documents, or paragraphs, this tool can distill the essential information, making it easier \
                for you to grasp the main points.")

    # Text area for user input
    raw_text = st.text_area("Please enter some text paragraphs for summarization with minimum 1000 words", height=200, max_chars=None)

    # Button to trigger text summarization
    if st.button("Text Summarize"):
        if not raw_text:
            st.warning("Please enter some text for summarization.")
        else:
            # Call the summarizer function
            summary_text, _, original_length, summary_length = summarizer(raw_text)

            # Use columns to create two separate sections for original and summarized text
            col1, col2 = st.columns(2)

            # Original Text Section
            with col1:
                st.subheader("Original Text:")
                st.markdown(raw_text)
                

                st.info(f"Original Text Length: {original_length} words") 

            # Summarized Text Section
            with col2:
                st.subheader("Summarized Text:")
                st.markdown(summary_text)
                st.info(f"Summary Length: {summary_length} words")

# Entry point to run the Streamlit app
if __name__ == "__main__":
    main()


