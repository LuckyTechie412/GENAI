import streamlit as st
from transformers import pipeline

# Load the sentiment analysis pipeline
classifier = pipeline("text-generation")

# Streamlit app
def main():
    st.title("Text Geberation App")

    # User input
    user_input = st.text_input("Enter a sentence:")

    if user_input:
        # Perform sentiment analysis
        result = classifier(user_input)
        st.write(result)
        # Display results
        #st.write("Sentiment:", result[0]['label'])
        #st.write("Confidence:", result[0]['score'])

if __name__ == "__main__":
    main()
