import streamlit as st
import openai
import pandas as pd
import gensim
import numpy as np
from BookRecommendationSystem import (
    recommend_books_tfidf,
    recommend_books_bert,
    recommend_books_roberta,
    recommend_books_distilbert,
    hybrid_recommendation,
)

# OpenAI API Key
openai.api_key = "" #Input API Key

# Load data
@st.cache_resource
def load_data():
    return pd.read_csv("data.csv")

# Load the dataset
data = load_data()

# Streamlit UI
st.title("Book Recommendation Chatbot")

# Main Chat Interface
st.header("Chat with the Book Recommendation Bot")
user_query = st.text_input("Ask for book recommendations!", "")

if user_query:
    # Generate recommendations
    recommendations = {
        "TF-IDF": recommend_books_tfidf(user_query),
        "BERT": recommend_books_bert(user_query),
        "RoBERTa": recommend_books_roberta(user_query),
        "DistilBERT": recommend_books_distilbert(user_query),
        "Hybrid": hybrid_recommendation(user_query, "some_user_topic"),
    }

    # OpenAI Chat Response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a book recommendation assistant."},
            {"role": "user", "content": f"User asked: {user_query}"},
            {"role": "assistant", "content": f"Here are recommendations: {recommendations}"}
        ],
        max_tokens=200
    )
    # Extract the response text
    chatbot_response = response['choices'][0]['message']['content']

    # Display chatbot response
    st.subheader("Chatbot Response")
    st.write(chatbot_response)

    # Display recommendations
    #st.subheader("Recommendations")
    #for model, rec in recommendations.items():
    #    st.write(f"**{model} Recommendations:**")
    #    st.dataframe(rec)