import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load the processed features
try:
    processed_features = pd.read_csv('processed_features_with_embeddings.csv')
except FileNotFoundError:
    st.error("The processed features file is missing. Please ensure 'processed_features_with_embeddings.csv' is in the directory.")
    st.stop()

# Ensure correct column names and data types
processed_features.columns = processed_features.columns.str.strip()
if 'published_year' in processed_features.columns:
    processed_features['published_year'] = pd.to_numeric(processed_features['published_year'], errors='coerce')

# App Title
st.title("Book Recommendation System with Topics")

# Inputs for filtering
st.sidebar.title("Filters")

# Year range filter
min_year = st.sidebar.slider("Minimum Published Year", min_value=1853, max_value=2019, value=1853)
max_year = st.sidebar.slider("Maximum Published Year", min_value=1853, max_value=2019, value=2019)

# Page range filter
number_of_pages = st.sidebar.number_input("Enter Approximate Number of Pages:", min_value=1, value=300, step=1)
page_min = number_of_pages - 50
page_max = number_of_pages + 50

# Ratings count filter
min_ratings_count = st.sidebar.number_input("Minimum Ratings Count:", min_value=0, value=100, step=1)

# Categories filter
categories = st.sidebar.text_input("Categories (comma-separated):", value="")

# Topic Modeling Function
@st.cache_resource
def perform_topic_modeling(descriptions, n_topics=5):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    text_matrix = vectorizer.fit_transform(descriptions)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(text_matrix)
    topics = [
        [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        for topic in lda.components_
    ]
    topic_assignments = lda.transform(text_matrix).argmax(axis=1)
    return topic_assignments, topics

# Perform topic modeling
n_topics = st.sidebar.slider("Number of Topics", min_value=2, max_value=10, value=5, step=1)
processed_features['topic_assignment'], topics = perform_topic_modeling(processed_features['description'].fillna(''), n_topics=n_topics)

# Topic Selection
st.sidebar.title("Explore Topics")
selected_topic = st.sidebar.selectbox("Select a Topic", range(n_topics))
st.sidebar.markdown(f"**Keywords for Topic {selected_topic}:** {', '.join(topics[selected_topic])}")

# Filter books based on topic
filtered_by_topic = processed_features[processed_features['topic_assignment'] == selected_topic]

# Apply filters
filtered_books = filtered_by_topic[
    (filtered_by_topic['published_year'] >= min_year) &
    (filtered_by_topic['published_year'] <= max_year) &
    (filtered_by_topic['num_pages'] >= page_min) &
    (filtered_by_topic['num_pages'] <= page_max) &
    (filtered_by_topic['ratings_count'] >= min_ratings_count)
]

# Apply category filtering
if categories.strip():
    if 'categories' in processed_features.columns:
        category_list = [cat.strip().lower() for cat in categories.split(',')]
        filtered_books = filtered_books[
            filtered_books['categories'].str.lower().apply(
                lambda x: any(cat in x for cat in category_list) if pd.notnull(x) else False
            )
        ]

# Display Recommended Books
st.subheader(f"Books in Topic {selected_topic}: {', '.join(topics[selected_topic])}")

if not filtered_books.empty:
    # Display the books in a table format
    st.write(
        filtered_books[['title', 'average_rating', 'ratings_count', 'num_pages', 'published_year']]
        .sort_values(by='average_rating', ascending=False)
        .reset_index(drop=True)
    )
else:
    st.warning("No books found matching your criteria.")
