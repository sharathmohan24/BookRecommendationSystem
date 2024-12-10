**Book Recommendation System with Chatbot**

A sophisticated book recommendation system that leverages Natural Language Processing (NLP) models to deliver personalized book suggestions through an interactive chatbot interface.

**Features**
Advanced NLP Models:

TF-IDF: Keyword-based matching.
BERT, RoBERTa, DistilBERT: Contextual recommendations using transformer embeddings.
Word2Vec, FastText, Doc2Vec: Semantic word and document-based similarity.
Graph-Based: Entity-relationship exploration for novel recommendations.
Topic Modeling: Thematic-based suggestions using Latent Dirichlet Allocation (LDA).
Hybrid Recommendations: Combines multiple models for comprehensive suggestions.
Interactive Chatbot:

Integrated with OpenAI's GPT model for conversational user interaction.
Provides book recommendations in response to user queries.
APIs:

Flask APIs for seamless integration and testing with tools like Postman.
Deployment:

User-friendly Streamlit interface for direct querying and recommendations.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/book-recommendation-system.git
cd book-recommendation-system
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set up API Keys:

Replace placeholders with your OpenAI API key in streamlit_chatbot.py.
Run the Flask API:

bash
Copy code
python flask_api.py
Launch Streamlit:

bash
Copy code
streamlit run streamlit_chatbot.py
Dataset
The system uses a dataset of books with the following key columns:

title, authors, categories, average_rating, description, published_year, num_pages.
Preprocessing Steps:

Handled missing data.
Generated embeddings using TF-IDF, BERT, and more.
Created graph-based relationships and topic distributions.
How It Works
Query the Chatbot:

Input your preferences or ask for recommendations.
Example: "Suggest books on history with high ratings."
Filters (Optional):

In Streamlit, filter results by year, rating, or page count.
Outputs:

The chatbot suggests books using multiple models.
Example Postman response (for history):
TF-IDF Recommendations
BERT Recommendations
... and others.
Hybrid Recommendations:

Aggregates results from all models for a balanced output.
Models Used
TF-IDF-Based Content Filtering:

Matches user queries with book descriptions using term frequency-inverse document frequency.
BERT, RoBERTa, DistilBERT:

Contextual embeddings for deeper semantic understanding.
Word2Vec, FastText, Doc2Vec:

Semantic similarity via dense vector representations.
Graph-Based:

Explores entity relationships like authors and categories.
Topic Modeling:

Suggests books aligned with thematic topics extracted using LDA.
Hybrid:

Integrates multiple models for comprehensive recommendations.
Example Usage
Flask API
Input: POST to /recommend with query history.
Output: Recommendations from all models, e.g.:
json
Copy code
{
  "tfidf_recommendations": [...],
  "bert_recommendations": [...],
  ...
}
Streamlit
Input your query and get recommendations with chatbot interaction.
Example: "Books on artificial intelligence."
Contributing
Feel free to open issues or submit pull requests. Contributions are welcome!

License
This project is licensed under the MIT License. See the LICENSE file for details.
