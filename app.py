import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import streamlit as st

# Download NLTK data files
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load Dataset
df = pd.read_csv("Recipe_Reviews.csv")

# Preprocess the Data
# Handle Missing Values
df = df.dropna(subset=['text', 'stars'])
df['thumbs_up'] = df['thumbs_up'].fillna(0)
df['thumbs_down'] = df['thumbs_down'].fillna(0)

# Convert Unix Timestamp to Datetime
df['created_at'] = pd.to_datetime(df['created_at'], unit='s')

# Clean Text Data
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
# Sidebar Filters
st.sidebar.header("Filters")
recipe_options = df['recipe_name'].unique()
selected_recipe = st.sidebar.selectbox("Select a Recipe", recipe_options)

# Filter Data Based on Selected Recipe
filtered_df = df[df['recipe_name'] == selected_recipe]

# Display Recipe Information
st.title(f"Recipe: {selected_recipe}")
st.write(f"Number of Reviews: {len(filtered_df)}")
st.write(f"Average Rating: {filtered_df['stars'].mean():.2f}")

# Display Summarized Reviews Using LDA
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_features = tfidf_vectorizer.fit_transform(filtered_df['cleaned_text'])

lda_model = LDA(n_components=5, random_state=42)
lda_model.fit(tfidf_features)

st.subheader("Summarized Reviews")
for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10 - 1:-1]]
    st.write(f"Topic {topic_idx + 1}: {' '.join(top_words)}")

# Display Sentiment Distribution
sentiment_counts = filtered_df['stars'].value_counts().sort_index()

plt.figure(figsize=(8, 4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title("Sentiment Distribution")
plt.xlabel("Star Ratings")
plt.ylabel("Number of Reviews")

st.subheader("Sentiment Distribution")
st.pyplot(plt)

# Display Top Reviews
top_reviews = filtered_df.nlargest(5, 'thumbs_up')[['text', 'thumbs_up', 'thumbs_down']]
st.subheader("Top Reviews")
st.table(top_reviews)

# Display Word Cloud
from wordcloud import WordCloud

all_reviews = " ".join(filtered_df['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Reviews")

st.subheader("Word Cloud of Reviews")
st.pyplot(plt)

# Display Top 5 Most Popular Recipes
st.subheader("Top 5 Most Popular Recipes")
top_recipes = df.groupby('recipe_name')['stars'].mean().sort_values(ascending=False).head(5)
st.table(top_recipes.reset_index().rename(columns={'stars': 'Average Rating'}))

# Display Bottom 5 Least Popular Recipes
st.subheader("Bottom 5 Least Popular Recipes")
least_popular_recipes = df.groupby('recipe_name')['stars'].mean().sort_values(ascending=True).head(5)
st.table(least_popular_recipes.reset_index().rename(columns={'stars': 'Average Rating'}))
