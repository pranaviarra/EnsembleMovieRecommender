import pandas as pd

# Load the datasets
movies = pd.read_csv('/Users/ramesharra/movie_recommender/data/tmdb_5000_movies.csv')
credits = pd.read_csv('/Users/ramesharra/movie_recommender/data/tmdb_5000_credits.csv')

# Check columns to ensure movie_id is present
print("Movies columns:", movies.columns)
print("Credits columns:", credits.columns)

# Strip any leading/trailing spaces in column names
movies.columns = movies.columns.str.strip()
credits.columns = credits.columns.str.strip()

# Check again
print(movies.columns)
print(credits.columns)

print('movie_id' in movies.columns)  # Should print True if exists
print('movie_id' in credits.columns)  # Should print True if exists

# Check if 'id' exists in movies and rename it
if 'id' in movies.columns:
    movies = movies.rename(columns={'id': 'movie_id'})

print('movie_id' in movies.columns)  
print('movie_id' in credits.columns) 

# Merge the datasets on 'movie_id'
movies = movies.merge(credits, on='movie_id')

# Check the first few rows to verify merge
print(movies.head())

# Check the first few entries of 'genres', 'keywords', and 'cast' columns
print(movies['genres'].head())
print(movies['keywords'].head())
print(movies['cast'].head())

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cosine
from collections import Counter

# Assuming 'movies' is your dataset with columns: 'movie_id', 'title', 'tags'
# Create a new 'tags' column as described earlier
movies['tags'] = movies['overview'] + " " + \
                  movies['genres'].apply(lambda x: " ".join([i['name'] for i in eval(x)])) + " " + \
                  movies['keywords'].apply(lambda x: " ".join([i['name'] for i in eval(x)])) + " " + \
                  movies['cast'].apply(lambda x: " ".join([i['name'] for i in eval(x)]))
movies['tags'] = movies['tags'].astype(str).apply(lambda x: x.lower())

# Feature extraction using TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, movies['movie_id'], test_size=0.2, random_state=42)

# Initialize models
svm_model = SVC(kernel='linear', probability=True)
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Fit models on the training data
svm_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

def get_recommendations(movie_title, n_recommendations=5):
    # Convert input movie title to TF-IDF vector
    movie_idx = movies[movies['title'] == movie_title].index[0]
    movie_vector = tfidf_matrix[movie_idx]
    
    # Predictions from all models
    svm_preds = svm_model.predict([movie_vector])
    dt_preds = dt_model.predict([movie_vector])
    knn_preds = knn_model.predict([movie_vector])
    gb_preds = gb_model.predict([movie_vector])
    
    # Voting mechanism: Use Counter to count the votes and return the most common movie ids
    all_predictions = [svm_preds[0], dt_preds[0], knn_preds[0], gb_preds[0]]
    
    # Get the most common movie id from the predictions
    movie_votes = Counter(all_predictions)
    most_common_movie = movie_votes.most_common(1)[0][0]
    
    # Get the index of the most common movie
    similar_movie_idx = movies[movies['movie_id'] == most_common_movie].index[0]
    
    # Recommend top n similar movies based on the cosine similarity
    cosine_sim = cosine(tfidf_matrix[movie_idx].toarray(), tfidf_matrix[similar_movie_idx].toarray())
    
    # Get the indices of the most similar movies
    sim_scores = cosine_sim[0]
    sim_movie_indices = sim_scores.argsort()[:n_recommendations]
    
    # Fetch recommended movie titles
    recommended_movies = movies.iloc[sim_movie_indices]['title']
    
    return recommended_movies

from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity matrix for all movies
cosine_sim = cosine_similarity(tfidf_matrix)

def get_ensemble_recommendations(movie_title, n_recommendations=10):
    # Ensure the movie exists in the dataframe
    if movie_title not in movies['title_x'].values:
        print(f"'{movie_title}' not found in dataset.")
        return []

    # Get the index of the movie
    idx = movies[movies['title_x'] == movie_title].index[0]

    # --- 1. Cosine Similarity Recommender ---
    cosine_scores = list(enumerate(cosine_sim[idx]))
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)
    cosine_indices = [i[0] for i in cosine_scores[1:n_recommendations+1]]

    # --- 2. KNN Recommender ---
    distances, knn_indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=n_recommendations+1)
    knn_indices = knn_indices[0][1:]  # Skip the movie itself

    # --- 3. Combine Recommendations via Voting ---
    from collections import Counter
    combined_indices = cosine_indices + list(knn_indices)
    vote_counts = Counter(combined_indices)
    top_voted = [i for i, _ in vote_counts.most_common(n_recommendations)]

    # --- 4. Return Movie Titles ---
    recommended_movies = movies.iloc[top_voted]['title_x'].values.tolist()
    return recommended_movies

input_movie = "Batman"
recommendations = get_ensemble_recommendations(input_movie)

print(f"\nRecommended Movies similar to '{input_movie}':")
for i, movie in enumerate(recommendations, 1):
    print(f"{i}. {movie}")


