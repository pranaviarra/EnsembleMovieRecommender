# EnsembleMovieRecommender

This project is a content-based movie recommendation system built using Python and machine learning. It leverages the TMDb 5000 Movies Dataset to suggest similar movies based on a user's input title. The system combines multiple machine learning models to improve accuracy and robustness through an ensemble approach.

🔍 Features
Merges movie metadata and credits for rich feature extraction.

Preprocesses data to generate descriptive "tags" for each movie.

Uses TF-IDF Vectorization to convert text data into numerical features.

Trains multiple ML models:

Support Vector Machine (SVM)

Decision Tree

K-Nearest Neighbors (KNN)

Gradient Boosting

Computes cosine similarity across movies for content similarity.

Implements an ensemble recommender combining:

Model voting

Cosine similarity

KNN neighborhood proximity

Returns a ranked list of similar movies based on the input title.

📂 Dataset
tmdb_5000_movies.csv

tmdb_5000_credits.csv

🚀 How to Use
Clone the repository and ensure dependencies are installed.

Run the script and input a movie title (e.g., "Batman").

Get top recommended movies based on content similarity and ensemble prediction.

🛠️ Tech Stack
Python

Scikit-learn

Pandas / NumPy

TF-IDF Vectorizer

Cosine Similarity

**The output looks like this**
![WhatsApp Image 2025-04-28 at 22 02 01_fa2dca4a](https://github.com/user-attachments/assets/5d696887-1ef6-42ad-b77f-54cfda41af74)

