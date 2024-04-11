# Import all required libs
import pandas as pd
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, jsonify
import requests

# Replace 'your_file.csv' with the correct file path
file_path = "/home/forenche/Music/movie_dataset.csv"

# Read the CSV file into a list of lists, skipping lines with too many fields
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    try:
        for row in csv_reader:
            data.append(row)
    except csv.Error as e:
        pass

# Convert the list of lists to a DataFrame
df = pd.DataFrame(data)

print("Printing columns",df.columns)

# Display the DataFrame
print(df)

print("DESCRIE\n")
print(df.describe)
print("HEAD\n")
print(df.head)
print("SHAPE\n")
print(df.shape)

print("DESCRBE AFTER DOING NULL\n")
df.isnull().sum()
print(df.describe)

print("Describe after null null\n")
df = df.dropna().copy()
print("HEAD\n")
print(df.head)

df.columns = ['movieId', 'title', 'genres', 'rating']
genress = df['genres'].str.get_dummies(sep='|')
# Set appropriate column names for the genress
genress.columns = [f'genres_{genres}' for genres in genress.columns]

# Combine the original DataFrame with the genres columns
df = pd.concat([df, genress], axis=1)

# Pop 'genres_genres' as it is a placeholder
df.pop('genres_genres')

# Drop 'genres' as we will concat all genres as columns
df.drop(columns=['genres'], inplace=True)
print("HEAD after dropping genres\n")
print(df.head)

print("printing columns\n")
print(df.columns)

print("DESCRIBE\n")
print(df.describe)

print("SHAPE\n")
print(df.shape)

# Print missing values for sanity
missing_values = df.isnull().sum()
print("missing_values:\n",missing_values)

df['title'].fillna('Unknown', inplace=True)
print("data type: \n",df.dtypes)

print("printing columns\n")
print(df.columns)
df['movieId'] = pd.to_numeric(df['movieId'], errors='coerce')
#print("HEAD after movied\n")
print(df.head)

# Convert 'rating' column to numeric type
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
#df['rating'] = df['rating'].astype(int)
print("Printing rating: \n",df.rating)

print(df.columns)

# Get genre columns
genre_columns = [col for col in df.columns if col.startswith('genres_')]

# Pivot the DataFrame to get a matrix of movie ratings based on genres
movie_ratings_pivot = df.pivot_table(index='movieId', values=genre_columns, aggfunc='mean')

# Fill missing values with 0
movie_ratings_pivot.fillna(0, inplace=True)

# Calculate cosine similarity between movies based on their genre ratings
movie_similarity = cosine_similarity(movie_ratings_pivot)

# Function to preprocess latest movies before calculating similarity
def preprocess_latest_movies(latest_movies):
    # Concatenate relevant features for each movie (e.g., title, overview, genres)
    movie_descriptions = [f"{movie['title']} {movie.get('overview', '')} {' '.join(movie.get('genres', []))}" for movie in latest_movies]

    # Vectorize the movie descriptions
    vectorizer = CountVectorizer(stop_words='english')
    movie_vectors = vectorizer.fit_transform(movie_descriptions)

    return movie_vectors

# Function to recommend movies based on a list of genres
def recommend_movies_by_genres(genres, latest_movies, movie_ratings, num_recommendations=10):
    # Extract titles of latest movies
    latest_movie_titles = [movie['title'] for movie in latest_movies]

    # Preprocess latest movies before calculating similarity
    movie_vectors = preprocess_latest_movies(latest_movies)

    # Combine genre vectors with movie ratings
    combined_vectors = pd.concat([pd.DataFrame(movie_vectors.toarray()), movie_ratings.reset_index(drop=True)], axis=1)

    # Handle missing values (NaN) in the combined vectors
    combined_vectors = combined_vectors.fillna(0)  # Replace NaN with 0

    # Calculate cosine similarity between movies based on their genre ratings and movie ratings
    movie_similarity = cosine_similarity(combined_vectors)

    # Select the movie with the highest average similarity score
    avg_similarity = movie_similarity.mean(axis=1)
    top_indices = avg_similarity.argsort()[::-1][:num_recommendations]

    # Get the titles of the top recommended movies
    recommended_movies = [latest_movies[i]['title'] for i in top_indices]

    return recommended_movies, latest_movie_titles

# (TODO)Dummy function to fetch movie ratings
def fetch_movie_ratings():
    # Example function to fetch movie ratings (you need to implement this)
    # This function can fetch ratings from a database, another API, or any other source
    # For demonstration purposes, we'll return dummy ratings here
    movie_ratings = pd.Series([4.5, 3.8, 4.2, 3.9, 4.1, 3.7, 4.0, 4.3, 4.4, 3.6], name='rating')
    return movie_ratings

# Read API key from text file, no leaking keys
keys_file = open("keys.txt")
lines = keys_file.readlines()
API_KEY = lines[0].rstrip()

# Initialize Flask application
app = Flask(__name__)

# Function to fetch latest movies from an external API
def fetch_latest_movies():
    # Example API call to fetch latest movies from The Movie Database (TMDB) API
    response = requests.get(f'https://api.themoviedb.org/3/movie/now_playing?api_key={API_KEY}')
    print(response.json())
    latest_movies = response.json().get('results', [])  # Access 'results' key safely
    return latest_movies

# Define API endpoint for recommending movies
@app.route('/recommend_movies', methods=['GET'])
def get_recommendations():
    # Fetch latest movies from an external API
    latest_movies = fetch_latest_movies()

    # Fetch movie ratings
    movie_ratings = fetch_movie_ratings()

    # Specify genres for recommendation
    genres_to_recommend = ['genres_Comedy', 'genres_Romance']  # Example list of genres

    # Ensure num_recommendations is a scalar value (integer)
    num_recommendations = 10  # Example value, you can replace it with the desired number
    
    # Recommend movies based on the specified genres and movie ratings
    recommended_movies, latest_movie_titles = recommend_movies_by_genres(genres_to_recommend, latest_movies, movie_ratings, num_recommendations)

    # Prepare response with recommended movies and latest movie titles
    response_data = {
        'recommended_movies': recommended_movies,
        'latest_movie_titles': latest_movie_titles
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)

# Example usage: Recommend movies based on a list of genres, TODO: make this a callable function instead of hardcoding movie genres
#genres_to_recommend = ['genres_Comedy', 'genres_Romance']  # Example list of genres, nuke this
#recommended_movies = recommend_movies_by_genres(genres_to_recommend)
#print("Recommended Movies for Genres {}:\n".format(genres_to_recommend))
#print(recommended_movies)

# TODO: NLP and API from https://developer.themoviedb.org/docs/getting-started to fetch latest movies accordingly.
# TODO: Frontend for prompts.