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
def preprocess_latest_movies(latest_movies, movie_ratings):
    # Initialize lists to store movie descriptions and ratings
    movie_descriptions = []
    ratings = []

    # Iterate through the latest movies
    for movie in latest_movies:
        # Extract relevant features (title, overview, genres) and concatenate them
        description = f"{movie['title']} {movie.get('overview', '')} {' '.join(movie.get('genres', []))}"
        # Check if the description is empty or contains only stop words
        if description.strip():
            movie_descriptions.append(description)
            # Get the rating for the current movie ID
            ratings.append(movie_ratings.get(str(movie['id']), 0.0))

    # If all descriptions are empty or contain only stop words, return None
    if not movie_descriptions:
        return None

    # Vectorize the movie descriptions
    vectorizer = CountVectorizer(stop_words='english')
    movie_vectors = vectorizer.fit_transform(movie_descriptions)

    # Calculate average rating for each genre
    genre_ratings = {}
    for movie in latest_movies:
        for genre in movie.get('genres', []):
            genre_ratings.setdefault(genre, []).append(movie_ratings.get(str(movie['id']), 0.0))

    avg_genre_ratings = {genre: sum(ratings) / len(ratings) for genre, ratings in genre_ratings.items()}

    # Create a list to store the average rating for each movie's genre
    avg_ratings = [avg_genre_ratings.get(genre, 0.0) for movie in latest_movies for genre in movie.get('genres', [])]

    # Concatenate the movie vectors with the average genre ratings
    combined_vectors = pd.concat([pd.DataFrame(movie_vectors.toarray()), pd.Series(ratings, name='rating'), pd.Series(avg_ratings, name='avg_rating')], axis=1)

    # Replace NaN values with 0
    combined_vectors = combined_vectors.fillna(0)

    return combined_vectors

# Function to fetch latest movies from an external API
def fetch_latest_movies():
    try:
        # Read API key from text file, no leaking keys
        with open("keys.txt") as keys_file:
            API_KEY = keys_file.readline().rstrip()

        # Example API call to fetch latest movies from The Movie Database (TMDB) API
        response = requests.get(f'https://api.themoviedb.org/3/movie/now_playing?api_key={API_KEY}')
        latest_movies = response.json().get('results', [])  # Access 'results' key safely
        return latest_movies
    except requests.exceptions.RequestException as e:
        print(f"Error fetching latest movies: {e}")
        return []

# Function to fetch movie ratings from the API
def fetch_movie_ratings():
    try:
        # Read API key from text file, no leaking keys
        with open("keys.txt") as keys_file:
            API_KEY = keys_file.readline().rstrip()

        # Example API call to fetch movie ratings from The Movie Database (TMDB) API
        response = requests.get(f'https://api.themoviedb.org/3/movie/now_playing?api_key={API_KEY}')
        data = response.json()

        # Extract movie IDs and ratings from the response
        movie_ratings = {}
        for movie in data.get('results', []):
            movie_id = movie.get('id')
            rating = movie.get('vote_average')
            if movie_id and rating:
                movie_ratings[str(movie_id)] = rating
        
        # Create a pandas Series with movie IDs as index and ratings as values
        movie_ratings_series = pd.Series(movie_ratings, name='rating')
        return movie_ratings_series
    except requests.exceptions.RequestException as e:
        print(f"Error fetching movie ratings: {e}")
        return pd.Series()  # Return an empty Series in case of an error

# Function to recommend movies based on a list of genres
def recommend_movies_by_genres(genres, latest_movies, movie_ratings, num_recommendations=10):
    # Extract titles of latest movies
    latest_movie_titles = [movie['title'] for movie in latest_movies]

    # Preprocess latest movies before calculating similarity
    combined_vectors = preprocess_latest_movies(latest_movies, movie_ratings)

    if combined_vectors is None:
        return [], latest_movie_titles

    # Calculate cosine similarity between movies based on their genre ratings and movie ratings
    movie_similarity = cosine_similarity(combined_vectors)

    # Select the movie with the highest average similarity score
    avg_similarity = movie_similarity.mean(axis=1)
    top_indices = avg_similarity.argsort()[::-1][:num_recommendations]

    # Get the titles of the top recommended movies
    recommended_movies = [latest_movies[i]['title'] for i in top_indices]

    return recommended_movies, latest_movie_titles

# Initialize Flask application
app = Flask(__name__)

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