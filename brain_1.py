# Import all required libs
import pandas as pd
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, jsonify, request, render_template
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

# Genre ID to name mapping (from TMDB: https://developer.themoviedb.org/reference/genre-movie-list)
genre_id_to_name = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western"
}

# Function to preprocess the user input and extract genre IDs
def extract_genre_ids_from_text(text):
    # Reverse the genre_id_to_name dictionary to map genre names to IDs
    name_to_genre_id = {genre_name.lower(): genre_id for genre_id, genre_name in genre_id_to_name.items()}
    
    # Initialize a list to store the extracted genre IDs
    extracted_genre_ids = []
    
    # Tokenize the text prompt and convert to lowercase
    tokens = text.lower().split()
    
    # Check each token against the genre names (converted to lowercase) and add the corresponding IDs to the list
    for token in tokens:
        genre_id = name_to_genre_id.get(token)
        if genre_id:
            extracted_genre_ids.append(genre_id)
    
    return extracted_genre_ids

# Function to recommend movies based on user preferences and genre similarity
def recommend_movies(genres_to_recommend):
    recommended_movies = []
    for genre in genres_to_recommend:
        # Find movies with the highest similarity to the selected genre
        similar_movies_indices = movie_similarity[:, genre].argsort()[::-1]
        # Exclude the selected genre itself
        similar_movies_indices = similar_movies_indices[similar_movies_indices != genre]
        # Get recommended movies for this genre
        recommended_movies.extend([genre_id_to_name.get(idx) for idx in similar_movies_indices])
    return recommended_movies

# Function to fetch latest movies from an external API
def fetch_latest_movies(genre_ids):
    try:
        # Read API key from text file, no leaking keys
        with open("keys.txt") as keys_file:
            API_KEY = keys_file.readline().rstrip()

        # Construct the URL with the provided genre IDs
        GENRES_ID_STR = "%2C".join(str(genre_id) for genre_id in genre_ids)
        url = f"https://api.themoviedb.org/3/discover/movie?include_adult=true&include_video=false&language=en-US&page=1&sort_by=popularity.desc&with_genres={GENRES_ID_STR}"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        # Fetch data from the API
        response = requests.get(url, headers=headers)
        print("Response status code:", response.status_code)  # Debugging statement
        data = response.json()
        print("Data:", data)  # Debugging statement

        # Extract results from the response
        latest_movies = data.get('results', [])
        return latest_movies

    except requests.exceptions.RequestException as e:
        print(f"Error fetching latest movies: {e}")
        return []

# Function to fetch movie ratings from the API
def fetch_movie_ratings(genre_ids):
    try:
        # Read API key from text file, no leaking keys
        with open("keys.txt") as keys_file:
            API_KEY = keys_file.readline().rstrip()

        # Construct the URL with the provided genre IDs
        GENRES_ID_STR = "%2C".join(str(genre_id) for genre_id in genre_ids)
        url = f"https://api.themoviedb.org/3/discover/movie?include_adult=true&include_video=false&language=en-US&page=1&sort_by=popularity.desc&with_genres={GENRES_ID_STR}"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }

        # Fetch data from the API
        response = requests.get(url, headers=headers)
        print("Response status code:", response.status_code)  # Debugging statement
        data = response.json()
        print("Data:", data)  # Debugging statement

        # Extract movie IDs and ratings from the response
        movie_ratings = {}
        for movie in data.get('results', []):
            movie_id = movie.get('id')
            rating = movie.get('vote_average')
            if movie_id and rating:
                adj_rating = 0.5 * rating
                movie_ratings[str(movie_id)] = adj_rating
        
        # Create a pandas Series with movie IDs as index and ratings as values
        movie_ratings_series = pd.Series(movie_ratings, name='rating')
        return movie_ratings_series
    except requests.exceptions.RequestException as e:
        print(f"Error fetching movie ratings: {e}")
        return pd.Series()  # Return an empty Series in case of an error

# Function to preprocess latest movies before calculating similarity
def preprocess_latest_movies(latest_movies, movie_ratings):
    # Initialize lists to store movie descriptions and ratings
    movie_descriptions = []
    ratings = []

    # Iterate through the latest movies
    for movie in latest_movies:
        # Extract relevant features (title, overview, genres) and concatenate them
        description = f"{movie['title']} {movie.get('overview', '')} {' '.join([genre_id_to_name.get(genre_id, '') for genre_id in movie.get('genre_ids', [])])}"
        # Check if the description is empty or contains only stop words
        if description.strip():
            movie_descriptions.append(description)
            # Get the rating for the current movie ID
            ratings.append(movie_ratings.get(str(movie['id']), 0.0))

    # If all descriptions are empty or contain only stop words, return None
    if not movie_descriptions:
        return None

    # Tokenize the movie descriptions
    vectorizer = CountVectorizer(stop_words='english')
    movie_vectors = vectorizer.fit_transform(movie_descriptions)

    # Get the tokenized words
    tokenized_words = vectorizer.get_feature_names_out()

    # Calculate average rating for each genre
    genre_ratings = {}
    for movie in latest_movies:
        for genre_id in movie.get('genre_ids', []):
            genre_name = genre_id_to_name.get(genre_id)
            if genre_name:
                genre_ratings.setdefault(genre_name, []).append(movie_ratings.get(str(movie['id']), 0.0))

    avg_genre_ratings = {genre: sum(ratings) / len(ratings) for genre, ratings in genre_ratings.items()}

    # Create a list to store the average rating for each movie's genre
    avg_ratings = [avg_genre_ratings.get(genre_id_to_name.get(genre_id), 0.0) for movie in latest_movies for genre_id in movie.get('genre_ids', [])]

    # Concatenate the movie vectors with the average genre ratings
    combined_vectors = pd.concat([pd.DataFrame(movie_vectors.toarray(), columns=tokenized_words), pd.Series(ratings, name='rating'), pd.Series(avg_ratings, name='avg_rating')], axis=1)

    # Replace NaN values with 0
    combined_vectors = combined_vectors.fillna(0)

    return combined_vectors

# Function to recommend movies based on a list of genres
def recommend_movies_by_genres(genres, latest_movies, movie_ratings):
    # Extract titles of latest movies
    latest_movie_titles = [movie['title'] for movie in latest_movies]

    # Preprocess latest movies before calculating similarity
    combined_vectors = preprocess_latest_movies(latest_movies, movie_ratings)

    if combined_vectors is None:
        return [], latest_movie_titles

    recommended_movies = []
    
    # Calculate cosine similarity for each genre separately
    for genre in genres:
        # Convert genre ID to genre name
        genre_name = genre_id_to_name.get(genre, "")
        if not genre_name:
            continue
        
        # Filter movies by genre
        genre_movies = [movie for movie in latest_movies if genre_name in [genre_id_to_name.get(g, "") for g in movie.get('genre_ids', [])]]
        genre_indices = [i for i, movie in enumerate(latest_movies) if genre_name in [genre_id_to_name.get(g, "") for g in movie.get('genre_ids', [])]]
        # Print genre movies and indices for debugging
        print(f"Genre: {genre_name}, Number of Movies: {len(genre_movies)}, Genre Indices: {genre_indices}")

        # Print the number of movies being considered
        print(f"Number of movies being considered for {genre_name}: {len(genre_movies)}")

        genre_vectors = combined_vectors.iloc[genre_indices]
        
        # Print genre vectors for debugging
        print(f"Genre Vectors:\n{genre_vectors}")    
        
        # Check if there are movies for this genre
        if not genre_movies:
            continue

        # Calculate cosine similarity only if there are movies available
        genre_similarity = cosine_similarity(genre_vectors)
        print("Genre similarity matrix:", genre_similarity)  # Debugging statement
        
        # Select top recommendations for this genre
        for i in range(len(genre_movies)):
            # Get the indices of movies with highest similarity to the current movie
            similar_movies_indices = genre_similarity[i].argsort()[::-1]
            # Exclude the current movie itself
            similar_movies_indices = [index for index in similar_movies_indices if index != i]
            # Ensure indices are within the range of genre_movies
            similar_movies_indices = [index for index in similar_movies_indices if 0 <= index < len(genre_movies)]
            # Add the titles of recommended movies to the list
            recommended_movies.extend([genre_movies[index]['title'] for index in similar_movies_indices])

    # Ensure only 10 movies are recommended
    recommended_movies = recommended_movies

    return recommended_movies, latest_movie_titles

# Initialize Flask application
app = Flask(__name__, template_folder="templates")

# Define API endpoint for recommending movies
@app.route('/recommend_movies', methods=['GET'])
def get_recommendations():
    # Get the text prompt from the user
    text_prompt = request.args.get('text_prompt')
    if not text_prompt:
        return jsonify({'error': 'Text prompt is required'}), 400
    
    # Extract genres from the text prompt
    genres_to_recommend = extract_genre_ids_from_text(text_prompt)

    # Fetch latest movies from an external API
    latest_movies = fetch_latest_movies(genres_to_recommend)
    print("Latest movies:", latest_movies)  # Debugging statement

    # Fetch movie ratings
    movie_ratings = fetch_movie_ratings(genres_to_recommend)
    print("Movies ratings:", movie_ratings)  # Debugging statement

    # Recommend movies based on the specified genres and movie ratings
    recommended_movies, latest_movie_titles = recommend_movies_by_genres(genres_to_recommend, latest_movies, movie_ratings)

    print("Recommended movies:", recommended_movies)  # Debugging statement

    # Prepare response with recommended movies and latest movie titles
    response_data = {
        'recommended_movies': recommended_movies,
        'latest_movie_titles': latest_movie_titles
    }

    return jsonify(response_data)

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
