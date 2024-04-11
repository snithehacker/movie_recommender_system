# Import all required libs
import pandas as pd
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
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

from sklearn.metrics.pairwise import cosine_similarity

# Get genre columns
genre_columns = [col for col in df.columns if col.startswith('genres_')]

# Pivot the DataFrame to get a matrix of movie ratings based on genres
movie_ratings_pivot = df.pivot_table(index='movieId', values=genre_columns, aggfunc='mean')

# Fill missing values with 0
movie_ratings_pivot.fillna(0, inplace=True)

# Calculate cosine similarity between movies based on their genre ratings
movie_similarity = cosine_similarity(movie_ratings_pivot)

# Function to recommend movies based on a list of genres
def recommend_movies_by_genres(genres, num_recommendations=10):
    # Filter movies that match the given genres
    filtered_movies = df[df[genres].sum(axis=1) == len(genres)]

    # Print the total number of movies being considered for similarity calculation
    total_movies = len(filtered_movies)
    print(f"Total movies considered for similarity calculation: {total_movies}")

    # Calculate cosine similarity between movies based on their genre ratings
    movie_ratings = filtered_movies[genres]
    movie_similarity = cosine_similarity(movie_ratings)

    # Select the movie with the highest average similarity score
    avg_similarity = movie_similarity.mean(axis=1)
    top_indices = avg_similarity.argsort()[::-1][:num_recommendations]

    # Get the titles of the top recommended movies
    recommended_movies = df.iloc[top_indices]['title']

    return recommended_movies

# Example usage: Recommend movies based on a list of genres, TODO: make this a callable function instead of hardcoding movie genres
genres_to_recommend = ['genres_Comedy', 'genres_Romance']  # Example list of genres, nuke this
recommended_movies = recommend_movies_by_genres(genres_to_recommend)
print("Recommended Movies for Genres {}:\n".format(genres_to_recommend))
print(recommended_movies)

# TODO: NLP and API from https://developer.themoviedb.org/docs/getting-started to fetch latest movies accordingly.
# TODO: Frontend for prompts.