import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import linear_kernel as kernel


def load_ratings_data(ratings_file: str):
    user_ratings = pd.read_csv(ratings_file)
    # Scaling 'rating' column to fit the model
    user_ratings['rating_scaled'] = StandardScaler().fit_transform(user_ratings[['rating']])

    return user_ratings


def prepare_dataset(user_ratings: pd.DataFrame):
    train, test = train_test_split(user_ratings, test_size=0.15, random_state=7)
    # Creating pivot table,userId as columns,movieId as index and rating_scaled as values of dataframe
    train_pivot = train.pivot(index="userId", columns="movieId", values='rating_scaled').fillna(0).reset_index()
    X = train_pivot.drop('userId', axis=1)
    # Using cosine_similarity metric to find similarity between vectors
    cosine_sim = kernel(X, X)

    return train, test, train_pivot, cosine_sim


# Function that takes in movie title as input and outputs most similar movies
def get_collaborative_recommendations(user_id, train: pd.DataFrame, train_pivot: pd.DataFrame, cosine_sim):
    # Get the index of the user that matches the user_id
    idx = train_pivot[train_pivot.userId == user_id].index[0]

    # Get the pairwise similarity scores of all users with that user
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the users based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar users
    sim_scores = sim_scores[1:11]

    # Get the user indices
    user_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar users
    users = train_pivot['userId'].iloc[user_indices].to_list()

    # Return movies of similar users
    df = train[train.userId.isin(users)]

    # Filter the movies rated by at least 3 similar people
    df = df.groupby('movieId').filter(lambda x: len(x) >= 3)

    # Take the average of the user ratings for a movie as our recommendation score
    movies = df.groupby('movieId')['rating'].mean()
    movies = pd.DataFrame(movies)
    movies.columns = ['rating_predicted']

    return movies
