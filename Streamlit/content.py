import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
import ast


# Converts json field to list of values
def convert_json_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    return df[column_name].\
               apply(ast.literal_eval).\
               apply(lambda x: [a['name'] for a in x]).\
               apply(lambda x: ','.join(x))


def load_movies(movies_files: str, keywords_file: str) -> pd.DataFrame:
    movies = pd.read_csv(movies_files, index_col=[0]).reset_index()
    movies = movies[movies.vote_count > 20].reset_index(drop=True)
    movies['id'] = movies.id.astype(int)
    movies = movies.rename(columns={'id': 'movie_id'})

    keywords = pd.read_csv(keywords_file)
    keywords.drop_duplicates(inplace=True)

    # Rename id column to movie_id, and set it as index
    keywords = keywords.rename(columns={'id': 'movie_id'})

    # Merge 'movies' and 'keywords' tables on 'movie_id'
    movies = pd.merge(movies, keywords, on = 'movie_id')
    movies['overview'].fillna('', inplace=True)

    # Convert json field to comma separated keywords
    movies['keywords'] = convert_json_column(movies, 'keywords')

    movies['imdb_id'] = 'https://www.imdb.com/title/' + movies['imdb_id']

    return movies


def get_tfidf(df: pd.DataFrame, column: str):
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a','and'
    tfidf = TfidfVectorizer(stop_words='english')
    return tfidf.fit_transform(df[column])


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(movies: pd.DataFrame, title: str, cosine_sim) -> pd.DataFrame:
    # Get the index of the movie that matches the title
    idx = movies[movies.title == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies[['title']].iloc[movie_indices]