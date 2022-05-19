import streamlit as st
from content import *
from collaborative import *
from scipy.sparse import hstack
from sklearn.metrics.pairwise import linear_kernel as kernel
from sklearn.cluster import KMeans


def run_collaborative(st_column):
    @st.cache
    def init():
        user_ratings = load_ratings_data('/Users/shafirasulov/anaconda_mysql/notebooks/FINAL_PROJECT-MOVIE_RECOMMENDATION/Python/data/ratings_small.csv')
        return prepare_dataset(user_ratings)

    train, test, train_pivot, cosine_sim = init()
    st_column.header('Collaborative')
    user_id = st_column.number_input('Enter user id:', value=15)
    recommendations = get_collaborative_recommendations(user_id, train, train_pivot, cosine_sim)
    user_actual_ratings = test[test.userId == user_id][['movieId', 'rating']]
    df = pd.merge(user_actual_ratings, recommendations, right_on='movieId', left_on='movieId')
    df['diff'] = abs(df.rating - df.rating_predicted)
    df.sort_values('diff', inplace=True)

    mae = df['diff'].mean()
    df.drop('diff', axis=1, inplace=True)
    st_column.dataframe(df)
    st_column.text(f'Mean absolute error: {mae}')


@st.cache
def init():
    movies_file = '/Users/shafirasulov/anaconda_mysql/notebooks/FINAL_PROJECT-MOVIE_RECOMMENDATION/Python/data/movies_metadata_raw.csv'
    keywords_file = '/Users/shafirasulov/anaconda_mysql/notebooks/FINAL_PROJECT-MOVIE_RECOMMENDATION/Python/data/keywords_raw.csv.zip'
    movies = load_movies(movies_file, keywords_file)

    tfidf_overview = get_tfidf(movies, 'overview')

    # TfIdf keywords column
    tfidf_keywords = get_tfidf(movies, 'keywords')

    # Concat keywords and overview tfidf matrices
    tfidf_matrix = hstack([tfidf_keywords, tfidf_overview])

    # Calculate similarity between all movie pairs
    cosine_sim = kernel(tfidf_matrix, tfidf_matrix)

    return movies, tfidf_matrix, cosine_sim


@st.cache
def get_kmeans(data):
    kmeans = KMeans(n_clusters=50)
    kmeans.fit(data)
    return kmeans


def run_content(st_column):
    movies, tfidf_matrix, cosine_sim = init()

    st_column.header('Content-based')
    title = st_column.text_input('Enter movie name:', value='Titanic')
    recommendations = get_recommendations(movies, title, cosine_sim)

    st_column.dataframe(recommendations)


def run_kmeans(st_column):
    movies, tfidf_matrix, cosine_sim = init()
    kmeans = get_kmeans(tfidf_matrix)

    title = st_column.text_input('Recommended movies with Kmeans', value='Titanic')
    movie_index = movies[movies.title == title].index[0]
    cluster = kmeans.predict(tfidf_matrix[movie_index])
    recommendations = movies[['title']][kmeans.labels_ == cluster]

    st_column.dataframe(recommendations)


if __name__ == '__main__':
    st.title('Movie Recommendation System')
    model = st.sidebar.selectbox(
        "Please choose filtering strategy:",
        ("Content-based", "Collaborative", "Kmeans")
    )

    # col1, col2 = st.columns(2)
    if model == 'Content-based':
        run_content(st)
    elif model == 'Kmeans':
        run_kmeans(st)
    else:
        run_collaborative(st)
