import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from pathlib import Path
from recommender import HybridRecommender

# Page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
    /* Dark theme for the entire app */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Grid layout for movie cards */
    .movie-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        padding: 20px 0;
    }
    
    .movie-card {
        border: 2px solid #2E2E2E;
        border-radius: 12px;
        padding: 20px;
        background-color: #1E1E1E;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        height: 100%;
        margin-bottom: 20px;
    }
    
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    .movie-title {
        font-size: 20px !important;
        font-weight: 600 !important;
        margin-bottom: 10px !important;
        color: #FFFFFF !important;
    }
    
    .movie-genres {
        color: #AAAAAA !important;
        font-size: 14px;
        margin-bottom: 15px;
        font-style: italic;
    }
    
    .movie-explanation {
        background-color: #2E2E2E;
        border-left: 4px solid #4CAF50;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 0 4px 4px 0;
        color: #DDDDDD;
    }
    
    /* Star rating styling */
    .rating-container {
        display: flex;
        justify-content: center;
        gap: 5px;
        margin-top: 10px;
    }
    
    .star-button {
        background: none;
        border: none;
        font-size: 24px;
        color: #d4d4d4;
        cursor: pointer;
        transition: color 0.2s;
        padding: 0 5px;
    }
    
    .star-button:hover {
        color: #ffd700;
    }
    
    .star-filled {
        color: #ffd700;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E;
        padding: 2rem 0;
    }
    
    /* Rating history styling */
    .rating-history-item {
        background-color: #262730;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
            
    /* Style the star rating buttons */
    button {
        background: none;
        border: none;
        color: #ffd700;
        font-size: 24px;
        padding: 0 5px;
        transition: transform 0.1s;
    }
            
    button:hover {
        transform: scale(1.2);
    }

    /* Fix column gaps */
    .stHorizontalBlock {
        gap: 1rem !important;
        padding: 0.5rem 0;
    }
    
    /* Improve spacing between cards */
    .element-container {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'movies_df' not in st.session_state:
    st.session_state.movies_df = None
if 'current_recommendations' not in st.session_state:
    st.session_state.current_recommendations = None

# Load data
@st.cache_data
def load_data():
    movies_df = pd.read_csv('./data/movies.csv')
    ratings_df = pd.read_csv('./data/ratings.csv')
    tags_df = pd.read_csv('./data/tags.csv')
    return movies_df, ratings_df, tags_df

# Initialize recommender
@st.cache_resource
def init_recommender(movies_df, ratings_df, tags_df):
    recommender = HybridRecommender(use_gpu=True)
    recommender.fit(movies_df, ratings_df, tags_df)
    return recommender

def get_all_genres(movies_df):
    genres_list = []
    for genres in movies_df['genres'].dropna():
        if isinstance(genres, str):
            genres_list.extend(genres.split('|'))
    return sorted(set(genres_list))

def plot_rated_movies_genre_distribution():
    if not st.session_state.user_ratings:
        return None
    
    rated_movies = st.session_state.movies_df[
        st.session_state.movies_df['movieId'].isin(st.session_state.user_ratings.keys())
    ]
    
    all_genres = rated_movies['genres'].str.split('|').explode()
    genre_counts = all_genres.value_counts()
    
    fig = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        title="Genre Distribution of Your Rated Movies",
        labels={'x': 'Genre', 'y': 'Count'}
    )
    
    fig.update_traces(marker_color='#4CAF50')
    fig.update_layout(
        showlegend=False,
        height=300,
        margin=dict(t=50, b=20, l=20, r=20),
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF'),
        title_x=0.5,
        title_y=0.95
    )
    
    return fig

def rate_movie(movie_id, rating):
    st.session_state.user_ratings[movie_id] = rating
    if st.session_state.recommender:
        recommendations = st.session_state.recommender.get_recommendations(
            st.session_state.user_ratings,
            n_recommendations=12
        )
        st.session_state.current_recommendations = recommendations
    st.rerun()

def create_star_rating(movie_id, key_prefix):
    cols = st.columns(5)
    current_rating = st.session_state.user_ratings.get(movie_id, 0)
    
    for i in range(5):
        with cols[i]:
            if st.button(
                "★" if i < current_rating else "☆", 
                key=f"{key_prefix}_star_{i+1}"
            ):
                rate_movie(movie_id, i + 1)

def display_movie_grid(display_df):
    # Create rows of 3 movies each
    for i in range(0, len(display_df), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(display_df):
                with cols[j]:
                    movie = display_df.iloc[i + j]
                    with st.container():
                        st.markdown(f"""
                            <div class="movie-card">
                                <div class="movie-title">{movie['title']}</div>
                                <div class="movie-genres">{movie['genres']}</div>
                                {f'<div class="movie-explanation">{movie["recommendation_explanation"]}</div>' 
                                  if 'recommendation_explanation' in movie else ''}
                            </div>
                        """, unsafe_allow_html=True)
                        create_star_rating(movie['movieId'], f"movie_{i}_{j}")

def display_movie_card(movie, key_prefix):
    card_html = f"""
    <div class="movie-card">
        <div class="movie-title">{movie['title']}</div>
        <div class="movie-genres">{movie['genres']}</div>
        {'<div class="movie-explanation">' + movie["recommendation_explanation"] + '</div>' 
          if 'recommendation_explanation' in movie else ''}
        {create_star_rating(movie['movieId'], key_prefix)}
    </div>
    """
    return card_html

def render_sidebar():
    with st.sidebar:
        st.markdown("## Movie Filters")
        
        # Search box with improved styling
        st.session_state.search_query = st.text_input(
            "Search movies",
            placeholder="Enter movie title...",
            help="Search by movie title",
            key="search_input"  # Add a key for the input
        )
        
        # Genre filter
        all_genres = get_all_genres(st.session_state.movies_df)
        st.session_state.selected_genre = st.selectbox(
            "Filter by genre",
            ["All"] + all_genres,
            help="Select a genre to filter movies",
            key="genre_select"  # Add a key for the selectbox
        )
        
        # Divider between filters and history
        st.divider()
        
        # Rating history section
        st.markdown("## Rating History")
        
        if st.session_state.user_ratings:
            movies_data = st.session_state.movies_df.set_index('movieId')
            
            # Sort ratings by most recent (or by rating value)
            sorted_ratings = sorted(
                st.session_state.user_ratings.items(),
                key=lambda x: x[1],  # Sort by rating value
                reverse=True  # Highest ratings first
            )
            
            for movie_id, rating in sorted_ratings:
                if movie_id in movies_data.index:
                    movie_title = movies_data.loc[movie_id, 'title']
                    
                    # Custom HTML for each rated movie
                    st.markdown(f"""
                        <div style="
                            padding: 8px;
                            margin: 4px 0;
                            border-radius: 4px;
                            background-color: rgba(46, 46, 46, 0.3);
                        ">
                            <div style="
                                font-size: 0.9em;
                                margin-bottom: 4px;
                                color: #ffffff;
                            ">{movie_title}</div>
                            <div style="
                                color: #ffd700;
                                font-size: 16px;
                            ">{"★" * int(rating)}{"☆" * (5 - int(rating))}</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No movies rated yet. Start rating movies to build your history!")

def main():
    # Load data first
    if st.session_state.movies_df is None:
        movies_df, ratings_df, tags_df = load_data()
        if movies_df is not None:
            st.session_state.movies_df = movies_df
            st.session_state.recommender = init_recommender(
                movies_df, ratings_df, tags_df
            )
        else:
            st.error("Failed to load data. Please check your data files.")
            return
    
    # Initialize search and filter in session state if they don't exist
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ''
    if 'selected_genre' not in st.session_state:
        st.session_state.selected_genre = 'All'
    
    # Render sidebar
    render_sidebar()
    
    # Main content title
    st.title("Movie Recommender")
    
    # Start with complete dataset
    display_df = st.session_state.movies_df.copy()
    
    # Apply filters first
    if st.session_state.search_query:
        display_df = display_df[
            display_df['title'].str.contains(
                st.session_state.search_query, 
                case=False, 
                na=False
            )
        ]
    
    if st.session_state.selected_genre != "All":
        display_df = display_df[
            display_df['genres'].str.contains(
                st.session_state.selected_genre, 
                case=True, 
                na=False
            )
        ]
    
    # After filtering, check if we should show recommendations
    if st.session_state.user_ratings and not st.session_state.search_query and st.session_state.selected_genre == "All":
        # Only show recommendations if there's no active search or filter
        if st.session_state.current_recommendations is not None:
            display_df = st.session_state.current_recommendations.copy()
            st.subheader("Recommended for You")
        else:
            st.subheader("Top Movies")
    else:
        # Show search/filter results
        if st.session_state.search_query or st.session_state.selected_genre != "All":
            st.subheader("Search Results")
        else:
            st.subheader("All Movies")
    
    # Display movies in grid
    if not display_df.empty:
        display_movie_grid(display_df.head(12))
    else:
        st.write("No movies found matching your criteria.")

if __name__ == "__main__":
    main()
