import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Insights Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for insights page
st.markdown("""
<style>
    .metric-card {
        border: 2px solid #2E2E2E;
        border-radius: 12px;
        padding: 20px;
        background-color: #1E1E1E;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #4CAF50;
    }
    
    .metric-label {
        font-size: 16px;
        color: #AAAAAA;
    }
    
    .genre-list {
        list-style-type: none;
        padding: 0;
    }
    
    .genre-item {
        padding: 8px 0;
        border-bottom: 1px solid #2E2E2E;
    }
    
    .rating-stars {
        color: #ffd700;
        font-size: 18px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E;
        padding: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

def calculate_rating_stats(user_ratings, movies_df):
    """Calculate various rating statistics"""
    if not user_ratings:
        return None
    
    ratings = pd.Series(user_ratings)
    stats = {
        'total_ratings': len(ratings),
        'average_rating': ratings.mean(),
        'rating_counts': ratings.value_counts().sort_index(),
        'rating_percentages': (ratings.value_counts(normalize=True) * 100).sort_index()
    }
    return stats

def calculate_genre_stats(user_ratings, movies_df):
    """Calculate genre-based statistics"""
    if not user_ratings:
        return None
    
    # Get rated movies
    rated_movies = movies_df[movies_df['movieId'].isin(user_ratings.keys())]
    
    # Calculate genre ratings
    genre_ratings = {}
    genre_counts = {}
    
    for idx, row in rated_movies.iterrows():
        rating = user_ratings[row['movieId']]
        genres = row['genres'].split('|')
        
        for genre in genres:
            if genre not in genre_ratings:
                genre_ratings[genre] = []
            genre_ratings[genre].append(rating)
    
    # Calculate averages
    genre_averages = {
        genre: np.mean(ratings)
        for genre, ratings in genre_ratings.items()
    }
    
    # Get all possible genres
    all_genres = set()
    for genres in movies_df['genres'].str.split('|'):
        all_genres.update(genres)
    
    # Find unexplored genres
    explored_genres = set(genre_ratings.keys())
    unexplored_genres = all_genres - explored_genres
    
    # Calculate diversity score
    genre_coverage = len(explored_genres) / len(all_genres)
    genre_balance = 1 - np.std([len(ratings) for ratings in genre_ratings.values()]) / np.mean([len(ratings) for ratings in genre_ratings.values()])
    diversity_score = int((genre_coverage * 0.6 + genre_balance * 0.4) * 100)
    
    return {
        'genre_averages': genre_averages,
        'unexplored_genres': unexplored_genres,
        'diversity_score': diversity_score,
        'genre_coverage': genre_coverage
    }

def plot_rating_distribution(rating_stats):
    """Create rating distribution chart"""
    if not rating_stats:
        return None
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=['⭐' * i for i in range(1, 6)],
        y=rating_stats['rating_percentages'],
        marker_color=['#FF4444', '#FF9933', '#FFDD33', '#99CC33', '#44BB44'],
        text=[f"{val:.1f}%" for val in rating_stats['rating_percentages']],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Your Rating Distribution",
        xaxis_title="Rating",
        yaxis_title="Percentage of Ratings",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF'),
        showlegend=False,
        height=400
    )
    
    return fig

def plot_genre_coverage(genre_stats):
    """Create genre coverage radar chart"""
    if not genre_stats:
        return None
    
    genre_avg = pd.Series(genre_stats['genre_averages'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=genre_avg.values,
        theta=genre_avg.index,
        fill='toself',
        marker_color='#4CAF50'
    ))
    
    fig.update_layout(
        title="Genre Coverage",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF'),
        height=400
    )
    
    return fig

def main():
    st.title("Insights Dashboard")
    
    # Load data if not in session state
    if 'movies_df' not in st.session_state:
        st.error("Please rate some movies on the main page first!")
        return
    
    # Calculate statistics
    rating_stats = calculate_rating_stats(
        st.session_state.user_ratings,
        st.session_state.movies_df
    )
    
    genre_stats = calculate_genre_stats(
        st.session_state.user_ratings,
        st.session_state.movies_df
    )
    
    if not rating_stats or not genre_stats:
        st.warning("Start rating movies to see your insights!")
        return
    
    # Sidebar - Discovery Opportunities
    with st.sidebar:
        st.header("Genres to explore:")
        
        if genre_stats['unexplored_genres']:
            for genre in sorted(genre_stats['unexplored_genres']):
                st.markdown(f"{genre}")
        else:
            st.success("Impressive! You've explored all available genres!")
    
    # Main Content
    # Overview Section
    col1, col2, col3 = st.columns(3)
    
    # Total Movies Rated
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{rating_stats['total_ratings']}</div>
                <div class="metric-label">Movies Rated</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Average Rating
    with col2:
        avg_rating = rating_stats['average_rating']
        stars = "⭐" * int(round(avg_rating))
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stars} ({avg_rating:.1f})</div>
                <div class="metric-label">Average Rating</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Genre Diversity Score
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{genre_stats['diversity_score']}/100</div>
                <div class="metric-label">Genre Diversity Score</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Rating Analysis and Genre Coverage in same row
    col1, col2 = st.columns(2)
    
    with col1:
        rating_dist_fig = plot_rating_distribution(rating_stats)
        if rating_dist_fig:
            st.plotly_chart(rating_dist_fig, use_container_width=True)
    
    with col2:
        genre_coverage_fig = plot_genre_coverage(genre_stats)
        if genre_coverage_fig:
            st.plotly_chart(genre_coverage_fig, use_container_width=True)

if __name__ == "__main__":
    main()