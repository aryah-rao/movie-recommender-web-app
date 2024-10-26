import streamlit as st
import pandas as pd

def render_about_page():
    st.set_page_config(
        page_title="About - Movie Recommender",
        page_icon="🎬",
        layout="wide"
    )

    # Custom CSS for better visual hierarchy
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .component-header {
            color: #FF4B4B;
        }
                
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1E1E1E;
            padding: 2rem 0;
        }
        
        /* Technical metrics table styling */
        [data-testid="stExpander"] {
            background-color: #1E1E1E !important;
            border-radius: 0.5rem;
        }
        
        /* Style the table itself */
        .styled-table {
            background-color: #1E1E1E;
            border-radius: 0.5rem;
            padding: 1rem;
        }
        
        .styled-table td, .styled-table th {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title with icon
    st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)

    # Developer Bio
    st.sidebar.markdown("""
    ### Hi, I'm Kona Venkata Aryah Arjun Rao! 👋
    
    I'm a Machine Learning Engineer passionate about building intelligent systems 
    that make a difference. This movie recommender system represents my expertise 
    in deep learning, and full-stack development.
    
    #### Connect With Me
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aryah-rao/)
    [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/aryah-rao)
    
    Feel free to reach out for collaborations or discussions about machine learning and software engineering!
    """)

    # Introduction with metrics
    st.markdown("""
    A smart movie recommendation engine that learns from your taste 
    """)

    # Benchmark Metrics
    st.markdown("### 📊 Performance Metrics")
    
    # Main metrics in columns with explanations
    perf1, perf2, perf3 = st.columns(3)
    
    with perf1:
        st.metric(
            label="Recommendation Diversity",
            value="94%",
            help="Measures the variety of movie genres and styles recommended"
        )

    with perf2:
        st.metric(
            label="Hit Rate",
            value="21%",
            help="Percentage of relevant recommendations"
        )
        
    with perf3:
        st.metric(
            label="Ranking Quality",
            value="0.26",
            help="NDCG score measuring recommendation ranking quality"
        )

    # Technical Details (collapsible)
    with st.expander("🔍 Detailed Technical Metrics"):
        tech_metrics = {
            "Precision": "0.0250",
            "Recall": "0.0078",
            "NDCG": "0.2606",
            "F1 Score": "0.0106",
            "Hit Rate": "0.2143",
            "Genre Diversity": "0.8036",
            "Novelty": "0.3214"
        }
        
        # Create and style the metrics table
        df = pd.DataFrame({
            'Metric': tech_metrics.keys(),
            'Value': tech_metrics.values()
        }).set_index('Metric')
        
        st.markdown('<div class="styled-table">', unsafe_allow_html=True)
        st.table(df)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='font-size: 0.8em; color: #666;'>
        These metrics are based on standard recommender system evaluation criteria.
        Values are calculated using the MovieLens dataset and standard evaluation protocols.
        </div>
        """, unsafe_allow_html=True)

    # Core Components in columns
    comp1, comp2, comp3 = st.columns(3)
    
    with comp1:
        st.markdown("""
        #### 🧠 Content-Based Engine
        - **BERT Analysis**: Processes movie descriptions, plot summaries, and dialogue to understand deeper thematic elements
        - **TF-IDF Analysis**: Identifies key terms and important keywords from movie metadata
        """)
    
    with comp2:
        st.markdown("""
        #### 🤝 Collaborative Engine
        - **Neural Network**: Deep learning model that learns complex patterns in user preferences
        - **Similarity Matrix**: Identifies similar users and movies based on rating patterns
        """)
    
    with comp3:
        st.markdown("""
        #### ⚖️ Smart Combination
        - **Adaptive Weighting**: Adjusts importance of different factors based on user history
        - **Popularity Integration**: Considers trending and highly-rated movies
        """)

    # How It Works in new line
    st.markdown("### ⚙️ How It Works")
    
    # Progress bars for weight distribution
    st.markdown("#### New Users")
    col_a, col_b, col_c = st.columns([7, 2, 1])
    with col_a:
        st.progress(0.7, "Content")
    with col_b:
        st.progress(0.2, "Collaborative")
    with col_c:
        st.progress(0.1, "Popular")
    
    st.markdown("#### Regular Users")
    col_d, col_e, col_f = st.columns([3, 6, 1])
    with col_d:
        st.progress(0.3, "Content")
    with col_e:
        st.progress(0.6, "Collaborative")
    with col_f:
        st.progress(0.1, "Popular")

    # Quick Start Guide with icons
    st.markdown("### 🚀 Quick Start")
    guide_col1, guide_col2, guide_col3 = st.columns(3)
    
    with guide_col1:
        st.info("#### 1️⃣ Rate Movies\n- Start with 5+ movies\n- Mix different genres\n- Be honest with ratings")
    
    with guide_col2:
        st.success("#### 2️⃣ Explore\n- Check explanations\n- Try genre filters\n- Discover new movies")
    
    with guide_col3:
        st.warning("#### 3️⃣ Improve\n- Rate suggestions\n- Update preferences\n- Use search")

    # Footer with privacy and feedback
    st.divider()
    footer_col1, footer_col2 = st.columns(2)
    
    with footer_col1:
        st.markdown("""
        ### 🔒 Privacy
        - Session-only storage
        - No personal data kept
        - Anonymous statistics
        """)
    
    with footer_col2:
        st.markdown("""
        ### 💬 Feedback
        Found a bug or have suggestions? Let us know!
        """)

if __name__ == "__main__":
    render_about_page()