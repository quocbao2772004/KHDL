import streamlit as st
import pandas as pd
import sys
import os
import time

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from recomendation.rcm_bert import recommend_by_title
except ImportError:
    st.error("Could not import recommendation module. Make sure 'recomendation/rcm_bert.py' exists.")
    st.stop()

import database as db

# Page config
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Initialize DB
if not os.path.exists(db.DB_NAME):
    db.init_db()

# Session state initialization
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

# --- Authentication Functions ---
def login():
    st.subheader("Login")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        user_id = db.verify_user(username, password)
        if user_id:
            st.session_state.user_id = user_id
            st.session_state.username = username
            st.success(f"Welcome back, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password")

def register():
    st.subheader("Register")
    username = st.text_input("Username", key="reg_user")
    password = st.text_input("Password", type="password", key="reg_pass")
    if st.button("Register"):
        if db.create_user(username, password):
            st.success("Registration successful! Please login.")
        else:
            st.error("Username already exists.")

def logout():
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.selected_movie = None
    st.rerun()

# --- Main App Logic ---

# Sidebar for Auth
with st.sidebar:
    if st.session_state.user_id:
        st.write(f"Logged in as: **{st.session_state.username}**")
        if st.button("Logout"):
            logout()
    else:
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            login()
        with tab2:
            register()

# Main Content
if st.session_state.user_id:
    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv("data/tmdb_cleaned.csv")
        return df

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Data file 'data/tmdb_cleaned.csv' not found.")
        st.stop()

    st.title("🎬 Movie Recommendation System")

    def select_movie(movie_title):
        st.session_state.selected_movie = movie_title
        # Log the view action
        db.log_action(st.session_state.user_id, "view_details", f"Viewed details for {movie_title}")

    # Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Browse Movies")
        
        # Search
        search_query = st.text_input("Search for a movie...", "")
        
        # Filter movies
        if search_query:
            filtered_df = df[df['title'].str.contains(search_query, case=False, na=False)]
            # Log search if it's a new query (simple debounce check could be added but skipping for simplicity)
            # We'll log it when they actually interact or maybe just log the search itself?
            # Let's log it here but maybe avoid spamming DB? 
            # For now, let's log only if they hit enter (which reruns the script)
            if search_query: 
                 # This runs on every keystroke if not careful, but streamlit text_input defaults to enter-to-submit
                 # unless on_change is used. Default is fine.
                 pass
        else:
            filtered_df = df.head(50) 

        # Display movies
        for index, row in filtered_df.iterrows():
            with st.container():
                c1, c2 = st.columns([1, 3])
                with c1:
                    poster_url = f"https://image.tmdb.org/t/p/w200{row['poster_path']}" if pd.notna(row['poster_path']) else "https://via.placeholder.com/100x150?text=No+Image"
                    st.image(poster_url, width=80)
                with c2:
                    st.write(f"**{row['title']}**")
                    if st.button("View Details", key=f"btn_{index}"):
                        select_movie(row['title'])
                st.divider()

    with col2:
        if st.session_state.selected_movie:
            movie_title = st.session_state.selected_movie
            movie_row = df[df['title'] == movie_title].iloc[0]
            
            st.header(movie_row['title'])
            
            # Movie Details
            c_img, c_info = st.columns([1, 2])
            with c_img:
                 poster_url = f"https://image.tmdb.org/t/p/w500{movie_row['poster_path']}" if pd.notna(movie_row['poster_path']) else "https://via.placeholder.com/300x450?text=No+Image"
                 st.image(poster_url, use_column_width=True)
            
            with c_info:
                st.write(f"**Release Date:** {movie_row['release_date']}")
                st.write(f"**Genres:** {movie_row['genres']}")
                st.write(f"**Rating:** {movie_row['vote_average']}/10")
                st.write("**Overview:**")
                st.write(movie_row['overview'])

            st.divider()
            st.subheader("Recommended Movies")
            
            try:
                recommendations = recommend_by_title(movie_title)
                
                # Log recommendation request
                db.log_action(st.session_state.user_id, "get_recommendations", f"Requested recommendations for {movie_title}")

                # Display recommendations
                rec_cols = st.columns(5)
                for i, (idx, rec_row) in enumerate(recommendations.iterrows()):
                    with rec_cols[i % 5]:
                        rec_poster = f"https://image.tmdb.org/t/p/w200{rec_row['poster_path']}" if pd.notna(rec_row['poster_path']) else "https://via.placeholder.com/150x225?text=No+Image"
                        st.image(rec_poster, use_column_width=True)
                        st.caption(rec_row['title'])
                        if st.button("View", key=f"rec_btn_{i}"):
                            select_movie(rec_row['title'])
                            st.rerun()
                            
            except Exception as e:
                st.error(f"Error getting recommendations: {e}")

        else:
            st.info("Select a movie from the list to see details and recommendations.")

else:
    st.info("Please Login or Register to access the Movie Recommendation System.")
    st.image("https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", use_column_width=True)
