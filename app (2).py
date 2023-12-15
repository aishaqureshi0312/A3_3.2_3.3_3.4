import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import shap
import random

# Load dataset
df = pd.read_csv("song_dataset.csv")

# Streamlit app structure
st.title("Song Recommendation App")

# Logic 1: Get recommendations based on user selection
st.header("Logic 1: User Selection")
user_ids = df['user'].unique()
user_id = st.selectbox('Select User ID:', user_ids, key='user_selection_1')

# Filter data for the selected user
user_df = df[df['user'] == user_id]

# Train model for the selected user
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(user_df[['user', 'song', 'play_count']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
model_logic_1 = SVD()
model_logic_1.fit(trainset)

submit_button_1 = st.button('Get Recommendations Logic 1')

if submit_button_1:
    not_listened_songs = list(set(df['song'].unique()) - set(user_df['song'].unique()))
    random.shuffle(not_listened_songs)  # Shuffle the order
    predictions = [(song, model_logic_1.predict(user_id, song).est) for song in not_listened_songs]
    predictions.sort(key=lambda x: x[1], reverse=True)
    n_recommendations = 5
    st.write(f"\nTop Recommendations for User {user_id} of songs not listened yet:")
    for song, rating in predictions[:n_recommendations]:
        title = df[df['song'] == song]['title'].iloc[0]
        st.write(f"{title} - {song} (Estimated Rating: {rating:.2f})")

# Logic 2: Get recommendations based on user and selected song
st.header("Logic 2: User and Song Selection")
user_id_logic_2 = st.selectbox('Select User ID:', user_ids, key='user_selection_2')
selected_song = st.selectbox('Select Listened Song:', df[df['user'] == user_id_logic_2]['song'].unique(), key='selected_song_2')

# Filter data for the selected user and song
user_song_df = df[(df['user'] == user_id_logic_2) & (df['song'] == selected_song)]

# Train model for the selected user and song
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(user_song_df[['user', 'song', 'play_count']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
model_logic_2 = SVD()
model_logic_2.fit(trainset)

submit_button_2 = st.button('Get Recommendations Logic 2')

if submit_button_2:
    not_listened_songs = list(set(df['song'].unique()) - set(user_song_df['song'].unique()))
    random.shuffle(not_listened_songs)  # Shuffle the order
    predictions = [(song, model_logic_2.predict(user_id_logic_2, song).est) for song in not_listened_songs]
    predictions.sort(key=lambda x: x[1], reverse=True)
    n_recommendations = 5
    st.write(f"\nTop Recommendations for User {user_id_logic_2} after selecting {selected_song} of songs not listened yet:")
    for song, rating in predictions[:n_recommendations]:
        title = df[df['song'] == song]['title'].iloc[0]
        st.write(f"{title} - {song} (Estimated Rating: {rating:.2f})")

# Create a background dataset
background_data = df.sample(100)  # Use a subset of your data as background
