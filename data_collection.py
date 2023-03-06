import os
import json

import pandas as pd
import requests
import lyricsgenius
from tqdm import tqdm
from requests.exceptions import Timeout

GENIUS_ACCESS_TOKEN = os.environ["GENIUS_ACCESS_TOKEN"]
LASTFM_API_KEY = os.environ["LASTFM_API_KEY"]


# Method to get the top genres from Last.fm API
def get_top_genres():
    base_url = "http://ws.audioscrobbler.com/2.0/"

    # Make a request to the Last.fm API to get the top artists for the given genre
    params = {"method": "tag.getTopTags", "api_key": LASTFM_API_KEY, "format": "json"}
    response = requests.get(base_url, params=params)

    # Parse the response data
    tags = response.json()["toptags"]["tag"]
    dataframe = pd.DataFrame(tags)

    return dataframe


# Method to get the top artists by genre from Last.fm API
def get_top_artists_by_genre(genre):
    base_url = "http://ws.audioscrobbler.com/2.0/"

    # Make a request to the Last.fm API to get the top artists for the given genre
    params = {
        "method": "tag.getTopArtists",
        "tag": genre,
        "api_key": LASTFM_API_KEY,
        "format": "json",
    }
    response = requests.get(base_url, params=params)

    # Parse the response data
    data = json.loads(response.text)

    # Extract the artist names and their play counts from the response
    artists = response.json()["topartists"]["artist"]

    return [artist["name"] for artist in artists]


# Method to get the top tracks by genre from Last.fm API
def get_top_tracks_by_genre(genre, limit):
    base_url = "http://ws.audioscrobbler.com/2.0/"

    # Make a request to the Last.fm API to get the top artists for the given genre
    params = {
        "method": "tag.getTopTracks",
        "tag": genre,
        "limit": limit,
        "api_key": LASTFM_API_KEY,
        "format": "json",
    }
    response = requests.get(base_url, params=params)

    # Parse the response data
    data = json.loads(response.text)

    tracks_list = [
        {"name": track["name"], "artist": track["artist"]["name"]}
        for track in data["tracks"]["track"]
    ]

    dataframe = tracks_list

    return dataframe


# Method to get the top tracks by artists from Last.fm API
def get_top_tracks_by_artist(artist):
    base_url = "http://ws.audioscrobbler.com/2.0/"

    # Make a request to the Last.fm API to get the top artists for the given genre
    params = {
        "method": "artist.getTopTracks",
        "artist": artist,
        "api_key": LASTFM_API_KEY,
        "format": "json",
    }
    response = requests.get(base_url, params=params)

    # Parse the response data
    data = json.loads(response.text)
    dataframe = pd.DataFrame(data["toptracks"]["track"]).sort_values(
        by="playcount", ascending=True
    )

    return dataframe["name"].tolist()[:10]


# Define an empty dictionary called genre_data
genre_data = {}

# Define a list of music genres
genres = [
    'electronic',
    'rock',
    'Hip-Hop',
    'indie',
    'jazz',
    'reggae',
    'british',
    'punk',
    '80s',
    'acoustic',
    'rnb',
    'hardcore',
    'country',
    'blues',
    'alternative',
    'rap',
    'metal',
]

# Iterate over each genre in the list
for genre in genres:
    # Call the function "get_top_tracks_by_genre" and store the returned tracks in the "tracks" variable
    tracks = get_top_tracks_by_genre(genre, 1000)

    # Add a new key-value pair to the genre_data dictionary for the current genre, with the key "top_tracks" and the value being the tracks variable
    genre_data[genre] = {"top_tracks": tracks}

# Import the lyricsgenius package and initialize it with the Genius access token
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)

# Remove section headers from the lyrics to improve search results
genius.remove_section_headers

# Loop through each genre
for genre in tqdm(genres):
    # Loop through the top tracks of the current genre
    for track in tqdm(genre_data[genre]['top_tracks']):
        # Retry up to 3 times in case of a Timeout or AttributeError
        retries = 0
        while retries < 3:
            try:
                # Search for the lyrics of the current track using the Genius API
                track['lyrics'] = genius.search_song(
                    track['name'], track['artist']
                ).lyrics
            except Timeout as e:
                retries += 1
                track['lyrics'] = ''
                continue
            except AttributeError as e:
                retries += 1
                track['lyrics'] = ''
                continue
            break

    # Define the filename for the JSON file
    filename = f"{genre}_data.json"

    # Open the file for writing
    with open(filename, "w") as file:
        # Write the dictionary of genre data to the file in JSON format
        json.dump(genre_data[genre], file)

# Define the filename for the JSON file
filename = "genre_data.json"

# Open the file for writing
with open(filename, "w") as file:
    # Write the dictionary to the file in JSON format
    json.dump(genre_data, file)
