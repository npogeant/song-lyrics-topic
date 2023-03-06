import re

import numpy as np
import pandas as pd


def clean_lyrics(lyrics):
    # Remove terms starting with Translations
    lyrics = re.sub(r"Translations\S*", " ", lyrics)

    # Remove text between square brackets
    lyrics = re.sub(r"\[.*?\]", " ", lyrics)

    # Remove any numbers followed by 'Embed'
    lyrics = re.sub(r"\d+Embed", " ", lyrics)

    # Remove new line characters
    lyrics = lyrics.replace('\n', ' ')

    # Convert all text to lowercase
    lyrics = lyrics.lower()

    # Remove specific text if it's followed by 'you might also like'
    lyrics = re.sub(
        r"see(?!.*see.*you might also like).*you might also like", "", lyrics
    )

    # Return the cleaned lyrics
    return lyrics


# A list of genres to clean lyrics for
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

if __name__ == "__main__":
    # Loop over all genres
    for genre in genres:
        # Read the top tracks data for the current genre
        dataframe = pd.read_json(f'./data/{genre}_data.json', orient='top_tracks')

        # Normalize the data
        dataframe = pd.json_normalize(dataframe["top_tracks"])

        # Apply the clean_lyrics function to the 'lyrics' column
        dataframe['lyrics'] = dataframe['lyrics'].apply(
            lambda lyrics: clean_lyrics(lyrics)
        )

        # Save the cleaned lyrics to a CSV file
        dataframe[['lyrics']].to_csv(
            f'./data/{genre}_lyrics.csv', index=None, header=False
        )
