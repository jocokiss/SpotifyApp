import json
import os

from typing import Any

import numpy as np
import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from pydantic import BaseModel


class Features(BaseModel):
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: int
    time_signature: int
    id: str


class Tracks(BaseModel):
    album: dict
    artists: list
    explicit: bool
    name: str
    uri: str
    popularity: int
    id: str
    genres: list = None


class SpotifyApp(BaseModel):

    def __init__(self, /, **data: Any):
        super().__init__(**data)
        load_dotenv('.env')  # Load the environment variables from the .env file

        self.__client_id = os.getenv('spotipy_client_id')
        self.__client_secret = os.getenv('spotipy_client_secret')

        self.__user_conn: spotipy.Spotify = self.__initialize_user_connection()
        self.__client_conn: spotipy.Spotify = self.__initialize_client_connection()

    @staticmethod
    def get_dataframe_from_tracks(tracklist: list[Tracks]) -> pd.DataFrame:
        dict_list = [
            {
                'album': track.album["name"],
                'artists': [artist['name'] for artist in track.artists],
                'explicit': track.explicit,
                'name': track.name,
                'uri': track.uri,
                'popularity': track.popularity,
                'id': track.id,
                'genres': []
            } for track in tracklist
        ]
        return pd.DataFrame(dict_list)

    def __initialize_user_connection(self):
        return spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=self.__client_id,
            client_secret=self.__client_secret,
            scope="user-library-read",
            redirect_uri="https://ca2f-89-135-32-37.ngrok-free.app"))

    def __initialize_client_connection(self):
        client_credentials_manager = SpotifyClientCredentials(
            client_id=self.__client_id, client_secret=self.__client_secret)

        return spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    def get_user_playlist(self) -> list[Tracks]:
        results = self.__user_conn.current_user_saved_tracks()
        tracks = results['items']
        while results['next']:
            results = self.__user_conn.next(results)
            tracks.extend(results["items"])
        return [Tracks(**track['track']) for track in tracks]

    def get_genres(self, tracklist: list[Tracks]) -> pd.DataFrame:
        artists = {artist['name'] for track in tracklist for artist in track.artists}
        df_dict = {}
        for artist in artists:
            query = f"artists:{artist}"
            search_results = self.__client_conn.search(q=query, type="artist", limit=50)["artists"]["items"]
            for result in search_results:
                if result['name'] == artist:
                    df_dict.update({artist: result["genres"]})

        normalized_data = [(artist, genre) for artist, genres in df_dict.items() for genre in genres]
        df = pd.DataFrame(normalized_data, columns=['artist', 'genre'])
        return df.groupby('artist').agg({'genre': lambda x: list(x)}).reset_index()

    def search_tracks(self, query: str, limit: int = 50, market: str = 'US') -> list[Tracks]:
        """Search tracks using Spotify API.

        Args:
            query: the query of the search conditions. Format: "year:2022-2023 genre:popular ..."
            limit: offset. Defaults to the maximum: 50.
            market: the market of the search conditions. Defaults to 'US'.

        Returns: a list of tracks found.

        """
        results = self.__client_conn.search(q=query, type="track", limit=limit, market=market)
        tracks = [Tracks(**track) for track in results['tracks']['items']]

        while results['tracks']['next'] and len(tracks) < 100:
            results = self.__client_conn.next(results['tracks'])
            search_results = results["tracks"]["items"]

            if not search_results:
                print("No additional tracks found in this page.")
                break
            tracks.extend(Tracks(**track) for track in search_results)
        return tracks

    def get_audio_features(self, tracklist: list[Tracks] | str) -> list[Features]:
        """Get audio features from tracks.

        Args:
            tracks: Provide either a list of tracks or a path to a json file containing tracks.

        Returns: list of audio features.

        """
        audio_features = []
        if isinstance(tracklist, str):
            with open(tracklist, "r") as f:
                audio_features = json.load(f)
        else:
            for i in range(0, len(tracklist), 100):
                # Get the current batch of track IDs
                batch_ids = [track.id for track in tracklist[i:i + 100]]
                batch_features = self.client_conn.audio_features(batch_ids)
                audio_features.extend(batch_features)
        return [Features(**feature) for feature in audio_features]

    @staticmethod
    def convert_to_numpy_array(audio_features: list[Features]) -> np.ndarray:
        return np.array([[value for key, value in vars(feature).items() if key != 'id'] for feature in audio_features])


SP = SpotifyApp()
queried_tracks = SP.search_tracks(query="year:2022-2023 genre:pop", limit=50, market='US')

user_songs_df = SP.get_dataframe_from_tracks(queried_tracks)
genre_df = SP.get_genres(queried_tracks)

pd.set_option('display.max_columns', None)

# Now printing the dataframe will show all columns


df_tracks_exploded = user_songs_df.explode('artists')

df_tracks_merged = pd.merge(df_tracks_exploded, genre_df, left_on='artists', right_on='artist', how='left')

print(df_tracks_merged)

df_tracks_merged['genre'] = df_tracks_merged['genre'].apply(lambda x: x if isinstance(x, list) else [])


df_result = (
    df_tracks_merged
    .groupby(['name', 'album', 'explicit', 'uri', 'popularity', 'id'])
    .agg({
        'artists': lambda x: list(set(x)),  # Aggregating unique artists
        'genre': lambda x: list(set(genre for sublist in x for genre in sublist))  # Flattening and deduplicating genres
    })
    .reset_index()
)

# Filter rows where 'genre' is an empty list
empty_genres_df = df_result[df_result['genre'].apply(lambda x: x == [])]

# Count the number of such rows
num_empty_genres = len(empty_genres_df)

# liked_features = SP.get_audio_features("audio_features.json")

# X_liked = SP.convert_to_numpy_array(features)
# Y_liked = SP.convert_to_numpy_array(liked_features)
# average_features = np.mean(X_liked, axis=0)

# df = SP.get_user_playlist_dataframe
# # Normalize the nested JSON in the 'track' column
# track_df = pd.json_normalize(df['track'])
# 
# # Extract and rename the required columns
# track_df = track_df[['album.name', 'artists', 'explicit', 'name', 'id']]
# track_df.columns = ['album', 'artist_list', 'explicit', 'name', 'id']
# 
# # Since 'artist_list' is a list of dictionaries, extract the artist names
# track_df['artist'] = track_df['artist_list'].apply(lambda x: ', '.join([artist['name'] for artist in x]))
# 
# # Drop the original 'artist_list' column
# track_df = track_df.drop(columns=['artist_list'])
# 
# # Concatenate the 'added_at' column
# final_df = pd.concat([df[['added_at']], track_df], axis=1)
