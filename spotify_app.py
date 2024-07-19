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
    id: str
    name: str
    uri: str

class SpotifyApp(BaseModel):

    def __init__(self, /, **data: Any):
        super().__init__(**data)
        load_dotenv('.env')  # Load the environment variables from the .env file

        self.__client_id = os.getenv('spotipy_client_id')
        self.__client_secret = os.getenv('spotipy_client_secret')

        self.__user_conn: spotipy.Spotify = self.__initialize_user_connection()
        self.__client_conn: spotipy.Spotify = self.__initialize_client_connection()

    @property
    def get_user_playlist_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.__get_user_playlist)

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

    def __get_user_playlist(self):
        results = self.__spotify_conn.current_user_saved_tracks()
        tracks = results['items']
        while results['next']:
            results = self.__user_conn.next(results)
            tracks.extend(results["items"])

        self.__user_playlist = tracks
        return [track['track']['id'] for track in tracks]

    def search_tracks(self, query: str, limit: int = 50, market: str = 'US') -> list:
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

        for track in tracks:
            print(track.name)
        return tracks

    def get_audio_features(self, tracks: list[Tracks] | str) -> list[Features]:
        """Get audio features from tracks.

        Args:
            tracks: Provide either a list of tracks or a path to a json file containing tracks.

        Returns: list of audio features.

        """
        audio_features = []
        if isinstance(tracks, str):
            with open(tracks, "r") as f:
                audio_features = json.load(f)
        else:
            for i in range(0, len(tracks), 100):
                # Get the current batch of track IDs
                batch_ids = [track.id for track in tracks[i:i + 100]]
                batch_features = self.__client_conn.audio_features(batch_ids)
                audio_features.extend(batch_features)
        return [Features(**feature) for feature in audio_features]

    @staticmethod
    def convert_to_numpy_array(audio_features: list[Features]) -> np.ndarray:
        return np.array([[value for key, value in vars(feature).items() if key != 'id'] for feature in audio_features])


SP = SpotifyApp()
tracks = SP.search_tracks(query="year:2022-2023 genre:pop", limit=50, market='US')
features = SP.get_audio_features(tracks)
liked_features = SP.get_audio_features("audio_features.json")

# X_liked = SP.convert_to_numpy_array(features)
# Y_liked = SP.convert_to_numpy_array(liked_features)
# average_features = np.mean(X_liked, axis=0)

df = SP.get_user_playlist_dataframe
print(df)
