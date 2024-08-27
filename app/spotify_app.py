"""Spotify App."""
import json
import os

from typing import Union

import numpy as np
import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials

from app.utilities import Tracks, Features, convert_to_numpy_array, strip_and_lower

pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


class SpotifyApp:
    """SpotifyApp class handling the Spotify API."""

    def __init__(self):
        load_dotenv('.env')  # Load the environment variables from the .env file
        self.__client_id = os.getenv('spotipy_client_id')
        self.__client_secret = os.getenv('spotipy_client_secret')
        self.__redirect_uri = os.getenv('redirect_uri')

        self.__user_conn: spotipy.Spotify = self.initialize_spotify_connection(user_auth=True)
        self.__client_conn: spotipy.Spotify = self.initialize_spotify_connection(user_auth=False)

    @property
    def user_conn(self):
        """Get user connection."""
        return self.__user_conn

    @user_conn.setter
    def user_conn(self, user_conn):
        """Set user connection."""
        self.__user_conn = user_conn

    @property
    def client_conn(self):
        """Get client connection."""
        return self.__client_conn

    @client_conn.setter
    def client_conn(self, client_conn):
        """Set client connection."""
        self.__client_conn = client_conn

    def initialize_spotify_connection(self, user_auth: bool = False) -> spotipy.Spotify:
        """
        Initialize the Spotify connection.

        Parameters:
        - user_auth (bool): If True, use user authentication (SpotifyOAuth).
                            If False, use client credentials (SpotifyClientCredentials).

        Returns:
        - spotipy.Spotify: An authenticated Spotify client.
        """
        if user_auth:
            return spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=self.__client_id,
                client_secret=self.__client_secret,
                scope="user-library-read",
                redirect_uri=self.__redirect_uri))
        client_credentials_manager = SpotifyClientCredentials(
            client_id=self.__client_id, client_secret=self.__client_secret)
        return spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    def get_user_playlist(self, genres: bool = False, max_iterations: int = 100) -> pd.DataFrame:
        """Get the user's playlist with a limit on the number of iterations to prevent infinite loops."""

        results = self.__user_conn.current_user_saved_tracks()
        tracks = results['items']
        iterations = 0

        while results['next'] and iterations < max_iterations:
            results = self.__user_conn.next(results)
            tracks.extend(results["items"])
            iterations += 1

        result_df = Tracks.get_dataframe([Tracks(**track['track']) for track in tracks])
        return self.update_tracks_with_genres(result_df) if genres else result_df

    def search_tracks(self, query: str, limit: int = 50, market: str = 'US') -> pd.DataFrame:
        """Search tracks using Spotify API.

        Args:
            query: the query of the search conditions. Format: "year:2022-2023 genre:popular ..."
            limit: offset. Defaults to the maximum: 50.
            market: the market of the search conditions. Defaults to 'US'.

        Returns: a dataframe of unique tracks found.

        Note: The API limits the maximum number of tracks that can be queried this way to 100.
                There’s no need to account for an infinite loop.

        """
        results = self.__client_conn.search(q=query, type="track", limit=limit, market=market)
        raw_tracks = results['tracks']['items']

        while results['tracks']['next'] and len(raw_tracks) < 200:
            results = self.__client_conn.next(results['tracks'])
            raw_tracks.extend(results["tracks"]["items"])

        track_df = Tracks.get_dataframe([Tracks(**track) for track in raw_tracks])
        return track_df.drop_duplicates(subset='id')

    def __get_genres_by_artists(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Get genres by artists.

        Args:
            dataframe: DataFrame of tracks.

        Returns: a DataFrame with genres per artist.

        """
        artists = {strip_and_lower(artist) for artist_list in dataframe['artists'] for artist in artist_list}
        df_dict = {}

        for artist in artists:
            query = f"artist:{artist}"
            search_results = self.__client_conn.search(q=query, type="artist", limit=50)["artists"]["items"]
            for result in search_results:
                if strip_and_lower(result['name']) == artist:
                    df_dict[artist] = result["genres"]
                    break
        normalized_data = [(artist, genre) for artist, genres in df_dict.items() for genre in genres]
        df = pd.DataFrame(normalized_data, columns=['artist', 'genre'])
        return df.groupby('artist').agg({'genre': list}).reset_index()

    def update_tracks_with_genres(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Update tracks with genres.

        Args:
            dataframe: DataFrame of tracks.

        Returns: a DataFrame with tracks with genres.

        """
        genre_df = self.__get_genres_by_artists(dataframe)

        # Normalize artist names in tracks DataFrame
        dataframe['artists'] = dataframe['artists'].apply(lambda x: [artist.strip().lower() for artist in x])
        tracklist_exploded = dataframe.explode('artists')
        # Merge tracklist with genres DataFrame
        df_tracks_merged = pd.merge(tracklist_exploded, genre_df, left_on='artists', right_on='artist', how='left')
        df_tracks_merged['genre'] = df_tracks_merged['genre'].apply(lambda x: x if isinstance(x, list) else [])

        # Group by track and aggregate genres
        df_grouped = df_tracks_merged.groupby(['name', 'uri']).agg({
            'album': 'first',
            'explicit': 'first',
            'popularity': 'first',
            'id': 'first',
            'artists': lambda x: list(set(x)),
            'genre': lambda x: list({genre for sublist in x for genre in sublist if genre})
        }).reset_index(drop=False)

        return df_grouped

    def __get_audio_features_by_ids(self, track_ids: list[str]) -> pd.DataFrame:
        """Helper function to get audio features by track IDs in batches.

        Args:
            track_ids: list of track IDs.

        Returns: a dataframe with audio features.

        """
        audio_features = []
        for i in range(0, len(track_ids), 100):
            batch_ids = track_ids[i:i + 100]
            batch_features = self.__client_conn.audio_features(batch_ids)
            audio_features.extend(batch_features)
        return Features.get_dataframe(Features.from_dict_list(audio_features))

    def get_audio_features(self, tracks: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Get audio features from tracks.

        Args:
            tracks: Provide either a path to a JSON file containing tracks or a DataFrame of tracks.

        Returns: a dataframe with audio features.

        """
        if isinstance(tracks, str):
            # Handling if tracks is a path to a JSON file
            with open(tracks, "r", encoding="utf-8") as f:
                audio_features = json.load(f)
            return Features.get_dataframe(Features.from_dict_list(audio_features))

        if isinstance(tracks, pd.DataFrame):
            # Handling if tracks is a DataFrame
            if 'id' not in tracks.columns:
                raise ValueError("DataFrame must contain an 'id' column with track IDs.")
            track_ids = tracks['id'].tolist()
            return self.__get_audio_features_by_ids(track_ids)
        raise TypeError("tracks must be either a path to a JSON file or a DataFrame of tracks.")

    def get_audio_features_mean(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Calculate the mean of the audio features array."""
        filtered_df_af = self.get_audio_features(dataframe)
        filtered_af_array = convert_to_numpy_array(filtered_df_af)
        return np.mean(filtered_af_array, axis=0)

    def get_similar_results(self,
                            genre_filter: str,
                            limit: int = 50,
                            market: str = 'US') -> pd.DataFrame:
        """Get similar tracks using Spotify API.

        Args:
            genre_filter: specify the genre for which you want the function to find similar tracks.
            limit: offset. Defaults to the maximum: 50.
            market: market of the search conditions. Defaults to 'US'.

        Returns: a dataframe with the 5 most similar tracks.

        """
        favorite_tracks = self.get_user_playlist(genres=True)
        # Filter dataframe by genre
        filtered_df = favorite_tracks[
            favorite_tracks['genre'].apply(lambda genres: any(
                genre_filter.lower() in genre.lower() for genre in genres))]
        if filtered_df.empty:
            raise ValueError(f"No tracks found for {genre_filter} genre filter.")

        filtered_mean = self.get_audio_features_mean(filtered_df)

        # Search for tracks based on the genre filter
        queried_df = self.search_tracks(query=f"genre:{genre_filter}", limit=limit, market=market)

        # Get audio features for the queried tracks
        queried_df_af = self.get_audio_features(queried_df)
        queried_af_array = convert_to_numpy_array(queried_df_af)

        # Calculate the distances between the queried tracks' audio features
        # and the mean of the filtered tracks' audio features
        distances = np.linalg.norm(queried_af_array - filtered_mean, axis=1)

        # Get the indices of the 5 tracks with the smallest distances
        closest_indices = np.argsort(distances)[:5]

        results = queried_df.iloc[closest_indices]

        return results[['name', 'artists', 'album', 'explicit', 'uri']].rename(columns={'name': 'song'})


if __name__ == '__main__':
    spotify_app = SpotifyApp()
    favorite_results = spotify_app.get_similar_results("hip-hop")
    print(favorite_results)
