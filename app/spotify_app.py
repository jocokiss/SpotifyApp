import json
import os

from typing import Any, Union, List

import numpy as np
import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials

from app.utilities import Tracks, Features, convert_to_numpy_array


class SpotifyApp:

    def __init__(self):
        load_dotenv('.env')  # Load the environment variables from the .env file
        self.__client_id = os.getenv('spotipy_client_id')
        self.__client_secret = os.getenv('spotipy_client_secret')

        self.__user_conn: spotipy.Spotify = self.__initialize_user_connection()
        self.__client_conn: spotipy.Spotify = self.__initialize_client_connection()

    def __initialize_user_connection(self) -> spotipy.Spotify:
        return spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=self.__client_id,
            client_secret=self.__client_secret,
            scope="user-library-read",
            redirect_uri="https://ca2f-89-135-32-37.ngrok-free.app"))

    def __initialize_client_connection(self) -> spotipy.Spotify:
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

    def search_tracks(self, query: str, limit: int = 50, market: str = 'US') -> list[Tracks]:
        """Search tracks using Spotify API.

        Args:
            query: the query of the search conditions. Format: "year:2022-2023 genre:popular ..."
            limit: offset. Defaults to the maximum: 50.
            market: the market of the search conditions. Defaults to 'US'.

        Returns: a list of tracks found.

        """
        results = self.__client_conn.search(q=query, type="track", limit=limit, market=market)
        tracks = Tracks.from_dict_list(results['tracks']['items'])

        while results['tracks']['next'] and len(tracks) < 200:
            results = self.__client_conn.next(results['tracks'])
            search_results = results["tracks"]["items"]

            if not search_results:
                print("No additional tracks found in this page.")
                break
            tracks.extend(Tracks.from_dict_list(search_results))

        track_df = Tracks.get_dataframe(tracks)
        track_df = track_df.drop_duplicates(subset='uri')

        # Convert the DataFrame back to a list of track objects
        unique_tracks = Tracks.from_dict_list(track_df.to_dict('records'))
        return unique_tracks

    def __get_genres_by_artists(self, tracklist: list[Tracks]) -> pd.DataFrame:
        artists = {artist['name'].strip().lower() for track in tracklist for artist in track.artists}
        df_dict = {}

        for artist in artists:
            query = f"artist:{artist}"
            try:
                search_results = self.__client_conn.search(q=query, type="artist", limit=50)["artists"]["items"]
                for result in search_results:
                    if result['name'].strip().lower() == artist:
                        df_dict[artist] = result["genres"]
                        break
            except Exception as e:
                print(f"Error fetching genres for artist {artist}: {e}")
                df_dict[artist] = []

        normalized_data = [(artist, genre) for artist, genres in df_dict.items() for genre in genres]
        df = pd.DataFrame(normalized_data, columns=['artist', 'genre'])
        return df.groupby('artist').agg({'genre': lambda x: list(x)}).reset_index()

    def update_tracks_with_genres(self, tracklist: list[Tracks]) -> pd.DataFrame:
        tracklist_df = Tracks.get_dataframe(tracklist)
        genre_df = self.__get_genres_by_artists(tracklist)

        tracklist_df['artists'] = tracklist_df['artists'].apply(lambda x: [artist.strip().lower() for artist in x])
        tracklist_exploded = tracklist_df.explode('artists')

        df_tracks_merged = pd.merge(tracklist_exploded, genre_df, left_on='artists', right_on='artist', how='left')
        df_tracks_merged['genre'] = df_tracks_merged['genre'].apply(lambda x: x if isinstance(x, list) else [])

        df_grouped = df_tracks_merged.groupby(['name', 'uri']).agg({
            'album': 'first',
            'explicit': 'first',
            'popularity': 'first',
            'id': 'first',
            'artists': lambda x: list(set(x)),
            'genre': lambda x: list(set([genre for sublist in x for genre in sublist]))
        }).reset_index(drop=False)

        return df_grouped

    def __get_audio_features_by_ids(self, track_ids: list[str]) -> pd.DataFrame:
        """Helper function to get audio features by track IDs in batches."""
        audio_features = []
        for i in range(0, len(track_ids), 100):
            batch_ids = track_ids[i:i + 100]
            batch_features = self.__client_conn.audio_features(batch_ids)
            audio_features.extend(batch_features)
        return Features.get_dataframe(Features.from_dict_list(audio_features))

    def get_audio_features(self, tracks: Union[list[Tracks], str, pd.DataFrame]) -> pd.DataFrame:
        """Get audio features from tracks.

        Args:
            tracks: Provide either a list of tracks, a path to a json file containing tracks, or a DataFrame of tracks.

        Returns: list of audio features.

        """
        if isinstance(tracks, str):
            with open(tracks, "r") as f:
                audio_features = json.load(f)
            return Features.get_dataframe(Features.from_dict_list(audio_features))

        if isinstance(tracks, pd.DataFrame):
            track_ids = tracks['id'].tolist()
        else:
            track_ids = [track.id for track in tracks]

        return self.__get_audio_features_by_ids(track_ids)

    def get_similar_results(self,
                            track_list: list[Tracks],
                            genre_filter: str,
                            limit: int = 50,
                            market: str = 'US') -> pd.DataFrame:

        track_df = self.update_tracks_with_genres(track_list)
        # Filter dataframe by genre
        filtered_df = track_df[
            track_df['genre'].apply(lambda genres: any(genre_filter.lower() in genre.lower() for genre in genres))]
        if filtered_df.empty:
            raise ValueError("No tracks found for the given genre filter.")

        # Get audio features for the filtered dataframe
        filtered_df_af = self.get_audio_features(filtered_df)
        filtered_af_array = convert_to_numpy_array(filtered_df_af)
        if filtered_af_array.size == 0:
            raise ValueError("Conversion to numpy array resulted in an empty array.")

        # Calculate the mean of the audio features array
        filtered_af_array_mean = np.mean(filtered_af_array, axis=0)

        # Search for tracks based on the genre filter
        genre_string = f"genre:{genre_filter}"
        queried_tracks = self.search_tracks(query=genre_string, limit=limit, market=market)
        queried_tracks_df = Tracks.get_dataframe(queried_tracks)

        # Get audio features for the queried tracks
        queried_tracks_af = self.get_audio_features(queried_tracks)
        queried_af_array = convert_to_numpy_array(queried_tracks_af)
        if queried_af_array.size == 0:
            raise ValueError("Conversion to numpy array resulted in an empty array for queried tracks.")

        # Calculate the distances between the queried tracks' audio features
        # and the mean of the filtered tracks' audio features
        distances = np.linalg.norm(queried_af_array - filtered_af_array_mean, axis=1)

        # Get the indices of the 5 tracks with the smallest distances
        closest_indices = np.argsort(distances)[:5]

        # Return the corresponding tracks
        results = queried_tracks_df.iloc[closest_indices]
        return results[['name', 'artists', 'album', 'explicit', 'uri']].rename(columns={'name': 'name.alias(song)'})



SP = SpotifyApp()

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)

favorite_tracks = SP.get_user_playlist()

favorite_results = SP.get_similar_results(favorite_tracks, "rock")
print(favorite_results)
