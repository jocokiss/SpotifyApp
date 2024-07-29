"""Tests for `spotify_app` module."""
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from app.spotify_app import SpotifyApp


# pylint: disable=too-many-instance-attributes
class TestSpotifyApp(unittest.TestCase):
    """Tests for `spotify_app` module."""

    def setUp(self):
        patch.dict('os.environ', {
            'SPOTIPY_CLIENT_ID': 'test_client_id',
            'SPOTIPY_CLIENT_SECRET': 'test_client_secret'
        }).start()
        self.addCleanup(patch.stopall)

        self.spotify_oauth_patcher = patch('spotipy.SpotifyOAuth')
        self.spotify_client_credentials_patcher = patch('spotipy.SpotifyClientCredentials')
        self.spotify_patcher = patch('spotipy.Spotify')

        self.mock_spotify = self.spotify_patcher.start()
        self.mock_spotify_oauth = self.spotify_oauth_patcher.start()
        self.mock_spotify_client_credentials = self.spotify_client_credentials_patcher.start()

        self.addCleanup(self.spotify_patcher.stop)
        self.addCleanup(self.spotify_oauth_patcher.stop)
        self.addCleanup(self.spotify_client_credentials_patcher.stop)

        self.spotify_app = SpotifyApp()

        self.mock_tracks = [
            {'track': {
                'id': '1',
                'name': 'Track 1',
                'artists': ['Artist 1'],
                'album': 'Album 1',
                'explicit': False,
                'popularity': 50,
                'uri': 'uri1'}},
            {'track': {
                'id': '2',
                'name': 'Track 2',
                'artists': ['Artist 2'],
                'album': 'Album 2',
                'explicit': True,
                'popularity': 60,
                'uri': 'uri2'}}
        ]

        self.mock_genre_df = pd.DataFrame({
            'artist': ['artist 1', 'artist 2'],
            'genre': [['genre1', 'genre2'], ['genre3']]
        })

        self.mock_audio_features = [
            {'id': '1', 'danceability': 0.5, 'energy': 0.8, 'key': 5},
            {'id': '2', 'danceability': 0.7, 'energy': 0.6, 'key': 6}
        ]

    @patch('app.spotify_app.SpotifyApp.get_audio_features')
    def test_get_audio_features_mean(self, mock_get_audio_features):
        """Test get_audio_features_mean."""
        mock_df = pd.DataFrame(self.mock_tracks)
        mock_get_audio_features.return_value = pd.DataFrame(self.mock_audio_features)
        mean_result = self.spotify_app.get_audio_features_mean(mock_df)
        self.assertIsInstance(mean_result, np.ndarray)
        self.assertEqual(len(mean_result), 3)


if __name__ == '__main__':
    unittest.main()
