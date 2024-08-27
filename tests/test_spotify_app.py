"""Tests for `spotify_app` package."""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from app.spotify_app import SpotifyApp


class TestSpotifyApp(unittest.TestCase):
    """Tests for `spotify_app` package."""

    def setUp(self):
        # Patch the initialize_spotify_connection method and start the patches
        self.patcher_user_conn = patch('app.spotify_app.SpotifyApp.initialize_spotify_connection')
        self.patcher_client_conn = patch('app.spotify_app.SpotifyApp.initialize_spotify_connection')

        # Start the patches
        mock_initialize_spotify_connection_user = self.patcher_user_conn.start()
        mock_initialize_spotify_connection_client = self.patcher_client_conn.start()

        # Create mock objects
        mock_user_conn = MagicMock()
        mock_client_conn = MagicMock()

        # Set the return values of the mocked methods
        mock_initialize_spotify_connection_user.return_value = mock_user_conn
        mock_initialize_spotify_connection_client.return_value = mock_client_conn

        # Instantiate the SpotifyApp with the mocked connections
        self.spotify_app = SpotifyApp()
        self.spotify_app.user_conn = mock_user_conn
        self.spotify_app.client_conn = mock_client_conn

    @patch('app.spotify_app.Tracks.get_dataframe')
    def test_get_user_playlist(self, mock_get_dataframe):
        """Test get_user_playlist."""
        # Setup mock data with all required fields
        mock_results = {
            'items': [{
                'track': {
                    'name': 'Song 1',
                    'artists': [{'name': 'Artist 1'}],
                    'album': {'name': 'Album 1'},
                    'explicit': False,
                    'uri': 'spotify:track:1',
                    'popularity': 80,
                    'id': '1',
                }
            }],
            'next': None
        }
        self.spotify_app.user_conn.current_user_saved_tracks.return_value = mock_results
        mock_get_dataframe.return_value = pd.DataFrame([{
            'name': 'Song 1',
            'artists': ['Artist 1'],
            'album': 'Album 1',
            'explicit': False,
            'uri': 'spotify:track:1',
            'popularity': 80,
            'id': '1',
        }])

        # Run the method
        result_df = self.spotify_app.get_user_playlist()

        # Assertions
        self.spotify_app.user_conn.current_user_saved_tracks.assert_called_once()
        mock_get_dataframe.assert_called_once()
        self.assertFalse(result_df.empty)

    @patch('app.spotify_app.Tracks.get_dataframe')
    def test_search_tracks(self, mock_get_dataframe):
        """Test search_tracks."""
        # Setup mock data with all required fields
        mock_results = {
            'tracks': {
                'items': [{
                    'name': 'Song 1',
                    'artists': [{'name': 'Artist 1'}],
                    'album': {'name': 'Album 1'},
                    'explicit': False,
                    'uri': 'spotify:track:1',
                    'popularity': 80,
                    'id': '1',
                }],
                'next': None
            }
        }
        self.spotify_app.client_conn.search.return_value = mock_results
        mock_get_dataframe.return_value = pd.DataFrame([{
            'name': 'Song 1',
            'artists': ['Artist 1'],
            'album': 'Album 1',
            'explicit': False,
            'uri': 'spotify:track:1',
            'popularity': 80,
            'id': '1',
        }])

        # Run the method
        result_df = self.spotify_app.search_tracks("genre:pop")

        # Assertions
        self.spotify_app.client_conn.search.assert_called_once()
        mock_get_dataframe.assert_called_once()
        self.assertFalse(result_df.empty)

    @patch('app.spotify_app.SpotifyApp.get_audio_features')
    def test_get_audio_features_mean(self, mock_get_audio_features):
        """Test get_audio_features_mean."""
        # Setup mock data
        mock_df = pd.DataFrame(np.random.rand(5, 3), columns=['feature1', 'feature2', 'feature3'])
        mock_get_audio_features.return_value = mock_df

        # Run the method
        result_mean = self.spotify_app.get_audio_features_mean(mock_df)

        # Assertions
        mock_get_audio_features.assert_called_once()
        self.assertEqual(result_mean.shape[0], 3)

    @patch('app.spotify_app.SpotifyApp.get_user_playlist')
    @patch('app.spotify_app.SpotifyApp.get_audio_features_mean')
    @patch('app.spotify_app.SpotifyApp.get_audio_features')
    @patch('app.spotify_app.SpotifyApp.search_tracks')
    @patch('app.spotify_app.convert_to_numpy_array')
    def test_get_similar_results(self,  # pylint: disable=too-many-arguments
                                 mock_convert_to_numpy_array,
                                 mock_search_tracks,
                                 mock_get_audio_features,
                                 mock_get_audio_features_mean,
                                 mock_get_user_playlist):
        """Test get_similar_results."""
        # Setup mock data
        mock_playlist_df = pd.DataFrame([{
            'name': 'Song 1',
            'artists': ['Artist 1'],
            'genre': ['hip-hop']
        }])
        mock_queried_df = pd.DataFrame([{
            'name': 'Song 2',
            'artists': ['Artist 2'],
            'album': 'Album 2',
            'explicit': False,
            'uri': 'spotify:track:2',
            'id': '1',
        }])
        mock_audio_features = np.random.rand(1, 3)
        mock_mean = np.random.rand(3)

        mock_get_user_playlist.return_value = mock_playlist_df
        mock_get_audio_features_mean.return_value = mock_mean
        mock_search_tracks.return_value = mock_queried_df
        mock_get_audio_features.return_value = pd.DataFrame(mock_audio_features)
        mock_convert_to_numpy_array.side_effect = [mock_audio_features, mock_audio_features]

        # Run the method
        result_df = self.spotify_app.get_similar_results("hip-hop")

        # Assertions
        mock_get_user_playlist.assert_called_once()
        mock_search_tracks.assert_called_once()
        mock_get_audio_features.assert_called()
        mock_convert_to_numpy_array.assert_called()
        self.assertFalse(result_df.empty)
        self.assertEqual(result_df.shape[0], 1)
        self.assertIn('song', result_df.columns)
        self.assertIn('artists', result_df.columns)
        self.assertIn('album', result_df.columns)
        self.assertIn('explicit', result_df.columns)
        self.assertIn('uri', result_df.columns)


if __name__ == '__main__':
    unittest.main()
