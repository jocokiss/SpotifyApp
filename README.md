# Spotify App

This is a Python application that interacts with the Spotify API to manage user playlists, search for tracks, and analyze audio features. The application includes functionality to filter tracks by genre and find similar tracks based on audio features.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Configuration](#configuration)
- [Example](#example)

## Features
- **User Playlist Management**: Retrieve and manage user playlists.
- **Track Search**: Search for tracks using specific query conditions.
- **Genre Filtering**: Filter tracks based on specified genres.
- **Audio Feature Analysis**: Analyze and retrieve audio features for tracks.
- **Similarity Search**: Find tracks similar to a specified genre based on audio features.

## Installation
- Clone the repository:
  ```bash
  git clone https://github.com/jocokiss/SpotifyApp.git
  cd SpotifyApp
- Create and activate a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
- Install the required dependencies:
  ```bash
  pip install -r requirements.txt
- Set up environment variables:
  Create a `.env` file in the root directory and add your Spotify API credentials:
  ```ini
  spotipy_client_id=your_client_id
  spotipy_client_secret=your_client_secret
  ```

## Usage
- Run the application:
  ```bash
  python app/spotify_app.py
- Get similar tracks:
By default, the script retrieves similar tracks based on the “rock” genre.
Modify the get_similar_results function call to use a different genre or other parameters.

## Modules
- **spotify_app.py**:
  - Main application class handling interactions with the Spotify API.
  - Contains methods for retrieving playlists, searching tracks, filtering by genre, and analyzing audio features.

- **utilities.py**:
  - Utility classes and functions for managing data structures and conversions.
  - Includes `Tracks` and `Features` models with DataFrame conversion methods.

## Configuration
Ensure you have the following environment variables set in your `.env` file:
```ini
spotipy_client_id=your_client_id
spotipy_client_secret=your_client_secret
```

## Example
Here's an example of how to use the application to find similar tracks based on the "rock" genre:
```python
if __name__ == '__main__':
    spotify_app = SpotifyApp()
    favorite_results = spotify_app.get_similar_results("rock")
    print(favorite_results)