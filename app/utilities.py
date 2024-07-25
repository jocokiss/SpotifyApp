"""Utilities for spotify app."""
import numpy as np

from typing import List, Type, TypeVar
from pydantic import BaseModel
import pandas as pd

T = TypeVar('T', bound='BaseModelWithDataFrame')


class BaseModelWithDataFrame(BaseModel):
    """Base model with DataFrame conversion methods."""

    @staticmethod
    def get_dataframe(obj_list: List[BaseModel]) -> pd.DataFrame:
        dict_list = [obj.dict() for obj in obj_list]
        return pd.DataFrame(dict_list)

    @classmethod
    def from_dict_list(cls: Type[T], dict_list: List[dict]) -> List[T]:
        """Convert a list of dictionaries to a list of instances."""
        return [cls(**d) for d in dict_list]


class Features(BaseModelWithDataFrame):
    """Features model for checking datatypes."""
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


class Tracks(BaseModelWithDataFrame):
    """Tracks model for checking datatypes."""
    album: dict
    artists: list
    explicit: bool
    name: str
    uri: str
    popularity: int
    id: str
    genres: list = None

    @staticmethod
    def get_dataframe(tracklist: List['Tracks']) -> pd.DataFrame:
        dict_list = [
            {
                'album': track.album["name"],
                'artists': [artist['name'] for artist in track.artists],
                'explicit': track.explicit,
                'name': track.name,
                'uri': track.uri,
                'popularity': track.popularity,
                'id': track.id,
                'genres': track.genres if track.genres else []
            } for track in tracklist
        ]
        return pd.DataFrame(dict_list)


def convert_to_numpy_array(audio_features: pd.DataFrame) -> np.ndarray:
    """Convert a DataFrame of audio features to a numpy array.

    Args:
        audio_features: A DataFrame to convert to numpy array

    Returns: A numpy array of the audio features.

    """
    if not isinstance(audio_features, pd.DataFrame):
        raise ValueError("Input should be a pandas DataFrame")
    return audio_features.drop(columns='id', errors='ignore').values

