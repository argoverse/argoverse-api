# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Sequence, Union

import numpy as np
import pandas as pd

__all__ = ["ArgoverseForecastingLoader"]


@lru_cache(128)
def _read_csv(path: Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """A caching CSV reader

    Args:
        path: Path to the csv file
        *args, **kwargs: optional arguments to be used while data loading

    Returns:
        pandas DataFrame containing the loaded csv
    """
    return pd.read_csv(path, *args, **kwargs)


class ArgoverseForecastingLoader:
    def __init__(self, root_dir: Union[str, Path]):
        """Initialization function for the class.

        Args:
            root_dir: Path to the folder having sequence csv files
        """
        self.counter: int = 0

        root_dir = Path(root_dir)
        self.seq_list: Sequence[Path] = [(root_dir / x).absolute() for x in os.listdir(root_dir)]

        self.current_seq: Path = self.seq_list[self.counter]

    @property
    def track_id_list(self) -> List[int]:
        """Get the track ids in the current sequence.

        Returns:
            list of track ids in the current sequence
        """
        _track_id_list: List[int] = np.unique(self.seq_df["TRACK_ID"].values).tolist()
        return _track_id_list

    @property
    def city(self) -> str:
        """Get the city name for the current sequence.

        Returns:
            city name, i.e., either 'PIT' or 'MIA'
        """
        _city: str = self.seq_df["CITY_NAME"].values[0]
        return _city

    @property
    def num_tracks(self) -> int:
        """Get the number of tracks in the current sequence.

        Returns:
            number of tracks in the current sequence
        """
        return len(self.track_id_list)

    @property
    def seq_df(self) -> pd.DataFrame:
        """Get the dataframe for the current sequence.

        Returns:
            pandas DataFrame for the current sequence
        """
        return _read_csv(self.current_seq)

    @property
    def agent_traj(self) -> np.ndarray:
        """Get the trajectory for the track of type 'AGENT' in the current sequence.

        Returns:
            numpy array of shape (seq_len x 2) for the agent trajectory
        """
        agent_x = self.seq_df[self.seq_df["OBJECT_TYPE"] == "AGENT"]["X"]
        agent_y = self.seq_df[self.seq_df["OBJECT_TYPE"] == "AGENT"]["Y"]
        agent_traj = np.column_stack((agent_x, agent_y))
        return agent_traj

    def __iter__(self) -> "ArgoverseForecastingLoader":
        """Iterator for enumerating over sequences in the root_dir specified.

        Returns:
            Data Loader object for the first sequence in the data
        """
        self.counter = 0
        return self

    def __next__(self) -> "ArgoverseForecastingLoader":
        """Get the Data Loader object for the next sequence in the data.

        Returns:
            Data Loader object for the next sequence in the data
        """
        if self.counter >= len(self):
            raise StopIteration
        else:
            self.current_seq = self.seq_list[self.counter]
            self.counter += 1
            return self

    def __len__(self) -> int:
        """Get the number of sequences in the data

        Returns:
            Number of sequences in the data
        """
        return len(self.seq_list)

    def __str__(self) -> str:
        """Decorator that returns a string storing some stats of the current sequence

        Returns:
            A string storing some stats of the current sequence
        """
        return f"""Seq : {self.current_seq}
        ----------------------
        || City: {self.city}
        || # Tracks: {len(self.track_id_list)}
        ----------------------"""

    def __getitem__(self, key: int) -> "ArgoverseForecastingLoader":
        """Get the DataLoader object for the sequence corresponding to the given index.

        Args:
            key: index of the element

        Returns:
            Data Loader object for the given index
        """

        self.counter = key
        self.current_seq = self.seq_list[self.counter]
        return self

    def get(self, seq_id: Union[Path, str]) -> "ArgoverseForecastingLoader":
        """Get the DataLoader object for the given sequence path.

        Args:
            seq_id: Fully qualified path to the sequence

        Returns:
            Data Loader object for the given sequence path
        """
        self.current_seq = Path(seq_id).absolute()
        return self
