# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Forecasting Loader unit tests"""

import pathlib

import numpy as np
import pandas as pd
import pytest

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

TEST_DATA_LOC = pathlib.Path(__file__).parent.parent / "tests" / "test_data" / "forecasting"


@pytest.fixture  # type: ignore
def data_loader() -> ArgoverseForecastingLoader:
    return ArgoverseForecastingLoader(TEST_DATA_LOC)


def test_id_list(data_loader: ArgoverseForecastingLoader) -> None:
    track_id_gt = (
        pd.DataFrame(
            [
                "00000000-0000-0000-0000-000000000000",
                "00000000-0000-0000-0000-000000007735",
                "00000000-0000-0000-0000-000000008206",
            ]
        )
        .values.flatten()
        .tolist()
    )
    assert data_loader.track_id_list == track_id_gt


def test_city_name(data_loader: ArgoverseForecastingLoader) -> None:
    assert data_loader.city == "MIA"


def test_num_track(data_loader: ArgoverseForecastingLoader) -> None:
    assert data_loader.num_tracks == 3


def test_seq_df(data_loader: ArgoverseForecastingLoader) -> None:
    assert data_loader.seq_df is not None


def test_agent_traj(data_loader: ArgoverseForecastingLoader) -> None:
    traj_gt = [[10, 5], [10, 10]]
    assert np.array_equal(data_loader.agent_traj, traj_gt)


def test_get(data_loader: ArgoverseForecastingLoader) -> None:
    data_1 = data_loader.get("0")
    data_2 = data_loader[0]
    assert data_1.current_seq == data_2.current_seq
