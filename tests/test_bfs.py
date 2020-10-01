#!/usr/bin/env python3

# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

from typing import List, Mapping, Sequence

from argoverse.utils.bfs import bfs_enumerate_paths

"""
Collection of unit tests to verify that Breadth-First search utility
on semantic lane graph works properly.
"""


def compare_paths(paths_lhs: List[List[str]], paths_rhs: List[List[str]]) -> bool:
    """
    Compare two input paths for equality.

        Args:
            paths_lhs: list of paths to compare against
            paths_rhs: other list of paths to compare against

        Returns:
            True if the paths are identical, false otherwise

    """
    paths_lhs.sort()
    paths_rhs.sort()

    return all(lhs == rhs for lhs, rhs in zip(paths_lhs, paths_rhs))


def get_sample_graph() -> Mapping[str, Sequence[str]]:
    """
        Args:
            None

        Returns:
            graph: Python dictionary representing an adjacency list
    """
    graph = {"1": ["2", "3", "4"], "2": ["5", "6"], "5": ["9", "10"], "4": ["7", "8"], "7": ["11", "12"]}
    return graph


def test_bfs_enumerate_paths_depth3() -> None:
    """Graph is in adjacent list representation."""
    graph = get_sample_graph()
    paths_ref_depth3: List[List[str]] = [
        ["1", "3"],
        ["1", "2", "6"],
        ["1", "4", "8"],
        ["1", "2", "5", "9"],
        ["1", "2", "5", "10"],
        ["1", "4", "7", "11"],
        ["1", "4", "7", "12"],
    ]
    paths = bfs_enumerate_paths(graph, start="1", max_depth=3)
    assert compare_paths(paths_ref_depth3, paths)


def test_bfs_enumerate_paths_depth2() -> None:
    """Graph is in adjacent list representation."""
    graph = get_sample_graph()
    paths_ref_depth2 = [["1", "3"], ["1", "2", "6"], ["1", "4", "8"], ["1", "2", "5"], ["1", "4", "7"]]
    paths = bfs_enumerate_paths(graph, "1", max_depth=2)
    assert compare_paths(paths_ref_depth2, paths)
