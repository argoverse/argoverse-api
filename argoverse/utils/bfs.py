# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""Breadth-First-Search of Graphs."""

from typing import Any, Collection, List, Mapping, MutableMapping, Sequence


def bfs_enumerate_paths(graph: Mapping[str, Sequence[str]], start: str, max_depth: int = 4) -> List[List[str]]:
    """Run Breadth-First-Search. Cycles are allowed and are accounted for.

    Find (u,v) edges in E of graph (V,E)

    Args:
       graph: Python dictionary representing an adjacency list
       start: key representing hash of start/source node in the graph search
       max_depth: maximum depth to traverse in graph from start node

    Returns:
      all_paths: list of graph paths
    """
    dists: MutableMapping[str, float] = {}

    # mark all vertices as not visited
    for k, neighbors in graph.items():
        dists[k] = float("inf")
        for v in neighbors:
            dists[v] = float("inf")

    dists[start] = 0
    paths: List[List[str]] = []
    # maintain a queue of paths
    queue: List[List[str]] = []
    # push the first path into the queue
    queue.append([start])
    while queue:  # len(q) > 0:
        # get the first path from the queue
        path: List[str] = queue.pop(0)
        # get the last node from the path
        u: str = path[-1]
        # max depth already exceeded, terminate
        if dists[u] >= max_depth:
            break
        # enumerate all adjacent nodes, construct a new path and push it into the queue
        for v in graph.get(u, []):
            if dists[v] == float("inf"):
                new_path: List[str] = list(path)
                new_path.append(v)
                queue.append(new_path)
                dists[v] = dists[u] + 1
                paths.append(new_path)

    return remove_duplicate_paths(paths)


def remove_duplicate_paths(paths: List[List[Any]]) -> List[List[str]]:
    """Remove duplicate subpaths from a set of paths.

    For example, if ``['1', '2', '6']`` and ``['1', '2']`` are
    included, remove the latter.

    Args:
       paths: Python list of lists, each element is a node key.

    Returns:
       Python list of lists, each element is a node key
    """
    joined_paths: List[str] = []
    # make keys for duplicate identification
    for path in paths:
        path = [str(node) for node in path]
        joined_paths += ["_".join(path)]

    indices_to_remove = set()

    # remove duplicate subpaths
    for i, joined_path in enumerate(joined_paths):
        for j, path_to_match in enumerate(joined_paths):
            if i == j:
                continue
            if path_to_match in joined_path:
                indices_to_remove.add(j)

    modified_paths = [path for i, path in enumerate(paths) if i not in indices_to_remove]
    return modified_paths
