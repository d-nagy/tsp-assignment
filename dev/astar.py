"""
Software Methodologies - AI Search Coursework.

A* Search implementation to solve TSP.

Solution A
"""

import heapq
import networkx as nx
import os
from file_io import import_instance, export_tour

OUTDIR = "test/TourfileA"


def search(filename):
    """
    Perform A* search to solve TSP for graph stored in a file.

    Args:
        filename: name of file containing TSP instance graph
    """
    G = import_instance(filename)
    tour = [0]
    name = os.path.splitext(filename)[0]

    root = (0, 0, tour)
    priority_q = [root]

    current = root

    while len(current[-1]) <= len(G.nodes()):
        neighbours = get_neighbours(current, G)
        for n in neighbours:
            heapq.heappush(priority_q, n)
        current = heapq.heappop(priority_q)

    print(current[-1])
    print('cost: ', current[1])
    export_tour(name, OUTDIR, tour, G)


def get_neighbours(node, G):
    """
    Get the neighbours of the current node in the A* search tree.

    Args:
        node: current node to get neighbours of
        G: the TSP instance graph

    Returns:
        List of neighbours of the given node.

    """
    f_n, path_cost, tour = node
    neighbours = []

    if len(tour) == len(G.nodes()):
        start_node = tour[0]
        new_tour = tour + [start_node]
        new_path_cost = path_cost + G[tour[-1]][start_node]['weight']
        return [(new_path_cost, new_path_cost, new_tour)]

    for city in G.nodes():
        if city not in tour:
            new_tour = tour + [city]
            new_path_cost = path_cost + G[tour[-1]][city]['weight']
            heuristic = h(
                new_tour,
                G,
                [i for i in G.nodes() if i not in new_tour]
            )
            new_node = (new_path_cost + heuristic, new_path_cost, new_tour)
            neighbours.append(new_node)
    return neighbours


def h(partial_tour, G, unvisited_cities):
    """
    Heuristic function for partial tour of graph G.

    Calculates the MST heuristic for the TSP.

    Args:
        partial_tour: a partial tour of G
        G: the TSP instance graph
        unvisited_cities: list of the unvisited_cities of G

    Returns:
        MST heuristic for partial_tour.

    """
    unvisited_subgraph = G.subgraph(unvisited_cities)
    mst = nx.minimum_spanning_tree(unvisited_subgraph, 'weight')
    mst_weight = mst.size('weight')

    current_node = partial_tour[-1]
    start_node = partial_tour[0]

    if len(partial_tour) == len(G.nodes()):
        return G[current_node][start_node]['weight']

    neighbours = [n for n in G.neighbors(current_node)
                  if n not in partial_tour]
    nearest_neighbour = min(
        neighbours,
        key=lambda n: G[current_node][n]['weight']
    )
    min_edge_next = G[current_node][nearest_neighbour]['weight']

    neighbours = [n for n in G.neighbors(start_node) if n not in partial_tour]
    nearest_neighbour = min(
        neighbours,
        key=lambda n: G[start_node][n]['weight']
    )
    min_edge_start = G[start_node][nearest_neighbour]['weight']

    return min_edge_next + mst_weight + min_edge_start


if __name__ == "__main__":
    search("NEWAISearchfile017.txt")  # Test example
    # for filename in os.listdir("test/city_files"):
    #    search(filename)
