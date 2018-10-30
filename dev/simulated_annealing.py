"""
Software Methodologies - AI Search Coursework.

Simulated Annealing implementation to solve TSP.

Solution B
"""

from numpy import exp
import os
import random
from file_io import import_instance, export_tour

OUTDIR = "test/TourfileB"


def search(filename):
    """
    Perform Simulated Annealing to solve TSP for graph stored in a file.

    Args:
        filename: name of file containing TSP instance graph
    """
    G = import_instance(filename)
    tour = G.nodes()
    name = os.path.splitext(filename)[0]

    temp = 200000
    cooling_rate = 0.0005

    root = [energy(tour, G), tour]
    current = root
    best = root

    # Anneal
    while temp > 1:
        new_sol = swap_cities(current[1])
        new_energy = energy(new_sol, G)

        if accept(current[0], new_energy, temp):
            current = [new_energy, new_sol]

        if current[0] < best[0]:
            best = current

        temp *= 1 - cooling_rate

    # Quench
    for i in range(10000):
        new_sol = swap_cities(best[1])
        new_energy = energy(new_sol, G)

        if new_energy < best[0]:
            best = [new_energy, new_sol]

    print(best[1])
    print('cost: ', best[0])
    # export_tour(name, OUTDIR, tour, G)


def swap_cities(tour):
    """
    Swap two randomly selected cities in the tour.

    Args:
        tour: the tour to swap two cities in

    Returns:
        A copy of the original tour with two random cities swapped.

    """
    size = len(tour)

    city_one = random.randint(0, size-1)
    city_two = random.randint(0, size-1)
    while city_two == city_one:
        city_two = random.randint(0, size-1)

    new_tour = tour.copy()
    new_tour[city_one] = tour[city_two]
    new_tour[city_two] = tour[city_one]
    return new_tour


def energy(tour, G):
    """
    Calculate "energy" of a tour (the length).

    Args:
        tour: list of distinct cities that make up a tour in G
        G: the TSP instance graph

    Returns:
        Length of the tour in G.

    """
    size = len(G.nodes())
    return sum([
        G[tour[i]][tour[(i+1) % size]]['weight']
        for i in range(size)
    ])


def accept(current_energy, new_energy, temp):
    """
    Determine whether to accept a neighbour solution or not.

    Args:
        current_energy: tour length of current solution
        new_energy: tour length of neighbouring solution
        temp: current temperature

    Returns:
        True if the new solution should be accepted, otherwise False.

    """
    if new_energy > current_energy:
        return True

    return exp((current_energy - new_energy) / temp) > random.random()


if __name__ == "__main__":
    search("NEWAISearchfile017.txt")  # Test example
    # for filename in os.listdir("test/city_files"):
    #     search(filename)
