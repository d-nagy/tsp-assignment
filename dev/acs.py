"""
Software Methodologies - AI Search Coursework.

Ant Colony System implementation to solve TSP.

Solution C
"""

from copy import deepcopy
import networkx as nx
import numpy as np
import os
from file_io import import_instance, export_tour


class Ant:
    def __init__(self, G, start, beta, q0, rho, tau0):
        self.cities = list(G.nodes())
        self.beta = beta
        self.q0 = q0
        self.rho = rho
        self.tau0 = tau0
        self.tour = [start]
        self.tour_length = 0

    def _get_unvisited_nodes(self):
        return [u for u in self.cities if u not in self.tour]

    def _calc_heuristic(self, G, u, v):
        return G[u][v]['phero'] * (G[u][v]['weight'] ** -self.beta)

    def _generate_p_distribution(self, graph, start, unvisited):
        p = []
        denominator = sum(
            [self._calc_heuristic(graph, start, u) for u in unvisited]
        )

        for u in unvisited:
            p.append(self._calc_heuristic(graph, start, u)/denominator)

        return p

    def _next_node(self, G):
        current_node = self.tour[-1]
        q = np.random.random()
        unvisited = _get_unvisited_nodes()

        if q <= self.q0:
            # Exploitation
            next_node = max(
                unvisited,
                key=lambda v: self._calc_heuristic(G, current_node, v)
            )
        else:
            # Biased exploration
            unvisited = self._get_unvisited_nodes()
            p_dist = self._generate_p_distribution(G, current_node, unvisited)

            next_node = np.random.choice(unvisited, 1, p=p_dist)[0]

        return next_node

    def _local_update(self, G):
        prev_node = self.tour[-2]
        new_node = self.tour[-1]

        tau = G[prev_node][new_node]['phero']
        pheromone = (1 - self.rho) * tau + self.rho * self.tau0

        G[prev_node][new_node]['phero'] = pheromone

        return G

    def one_step(self, G):
        current_node = self.tour[-1]
        next_node = self._next_node(G)
        self.tour.append(next_node)
        self.tour_length += G[current_node][next_node]['weight']
        G = self._local_update(G)
        return G


class AntColonyTSP:
    def __init__(self, G, colony_size=10, iterations=20,
                 alpha=0.1, beta=2, q0=0.9, rho=0.1):
        self.G = deepcopy(G)
        self.cities = list(self.G.nodes())
        self.colony_size = min([colony_size, len(self.cities)])
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.rho = rho

        self.local_tau0 = 1 / (len(cities) * self._nearest_neighbour_approx())
        self.ants = []
        self.best = None

        self._init_ants()

    def _init_ants(self):
        start_cities = np.random.choice(
            self.cities,
            colony_size,
            replace=False
        )

        for city in start_cities:
            ant = Ant(
                self.G,
                city,
                self.beta,
                self.q0,
                self.rho,
                self.local_tau0
            )
            self.ants.append(ant)

    def _init_pheromone(self):
        for u, v in self.G.edges():
            self.G[u][v]['phero'] = self.local_tau0

    def _nearest_neighbour_approx(self):
        start = np.random.choice(self.cities, 1)[0]
        tour = [start]
        n = len(self.cities)
        tour_length = 0

        for _ in range(n-1):
            current_node = tour[-1]
            next_node = min(
                [n for n in self.G[current_node] if n != current_node],
                key=lambda v: self.G[current_node][v]['weight']
            )
            tour.append(next_node)
            tour_length += self.G[current_node][next_node]['weight']

        tour_length += self.G[tour[-1]][start]['weight']
        return tour_length

    def _find_tours(self):
        for _ in range(len(self.cities)):
            for ant in self.ants:
                self.G = ant.one_step(self.G)

        iteration_best = min(
            [ant for ant in self.ants],
            key=lambda ant: ant.tour_length
        )

        self.best = min(
            [iteration_best, self.best],
            key=lambda ant: ant.tour_length
        )

    def _global_update(self):
        best_tour = self.best.tour
        best_tour_length = self.best.tour_length
        n = len(best_tour)

        best_tour_edges = [(best_tour[i], best_tour[(i+1) % n])
                           for i in range(n)]

        for u, v in self.G.edges():
            tau = self.G[u][v]['phero']
            tau0 = 1 / best_tour_length if (u, v) in best_tour_edges else 0
            pheromone = (1 - self.alpha) * tau + self.alpha * tau0
            self.G[u][v]['phero'] = pheromone

    def solve(self):
        self._init_pheromone()
        for _ in self.iterations:
            self._find_tours()
            self._global_update()
