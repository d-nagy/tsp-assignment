"""
Software Methodologies - AI Search Coursework.

Importing and exporting data.
"""

import networkx as nx
import os

INDIR = "test/city_files"


def import_instance(filename):
    """
    Import TSP instance from file and return networkx graph object.

    Args:
        filename: name of file containing TSP instance

    Returns:
        A networkx Graph object representing the TSP instance.

    """
    text = ""
    city = []

    with open(f'{INDIR}/{filename}.txt') as f:
        text = f.read()
        text = text.replace("\n", "").replace(" ", "")

    text = text.split(',')

    name = next(x for x in text if "NAME" in x.upper())
    name = name.split('=')[-1].strip()

    size = next(x for x in text if "SIZE" in x.upper())
    size = int(text[1].split('=')[-1].strip())

    distances = [int(''.join([d for d in i if d.isdigit()])) for i in text[2:]]

    j = 0
    for i in range(size-1, 0, -1):
        city.append(distances[j:i+j])
        j += i

    print(city)  # DEBUG

    # Make graph
    g = nx.complete_graph(size)

    for u in range(size):
        for v in range(size-u-1):
            g[u][u+v+1]['weight'] = city[u][v]

    return g


def export_tour(name, dir, tour, G):
    """
    Write TSP solution to a file.

    Args:
        filename: name of a TSP instance
        dir: name of directory to save tourfile to
        tour: a list of distinct nodes representing a tour in the TSP instance
        G: a networkx Graph representing the TSP instance
    """
    size = len(tour)
    length = sum(
        [
            G[tour[i]][tour[(i+1) % size]]['weight']
            for i in range(size)
        ]
    )

    tour_str = ','.join(map(lambda x: str(x+1), tour))

    if not os.path.exists(dir):
        print(f'Directory "{dir}" does not exist.')
        return

    with open(f'{dir}/tour{name}.txt', 'w') as f:
        f.writelines(
            [
                f'NAME = {name},\n',
                f'TOURSIZE = {size},\n',
                f'LENGTH = {length},\n',
                tour_str
            ]
        )


if __name__ == '__main__':
    # Test
    INDIR = "sample_files"
    dir = "sample_files"
    filename = "AISearchtestcase"
    G = import_instance(filename)
    export_tour(filename, dir, [i for i in range(8)], G)
