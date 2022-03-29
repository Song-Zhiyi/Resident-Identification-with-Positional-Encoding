import pickle
import gzip
import os

from pathlib import Path
from functools import partial, cached_property

import numpy as np

import src.graph as graph

def load_sensor_map_file(path):

    a = partial(np.array, dtype=np.float32)

    with open(path, "rb") as fp:
        dat = pickle.load(fp)
    g = graph.CoordGraph()

    base = a(dat['base'])
    for k, v in dat['sensor_map'].items():
        g.add_coord_vertex(k.upper(), a(v) - base)

    g.complete_graph()

    walls = graph.CoordGraph()
    for k, tup in dat['wall_map'].items():
        v_n1 = f"{k}1"
        v_n2 = f"{k}2"

        c1, c2 = tup

        walls.add_coord_vertex(v_n1, a(c1) - base)
        walls.add_coord_vertex(v_n2, a(c2) - base)
        walls.add_edge(v_n1, v_n2)

    graph.complete_graph_prune(g, walls)
    return g

def graph_to_coord(g, floor):
    return {
        v.cname: np.concatenate((v.coord, [floor])) for v in g.vertices.values()
    }

def prepare_train_graph(floors, add_edges):
    base = load_sensor_map_file(floors[0])
    floors = [load_sensor_map_file(fname) for fname in floors]
    sensor_coord = graph_to_coord(base, 1)
    for n_floor, g in enumerate(floors[1:]):
        base.do_update(g)
        sensor_coord.update(graph_to_coord(g, n_floor + 2))

    for edge_pair in add_edges:
        base.add_edge(*edge_pair)

    return base, sensor_coord


if __name__ == '__main__':
    base, sensor_coord = prepare_train_graph([
        ".cache/graph/floor1.pkl",
        ".cache/graph/floor2.pkl",
    ], [("M26", "M27")])

    with open(".cache/graph/twor_sensor_graph.pkl", "wb") as fp:
        pickle.dump({
            "adj_matrix": base.to_numpy(),
            "sensor_coord": sensor_coord
        }, fp)
