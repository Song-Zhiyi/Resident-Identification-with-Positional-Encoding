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


import click


@click.command()
@click.argument(
    "src_file", nargs=1, required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path))
@click.argument(
    "tgt_file", nargs=1, required=True,
    type=click.Path(exists=False, dir_okay=False, writable=True, path_type=Path)
)
def main(src_file: str, tgt_file: str):
    from src.caching import pickle_dump_to_file
    base, sensor_coord = prepare_train_graph(
        [src_file,], []
    )
    base = load_sensor_map_file(src_file)
    adj_matrix = base.to_numpy()
    sensor_name_to_id = base.vname_vid

    import networkx as nx
    from src.extractor import adj_mat_to_t_mat

    graph = nx.from_numpy_matrix(
        adj_mat_to_t_mat(adj_matrix, None)
    )

    pickle_dump_to_file(
        tgt_file, {
            "graph": graph,
            "adj_matrix": adj_matrix,
            "sensor_coord": sensor_coord,
            "sensor_name_to_id": sensor_name_to_id,
        }, False
    )


if __name__ == '__main__':
    main()
