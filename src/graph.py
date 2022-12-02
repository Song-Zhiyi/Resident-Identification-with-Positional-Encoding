from __future__ import annotations

import os
import itertools
import logging
import pickle
import typing

from functools import cached_property, partial
from pprint import pprint

import numpy as np
import numpy.typing as npt
import networkx as nx

from scipy.sparse import csr_matrix

_CoordType = npt.NDArray[np.float_]

logger = logging.getLogger("preprocessing.complete_graph_prune")

def serial_gen(start: int = 0) -> typing.Generator[int]:
    while True:
        yield start
        start += 1

class CoordVertex(object):
    def __init__(self, cname: str, coord: _CoordType):
        self.cname = cname
        self.coord = coord

    def __repr__(self):
        return f"V{self.coord}"

def vec_length(vec: _CoordType) -> float:
    r"""Calculate the length of the vector

    .. :math::
        \text{length} = \sqrt{(\sum_{i}x_i^2)}}

    :param vec: the vector to be calculated
    :returns: the length
    """
    return np.sqrt(np.sum(vec ** 2))

class Edge:
    def __init__(self, v1: CoordVertex, v2: CoordVertex):
        self.v1 = v1
        self.v2 = v2
        self.vector = v1.coord - v2.coord

    def __eq__(self, __o: object):
        if not isinstance(__o, Edge):
            return NotImplemented
        o: Edge = __o
        return (
            self.v1 is o.v1 and
            self.v2 is o.v2 and
            (self.vector == o.vector).all()
        )

    @cached_property
    def unit_vector(self):
        raise NotImplemented
        return None

    @cached_property
    def length(self) -> float:
        return vec_length(self.vector)

    @cached_property
    def angle(self) -> float:
        stdvec = np.zeros(self.vector.shape, dtype=self.vector.dtype)
        stdvec[0] = 1

        return np.arccos(np.sum(stdvec * self.vector) / self.length)

    def __hash__(self):
        return hash(self.v1) + hash(self.v2) + hash(self.length) + hash(self.angle)

    def __repr__(self):
        return f"E({self.v1.cname} - {self.vector} - {self.v2.cname})"

class CoordGraph:
    def __init__(self):
        self.vertices: dict[str, CoordVertex] = {}
        self.vname_vid: dict[str, int] = {}
        self.edges: dict[tuple[CoordVertex, float], Edge] = {}
        self.vid_gen = serial_gen()

    def do_update(self, o: CoordGraph):
        new_vidbase = next(self.vid_gen)
        self.vid_gen = serial_gen(new_vidbase + next(o.vid_gen) - 1)
        for vname, vid in o.vname_vid.items():
            self.vname_vid[vname] = vid + new_vidbase
        self.vertices.update(o.vertices)
        self.edges.update(o.edges)

    def add_coord_vertex(self, cname: str, coord: _CoordType) -> CoordVertex:
        vid = next(self.vid_gen)
        self.vname_vid[cname] = vid
        v = CoordVertex(cname, coord)
        self.vertices[cname] = v
        return v

    def add_edge(self, v1_cname: str, v2_cname: str):
        v1 = self.vertices[v1_cname]
        v2 = self.vertices[v2_cname]
        self._do_add_edge(v1, v2)

    def remove_edge(self, e: Edge):
        v1 = e.v1
        v2 = e.v2
        v1vid = self.vname_vid[v1.cname]
        v2vid = self.vname_vid[v2.cname]

        if v1vid > v2vid:
            v1, v2 = v2, v1

        key = (v1, e.angle)
        del self.edges[key]

    def _do_add_edge(self, v1: CoordVertex, v2: CoordVertex):
        if v1 is v2:
            return
        v1vid = self.vname_vid[v1.cname]
        v2vid = self.vname_vid[v2.cname]

        if v1vid > v2vid:
            v1, v2 = v2, v1

        e = Edge(v1, v2)
        key = (v1, e.angle)
        try:
            same_angle = self.edges[key]
            if same_angle.length > e.length:
                logger.info(f"edge {same_angle} dropped")
                self.edges[key] = e
        except KeyError:
            self.edges[key] = e

    def print_stat(self):
        pprint(self.vertices)
        pprint(self.edges)

    def __repr__(self):
        return f"{self.vertices!s} {self.edges!s}"

    def complete_graph(self):
        for v1, v2 in itertools.permutations(self.vertices.values(), 2):
            self._do_add_edge(v1, v2)

    def to_numpy(self):
        n_vertices = len(self.vertices)
        mat = np.zeros((n_vertices, n_vertices), dtype=np.float32)

        for e in self.edges.values():
            vid1 = self.vname_vid[e.v1.cname]
            vid2 = self.vname_vid[e.v2.cname]
            length = e.length

            mat[vid1, vid2] = length
            mat[vid2, vid1] = length

        return mat

    def to_csr_matrix(self):
        return csr_matrix(self.to_numpy())

    def to_networkx(self):
        return nx.from_scipy_sparse_matrix(self.to_csr_matrix())

def numpy_adjmatrix_to_csr_matrix(mat: np.ndarray):
    return csr_matrix(mat)

def numpy_adjmatrix_to_networkx(mat: np.ndarray):
    return nx.from_scipy_sparse_matrix(numpy_adjmatrix_to_csr_matrix(mat))

def on_segment(p1: _CoordType, p2: _CoordType, p: _CoordType) -> bool:
    m = np.concatenate((p1.reshape((1, -1)), p2.reshape((1, -1))), 0)
    return all(
        np.min(dim) <= x <= np.max(dim) for dim, x in zip(m.T, p)
    )

def edge_collision(e1: Edge, e2: Edge) -> bool:
    p1 = e1.v1.coord
    p2 = e1.v2.coord
    p3 = e2.v1.coord
    p4 = e2.v2.coord
    return lineseg_collsion((p1, p2), (p3, p4))


def lineseg_collsion(seg1: tuple[_CoordType, _CoordType],
                     seg2: tuple[_CoordType, _CoordType]) -> bool:
    p1, p2 = seg1
    p3, p4 = seg2

    d1 = np.cross(p1 - p3, p4 - p3)
    d2 = np.cross(p2 - p3, p4 - p3)
    d3 = np.cross(p3 - p1, p2 - p1)
    d4 = np.cross(p4 - p1, p2 - p1)

    if ((d1 * d2 < 0) and (d3 * d4 < 0)):
        return True

    elif d1 == 0 and on_segment(p3, p4, p1):
        return True
    elif d2 == 0 and on_segment(p3, p4, p2):
        return True
    elif d3 == 0 and on_segment(p1, p2, p3):
        return True
    elif d4 == 0 and on_segment(p1, p2, p4):
        return True
    else:
        return False


def complete_graph_prune(complete_graph: CoordGraph, walls: CoordGraph):
    to_prune = complete_graph.edges
    walls = list(walls.edges.values())
    keys = list(to_prune.keys())

    for k in keys:
        try:
            e = to_prune[k]
            if any(edge_collision(e, wall) for wall in walls):
                del to_prune[k]
        except KeyError:
            # has been removed
            continue


def graph_random_walk_gen(mat: npt.NDArray[np.float_],
                          seq_len: int, n: int = 100) -> npt.NDArray[np.float_]:
    """generate random walk sequence from graph with adjacent matrix representation

    :param mat: the adjacent matrix representing the graph
    :type mat: npt.NDArray[np.float_]
    :param seq_len: length of the output sequence
    :type seq_len: int
    :param n: how many sequence to generate, defaults to 100
    :type n: int, optional
    :yield: array with shape (seq_len, 1)
    :rtype: npt.NDArray[np.float_]
    """
    x, x = mat.shape
    np_choice = np.random.choice
    starts = list(range(x))
    vm = []
    wm = []
    for r in mat:
        print(r)
        sels = (r > 0)
        w = r[sels]
        w /= w.sum()
        v = np.argwhere(sels).reshape(-1)
        vm.append(v)
        wm.append(w)

    for _ in range(n):
        now = np_choice(starts)
        yield [now := np_choice(vm[now], p=wm[now]) for i in range(seq_len)]


from typing import TypedDict

class SensorMapDict(TypedDict):
    sensor_map: dict[str, tuple[float, float]]
    wall_map: dict[str, tuple[tuple[float, float], tuple[float, float]]]

def build_pruned_graph(sensor_map: SensorMapDict):
    a = partial(np.array, dtype=np.float32)
    g = CoordGraph()

    base = a(sensor_map['base'])
    for k, v in sensor_map['sensor_map'].items():
        g.add_coord_vertex(k.upper(), a(v) - base)

    g.complete_graph()

    walls = CoordGraph()
    for k, tup in sensor_map['wall_map'].items():
        v_n1 = f"{k}-1"
        v_n2 = f"{k}-2"

        c1, c2 = tup

        walls.add_coord_vertex(v_n1, a(c1) - base)
        walls.add_coord_vertex(v_n2, a(c2) - base)
        walls.add_edge(v_n1, v_n2)

    complete_graph_prune(g, walls)
    return g

def load_sensor_map_file(path: os.PathLike) -> SensorMapDict:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def graph_to_coord(g: CoordGraph, floor: int):
    return {
        v.cname: np.concatenate((v.coord, [floor])) for v in g.vertices.values()
    }

def compose_floors(floors: list[CoordGraph], add_edges: list[tuple[str, str]]):
    base = floors[0]
    sensor_coord = graph_to_coord(base, 1)
    for n_floor, g in enumerate(floors[1:]):
        base.do_update(g)
        sensor_coord.update(graph_to_coord(g, n_floor + 2))

    for edge_pair in add_edges:
        base.add_edge(*edge_pair)

    return base, sensor_coord


