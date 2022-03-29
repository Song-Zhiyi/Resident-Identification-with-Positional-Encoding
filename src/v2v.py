import pickle
import logging
import gzip
import os

from pathlib import Path
from functools import partial, cached_property
from collections import namedtuple

import numpy as np
from gensim.models import Word2Vec

from . import graph as cgp

V2VEmbeddingResult = namedtuple("V2VEmbeddingResult", "vv vname_vid sersor_coord")

logger = logging.getLogger("e.models.v2v")

def load_sensor_map_file(path):

    a = partial(np.array, dtype=np.float32)

    with open(path, "rb") as fp:
        dat = pickle.load(fp)
    g = cgp.CoordGraph()

    base = a(dat['base'])
    for k, v in dat['sensor_map'].items():
        g.add_coord_vertex(k.upper(), a(v) - base)

    g.complete_graph()

    walls = cgp.CoordGraph()
    for k, tup in dat['wall_map'].items():
        v_n1 = f"{k}1"
        v_n2 = f"{k}2"

        c1, c2 = tup

        walls.add_coord_vertex(v_n1, a(c1) - base)
        walls.add_coord_vertex(v_n2, a(c2) - base)
        walls.add_edge(v_n1, v_n2)

    cgp.complete_graph_prune(g, walls)
    return g

def random_work_data(g):
    mat = g.to_numpy()
    x, x = mat.shape
    if seqlen is None:
        seqlen = 2 * x
    if n_walks is None:
        n_walks = 4 * x
    list(cgp.graph_random_walk_gen(mat, seqlen, n_walks))

def v2v_embedding(train_data, n_feats=128, window_size=16):
    return Word2Vec(
        train_data,
        vector_size=n_feats,
        window=window_size,
    ).wv.vectors

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

def v2v_train(floors, add_edges):
    train_graph, sensor_coord = prepare_train_graph(floors, add_edges)

    mat = train_graph.to_numpy()
    x, x = mat.shape

    vv = v2v_embedding(list(cgp.graph_random_walk_gen(mat, 8 * x, n=512 * x)), window_size=x, n_feats=4*x)
    vv.setflags(write=False)
    vname_vid = train_graph.vname_vid

    sensor_coord = {
        vname_vid[name]: np.float32(value) for name, value in sensor_coord.items()
    }
    return V2VEmbeddingResult(vv, vname_vid, sensor_coord)

def get_v2v(train_kw, cache_path="cache/v2v.pkl"):
    try:
        logger.info(f"trying to load cache from {cache_path}")
        with open(cache_path, "rb") as fp:
            return pickle.load(fp)
    except Exception as e:
        logger.info(f"load cache from {cache_path} failed, rebuild v2v embedding")
    result = v2v_train(**train_kw)
    with open(cache_path, "wb") as fp:
        pickle.dump(result, fp)
        logger.info(f"v2v embedding wrote to {cache_path}")
    return result

class V2VTrainner:
    def __init__(self, get_train_graph, cache_dir: os.PathLike = "cache/v2v"):
        self.get_train_graph = get_train_graph
        self.cache_dir = Path(cache_dir).expanduser().absolute()
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    @cached_property
    def _train_graph(self):
        return self.get_train_graph()

    @cached_property
    def sensor_coord(self):
        _, sensor_coord = self._train_graph
        return sensor_coord

    @cached_property
    def g(self):
        g, _ = self._train_graph
        return g

    @cached_property
    def mat(self):
        return self.g.to_numpy()

    @cached_property
    def n_sensor(self):
        x, x = self.mat.shape
        return x

    def _do_v2v_train(self, seq_len, n_sample, window_size, n_feats):
        vv = v2v_embedding(
            list(cgp.graph_random_walk_gen(self.mat, seq_len, n=n_sample)),
            window_size=window_size,
            n_feats=n_feats
        )
        vv.setflags(write=False)
        vname_vid = self.g.vname_vid

        sensor_coord = {
            vname_vid[name]: np.float32(value) for name, value in self.sensor_coord.items()
        }
        return V2VEmbeddingResult(vv, vname_vid, sensor_coord)

    def get_v2v(self, seq_len: int, n_sample: int, window_size: int, n_feats, force: bool=False):
        cache_path = self.cache_dir / f"{seq_len}-{n_sample}-{window_size}-{n_feats}.pkl"
        logger.info(f"try load cache from {cache_path}")
        try:
            if not force:
                with gzip.open(cache_path, "rb") as fp:
                    return pickle.load(fp)
        except Exception as e:
            logger.warning(f"load cache from {cache_path} failed, rebuilding")

        result = self._do_v2v_train(seq_len, n_sample, window_size, n_feats)

        try:
            with gzip.open(cache_path, "wb", compresslevel=5) as fp:
                pickle.dump(result, fp)
        except Exception as e:
            logger.info(f"save cache to {cache_path} failed")

        return result

def save_v2v(fname, result: V2VEmbeddingResult):
    with gzip.open(fname, "wb", compresslevel=5) as fp:
        pickle.dump(result, fp)

def load_v2v(fname) -> V2VEmbeddingResult:
    with gzip.open(fname, "rb") as fp:
        return pickle.load(fp)

if __name__ == '__main__':
    import click

    def _main(floors, add_edges, n_features, output):
        add_edges = [e.strip().split(",") for e in add_edges]
        result = v2v_train(floors, add_edges)
        if output is not None:
            save_v2v(output, result)

    click_readable_file = click.Path(
        exists=True, readable=True, resolve_path=True, dir_okay=False)
    click_writable_file = click.Path(
        exists=False, writable=True, resolve_path=True, dir_okay=False)
    @click.command()
    @click.option('--n-features', type=int, default=128,
                  help="number of features of the final V2V embeddings")
    @click.option('--output', '-o', type=click_writable_file, default=None,
                  help="embedding file to write")
    @click.option('--add-edges', '-a', type=str, multiple=True,
                  help="additional edges to add, in comma seperated tuple of vertex name")
    @click.argument('floors', metavar='Floor description files',
                    type=click_readable_file, nargs=-1, required=True)
    def main(**kw):
        try:
            _main(**kw)
        except Exception as e:
            raise

    main()
