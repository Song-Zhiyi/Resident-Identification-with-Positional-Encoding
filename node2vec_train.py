import sys
import click
from pathlib import Path

@click.command()
@click.option("--jobs", "-J", help="number of workers", type=int, default=1)
@click.option("--save-dir", help="The directory for result to be save",
              type=click.Path(dir_okay=True, path_type=Path), default="preprocessed")
@click.option("--name", help="name for embeddings", default="UNKNOWN", type=str)
@click.option("--walks", type=int, required=True, default=700, help="Number of random walks", show_default=True)
@click.option("--walk-length", type=int, required=True, default=1000, help="The length of each walk (sequence length)", show_default=True)
@click.option("--dimensions", type=int, required=True, default=256, help="The dimensions of the outcome embeddings", show_default=True)
@click.option("--window-size", type=int, required=True, default=5, help="The window_size of underlying Word2Vec model", show_default=True)
@click.option("--rebuild", is_flag=True, default=False)
@click.option("--target",
              type=click.Path(exists=True, readable=True, resolve_path=True, path_type=Path),
              help="path to a pickle file containing only dict with entry 'graph' being a networkx graph",
              required=True)
def entry(jobs, save_dir: Path, walks, walk_length, dimensions, rebuild: bool, target: Path, name: str, window_size: int):
    save_dir = save_dir / "emb" / name
    save_dir.mkdir(exist_ok=True, parents=True)
    file_name = save_dir / f"node2vec-{walks}-{walk_length}-{dimensions}-{window_size}"
    if file_name.exists() and not rebuild:
        sys.exit(f"{file_name!s} already exists, skipping. pass --rebuild to force rebuild")
    import src.node2vec as node2vec
    from src.caching import pickle_load_from_file

    g = pickle_load_from_file(target)['graph']
    model = node2vec.Node2Vec(g, num_walks=walks, walk_length=walk_length,
                              workers=jobs, dimensions=dimensions)

    import pickle
    import gzip

    with gzip.open("test", "wb", 9) as fp:
        pickle.dump(model, fp)

    emb = model.fit(window=window_size)

    import numpy as np
    with open(file_name, "wb") as fp:
        np.savez_compressed(fp, emb.wv.vectors)

if __name__ == '__main__':
    entry()