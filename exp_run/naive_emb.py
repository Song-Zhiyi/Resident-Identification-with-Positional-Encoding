from src.caching import pickle_load_from_file

import os
from pathlib import Path

import numpy as np

def save_emb(file_name: os.PathLike, emb_array: np.ndarray):
    with open(file_name, "wb") as fp:
        np.savez_compressed(fp, emb_array)

def build_coord_emb(graph_file: os.PathLike, savedir: os.PathLike, force: bool=False):
    if Path(savedir).exists() and not force:
        return

    s = pickle_load_from_file(graph_file)
    sensor_coord = s['sensor_coord']
    sensor_name_to_id = s['sensor_name_to_id']
    l = []
    id_to_name = {
        v: k for k, v in sensor_name_to_id.items()
    }
    for i in range(len(sensor_name_to_id)):
        try:
            coord = sensor_coord[id_to_name[i]]
        except KeyError:
            coord = np.array([0,0,0])

        l.append(coord)

    emb = np.array(l, dtype=np.float32)

    for i, r in enumerate(emb):
        assert (r == sensor_coord[id_to_name[i]]).all()

    save_emb(savedir, emb)

SENSOR_ROOM_MAP = {
    "M47": 0, "M48": 0, "M46": 0, "M49": 0, "M45": 0, "M50": 0, "M41": 0, "D04": 0, "M44": 0,

    "M42": 1, "M43": 1, "M27": 1, "M28": 1, "M29": 1,

    "M37": 2, "M38": 2, "M39": 2, "M40": 2, "D05": 2,

    "M41": 3, "D06": 3,

    "M30": 4, "M31": 4, "M32": 4, "M33": 4, "M34": 4, "M35": 4, "M36": 4, "D03": 4,

    "M01": 5, "M02": 5, "M03": 5, "M04": 5, "M05": 5, "M06": 5, "M07": 5, "M08": 5,
    "M09": 5, "M10": 5, "M11": 5, "M12": 5, "M13": 5, "M14": 5, "M15": 5, "D02": 5,

    "M21": 6, "M22": 6, "M23": 6, "M24": 6, "M25": 6, "M26": 6, "D01": 6,

    "M19": 7,

    "M20": 8,

    "M51": 9, "D11": 9,

    "M16": 10, "M17": 10, "M18": 10, "D08": 10, "D09": 10, "D10": 10,

    "D12": 11,
}

def build_cluster_emb(graph_file: os.PathLike, savedir: os.PathLike, force: bool=False):
    if Path(savedir).exists() and not force:
        return

    s = pickle_load_from_file(graph_file)
    sensor_name_to_id = s['sensor_name_to_id']
    l = []
    id_to_name = {
        v: k for k, v in sensor_name_to_id.items()
    }

    unknown_id = len(SENSOR_ROOM_MAP)

    for i in range(len(sensor_name_to_id)):
        try:
            cluster_id = SENSOR_ROOM_MAP[id_to_name[i]]
        except KeyError:
            cluster_id = unknown_id

        l.append(cluster_id)

    emb = np.array(l, dtype=np.float32)

    n, = emb.shape
    emb = emb.reshape((n, 1))

    for i, r in enumerate(emb):
        assert (r == SENSOR_ROOM_MAP[id_to_name[i]]).all()

    save_emb(savedir, emb)

build_coord_emb(
    "preprocessed/graph/twor2009_sensor_graph.pkl",
    "preprocessed/emb/kyoto-layout1/coord",
)

build_cluster_emb(
    "preprocessed/graph/twor2009_sensor_graph.pkl",
    "preprocessed/emb/kyoto-layout1/cluster",
)
