import networkx as nx
import pandas as pd
import src.data_preprocessing.twor2009
from src.caching import pickle_load_from_file, pickle_dump_to_file
from src.extractor import adj_mat_to_t_mat

dat = src.data_preprocessing.twor2009.get_data()
s = pickle_load_from_file("preprocessed/graph/twor2009_sensor_graph.pkl")
dat = dat.merge(
    pd.DataFrame(
        s['sensor_name_to_id'].items(), columns=["sensor", "sensor_id"]
    )
)
pickle_dump_to_file(
    "preprocessed/graph/kyoto-layout1-full-pruned-prob.pkl",
    dict(
        graph=nx.from_numpy_matrix(
            adj_mat_to_t_mat(s['adj_matrix'], None)
        )
    )
)