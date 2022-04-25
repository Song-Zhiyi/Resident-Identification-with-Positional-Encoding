import src.data_preprocessing.twor2009
from src.caching import pickle_load_from_file

import pandas as pd

dat: pd.DataFrame = src.data_preprocessing.twor2009.get_data()
s: dict = pickle_load_from_file("preprocessed/graph/twor2009_sensor_graph.pkl")
dat = dat.merge(
    pd.DataFrame(
        s['sensor_name_to_id'].items(), columns=["sensor", "sensor_id"]
    )
)
dat = dat[dat['resident_id'] >= 0]
dat.sort_values(by="datetime", inplace=True)
X_names = ['weekday_sin', 'weekday_cos', 'dayofyear_sin', 'dayofyear_cos', 'secondofday_sin', 'secondofday_cos', 'read_bin', 'sensor_id']
y_name = 'resident_id'
dat = dat[X_names + [y_name]].dropna(axis=0)

del s