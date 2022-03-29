# TO get sensor map:

run `python -m src.helpers.image_coord_helper data/twor.2009/sensorlayout.jpg`

TODO: how to use this prog

# TO get the pruned version of adjacent matrix of the graph representation

`python -m src.helpers.twor2009_graph`

result: a dict of following structure:

```{python}
{
    "adj_matrix": np.ndarray() # the matrix representing the graph.
    "sensor_coord": # coordination of sensors
}
```


Everything goes into .cache

Put data directories into data folder