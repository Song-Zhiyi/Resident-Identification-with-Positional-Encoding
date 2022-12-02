# Positional encoding of non-intrusive sensors for resident identification in smart homes

Python dependencies:

- Python >= 3.9
- Scikit-learn
- Pytorch
- Numpy
- Pandas
- Networkx

*Important*: all code, script, python module should be only executed when you at the root of the project directory.
Use `python -m src.*` to run modules

## Data

Please put the unzipped CASAS data in the `data` folder

A copy of CASAS dataset is located in the `data/twor2009` folder.

## Preprocessed data

Some preprocessed data and intermediate data are located in preprocessed data folder, for caching purpose.

### Node2Vec embeddings could be build by:

`python node2vec_train.py --target preprocessed/graph/kyoto-layout1-full-pruned-prob.pkl`

Please refer to `python node2vec_train.py --help` for full list of options available for building Node2Vec embeddings.

## Experiments

Experiments are separated by individual setups, where we call it preset here.

You can run the experiments by:

`python twor2009_run.py --on-error stop --exp-name=test0 --repeat 10 <preset name>`

Use `python twor2009_run.py --help` for full list of options available for experiment runner.

Available preset names could be retrieved by run the command without any preset name.

## Result

All result will be located in `result` folder.

You can used code in `exp_loader.py` to load/manage experiments.

Also a jupyter notebook `vis.ipynb` is available for visualization used in the paper.
