# Positional encoding of non-intrusive sensors for resident identification in smart homes

*Important*: all code, script, python module should be only executed when you at the root of the project directory.
Use `python -m src.*` to run modules

## Data

Please put the unzipped CASAS data in the `data` folder

## Preprocessd data

Some data, for example, node embeddings, sensor graph and preprocess version of data are located in the proprocessed folder

### Node2Vec embeddings could be build by:

`python node2vec_train.py --target preprocessed/graph/kyoto-layout1-full-pruned-prob.pkl`

You can used `python node2vec_train.py --help` for full list of options

## Experiments

You can run the experiments by:

`python twor2009_run.py --on-error stop --exp-name=test0 --repeat 10 <preset name>`

Use `python twor2009_run.py --help` for full list of options

Available preset names could be retrieved by run the command without any preset name.

## Result

All result will be located in `result` folder.

You can used code in `exp_loader.py` to load/manage experiments.

Also a jupyter notebook `vis.ipynb` is available for visualization used in the thesis.
