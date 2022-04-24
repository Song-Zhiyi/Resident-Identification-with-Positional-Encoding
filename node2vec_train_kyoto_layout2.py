import os
for walks, walk_length, dim in [
    (100, 100, 128),
    (100, 100, 256),
    (100, 100, 64),
    (200, 200, 128),
    (300, 100, 256),
    (300, 500, 256),
    (700, 1000, 256),
    (700, 100, 64),
    (700, 500, 128),
    (700, 500, 64),
]:
    os.system(f"""python node2vec_train.py --target preprocessed/graph/kyoto-layout1-full-pruned-prob.pkl -J 8 --name kyoto-layout1 --dimensions {dim} --walks {walks} --walk-length {walk_length}
""")
