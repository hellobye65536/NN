import gzip

import numpy as np
import datasets

mnist = datasets.load_dataset('mnist')

with gzip.open('train.csv', 'wt') as f:
    for v in mnist['train']:
        print(v['label'], end=',', file=f)
        print(*np.array(v['image']).flatten(), sep=',', file=f)

with gzip.open('test.csv', 'wt') as f:
    for v in mnist['test']:
        print(v['label'], end=',', file=f)
        print(*np.array(v['image']).flatten(), sep=',', file=f)

