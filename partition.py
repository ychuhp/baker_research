import os
import ytil
from os.path import join
from collections import defaultdict
loaddir = '/raid0/ychnh/seq'
L = os.listdir(loaddir)
L = [join(loaddir,l) for l in L]
#ytil.file__list( L, 'SEQS')
# TODO: Save partition to file
# TODO: Add error handling 

def partition(n,X):
    P = defaultdict(list)
    for i,x in enumerate(X):
        P[i%n].append(x)
    return P

S = 0
P = partition(10,L)
P
for k,v in P.items():
    print(k, len(v))
    S += len(v)
print(S)
import numpy as np
np.save('seq_partition', P)
