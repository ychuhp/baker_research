import numpy as np
import string

seqs = ['abcdef', 'abcdef']
msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
print(msa)

table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
print(table)
print(string.ascii_lowercase)



fr = open('./PAR.a3m')
lines = fr.readlines()
 
 
for l in lines:
    if not l.startswith('>'):
        l = l.translate( {ord(i):None for i in 'abcdefghijklmnopqrstuvwxyz'})
        print(len(l))
