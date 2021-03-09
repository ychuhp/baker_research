# -*- coding: utf-8 -*-
__author__ = "Yechan Hong"
__maintainer__ = "Yechan Hong"
__email__ = "ychuh@pharmcadd.com"
__status__ = "Dev"

import cb_pdb_map_parser_ALL as cbpdb
from os import path
import math
import numpy as np
from PIL import Image
import ytil
from collections import defaultdict


def cbpdb_get_chains(pdb_file):
    ''' Retrieves the chain info from the pdb using cbpdb library
    pdb_file: pdb filepath
    return: dictionary of (chain name, chain) where chain is a object for input of distM
    '''
    x = cbpdb.parse_pdb(file_loc=pdb_file, pdb='', chain_dict=None, strict=False)
    x = cbpdb_fmt_parse(x)
    return x

def cbpdb_fmt_parse(parse):
    ''' Formats cbpdb parse
    parse:
    return:
    '''
    chains = {}
    for c in parse:
        key,c = cbpdb_fmt_chain(c)
        chains[key] = c
    return chains

def cbpdb_fmt_chain(chain):
    ''' Formats the cbpdb chain
    chain:
    return:
    '''
    i = {'key':1, 'type':6, 'x':7, 'y':8, 'z':9}
    fchain = []

    a = chain[0]
    key = a[1]

    for a in chain:
        if a[ i['key'] ]==key and a[ i['type'] ]=='CA' and a[ i['x'] ]!='nan': 
            fchain.append( a[-3:] )
    return key, np.array( fchain, dtype=float )


def distM(chain):
    ''' Creates a distance matrix
    chain: numpy chain of shape LxC. L length, C channels
    '''
    L = len(chain)
    M = np.stack(L*[chain], axis=1)
    sq_dist = ( M-M.transpose([1,0,2]) )**2
    return np.sqrt( sq_dist.sum(2) )

def ffindex(path):
    ''' Creates a ffindex dictionary
    path: ffindex filepath
    return: dictionary of (protein name, list of chain names)
    '''
    index = ytil.list__file(path)
    D = defaultdict(list)
    for l in index:
        pname, cname = l.split()[0].split('_')
        D[pname].append(cname)
    return D


# -------------------------------------------------------------------------------------------
# MAIN --------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

if __name__ == '__main__':

    save_dir = '/raid0/total_pdb/ca_matrix'
    FFINDEX = ffindex('pdb70_a3m.ffindex')
    pC = list( FFINDEX.items() )
    sidx = 2281

    for i,(p,C) in enumerate(pC[sidx:]):
        print(i+sidx,p,C)
        chains = cbpdb_get_chains('PDB/'+p+'.pdb')
        for c in C:
            chain = chains[c]
            M = distM(chain)

            im = Image.fromarray( M.astype(np.int8) )
            im.save( path.join(save_dir, p+'_'+c+'.png') )
