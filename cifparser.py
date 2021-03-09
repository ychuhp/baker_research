from collections import defaultdict
import numpy as np


DICT_R2RES = {
    'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D','CYS':'C', 'GLN':'Q', 'GLU':'E', 'GLY':'G',
    'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K','MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S',
    'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V','ASX':'N', 'GLX':'Q', 'UNK':'G', 'HSD':'H',
    }


def process_raw_line(l):
    # Courtesy of Manuel Alessandro Collazo
    l = l.split()
    chain_id = l[18], 
    idx, RES, atom = l[8], l[5], l[3]

    res_idx = l.index(RES,5+1)
    chain_idx = res_idx+1
    chain_id = l[chain_idx]

    x,y,z = float(l[10]), float(l[11]), float(l[12])
    return chain_id, idx, RES, atom, x,y,z

def cb(ca,c,n):
    ca,c,n = np.array(ca), np.array(c), np.array(n)
    b,c = ca-n, c-ca
    a = np.cross(b,c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca

def isatomline(l):
    if not l.startswith('ATOM '):
        return False
    l = l.split()
    
    if not len(l)>19: return False
    RES = l[5]
    ATOM = l[3]
    try:
        res_idx = l.index(RES,5+1) # residual match check
        atom_idx = l.index(ATOM,3+1) # atom match check
        x,y,z = float(l[10]), float(l[11]), float(l[12]) # float x,y,z check
    except ValueError as e:
        return False

    #print(RES, l[res_idx])
    #print(ATOM, l[atom_idx])
    if not RES == l[res_idx]: #RESIDUE
        return False
    if not ATOM == l[atom_idx]: #ATOM TYPE
        return False
    #print(l)
    return True

def parse(filepath):
    fr = open(filepath)
    lines = fr.readlines()

    #
    # First Parse: build structured data
    #
    Sdata = defaultdict( lambda:defaultdict(dict) )
    for l in lines:
        if isatomline(l):
            chain_id, idx, RES, atom, x,y,z = process_raw_line(l)
            if RES in DICT_R2RES:
                R = DICT_R2RES[RES]
                Sdata[chain_id][idx][atom] =  {'R':R, 'coord':[x,y,z]}

    # 
    # Second Parse: format and construct from structured data
    #
    Fdata = defaultdict( dict )
    for chain in Sdata:
        CA,CB,C,N = [],[],[],[]
        seq = ''
        for idx in Sdata[chain]:
            r = Sdata[chain][idx]
            #if ('CA' in r) and ('N' in r) and ('C' in r):
            if ('CA' in r):
                seq += r['CA']['R']
                ca = r['CA']['coord']
                CA.append(ca)
                '''
                c,n = r['C']['coord'], r['N']['coord'] 
                C.append(c)
                N.append(n)
                if 'CB' in r: CB.append( r['CB']['coord'] )
                else: CB.append( cb(ca,c,n) )
                '''

        Fdata[chain]['SEQ'] = seq
        Fdata[chain]['CA'] = np.array(CA)
        '''
        Fdata[chain]['CB'] = np.array(CB)
        Fdata[chain]['C'] = np.array(C)
        Fdata[chain]['N'] = np.array(N)
        '''

    return Fdata



# Issue 1
# There are discrepencies between using biopython and non-biopython
# This is because some amino acids are placed under HETATM (non-standard atoms) with valid residue names
# example: 12AS.cif
# Since the baker group only parses atoms under the label ATOM, we will also do the same

# Issue 2
# Seems like most of the structural files have missing residues and segments of the chain
# We will adopt the same strategy as the baker group which is to just proceed
# We will save the sequence data for which we were able to extract the 3d coordinates

# Issue 3
# 10 1A02 chains F,J are non amino acid chains which do not exist
# Should not be in the ffindex
# How did the original parser find out

# Issue 4
# It seems like for some proteins the chain id is referred from the -3 index
# Ok so chain id is always listed at -3 index and not the index-like identifier at position 6


# Issue 5
# 59234 6kzb issue where additional ? at end of the line causes the minus indexing to be off
# Need to adjust these indicies to positive indicies.

# TODO: Issue 6
# There are a quite a number of files were only the CA are specified
# This may become an issue when calculating diherals and other angles
# 1AIN
# 1C53

# Issue 7
# 16971 2MUP [Errno 2] No such file or directory: '/raid0/20210112_CIF/2mup.cif'
#

# Issue 8
# Some atom lines have additional elements in between the repeated RES, chain_id, and ATOM type at the end
# Preventing us to get a relative checking
# So we add code to dynamically calculate the position of these elements to parse the indices
# 1dow.cif

# Issue 9
# Addition checks required
# 3flo has a non atom line that passes all the previous checks
# Added a float check
