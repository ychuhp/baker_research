# TODO: FNF error not appending

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
from test import ffindex, distM
from collections import defaultdict
from cifparser import parse


from Bio.PDB.PDBParser import PDBParser
parser = PDBParser(QUIET=True)

#from Bio.PDB.MMCIFParser import MMCIFParser as MMCIFParser
from Bio.PDB.MMCIFParser import FastMMCIFParser as MMCIFParser
parser = MMCIFParser(QUIET=True)

def biopython_get_chains(filepath):
    structure = parser.get_structure("", filepath)
    x = {}
    for chain in structure[0].get_chains():
        x[chain.id] = extract_data(chain)
    return x


DICT_R2RES = {
    'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D','CYS':'C', 'GLN':'Q', 'GLU':'E', 'GLY':'G',
    'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K','MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S',
    'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V','ASX':'N', 'GLX':'Q', 'UNK':'G', 'HSD':'H',
    }


# GOAL:
#
#
# Save coordinate info, save CA,N,C,(CB) info
#   See if CA,N,C
#   CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
#
# Save sequence info
#   residue.get_resname()    # returns the residue name, e.g. "ASN"


def extract_data(chain):
    # https://biopython.readthedocs.io/en/latest/chapter_pdb.html
    # A residue id is a tuple with three elements (hetero-field, sequence identifier, insertion code)
    # hetero-field is blank for standard amino and nucleic acids.
    isamino = lambda r: r.get_id()[1] == ' '
    # The above method seems to crash when a amino residue has no CA?
    # As an alternative we use this but I am not sure if it is as good
    isamino = lambda r: ('CA' in r) and ('C' in r) and ('N' in r)
    x = []
    seq = ''
    for residue in chain:
        RES = residue.get_resname()
        if isamino(residue) and RES in DICT_R2RES:
            c = residue['CA'].get_coord()
            r = DICT_R2RES[RES]
            seq += r
            x.append(c)

    x = np.array(x)
    return {'CA':x, 'SEQ':seq}


# -------------------------------------------------------------------------------------------
# MAIN --------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

if __name__ == '__main__':

    save_dir = '/raid0/total_pdb/ca_meta'
    FFINDEX = ffindex('pdb70_a3m.ffindex')
    pC = list( FFINDEX.items() )
    sidx = 0
    #sidx = 24768
    #sidx = 59234 # 6kzb issue where additional ? at end of exp. Resolved when using positive only indicies

    problems = []
    for i,(p,C) in enumerate(pC[sidx:]):

        try:
            print(i+sidx,p,C)
            #chains = biopython_get_chains('PDB/'+p+'.pdb')
            #chains = biopython_get_chains('/raid0/20210112_CIF/'+p.lower()+'.cif')
            chains = parse('/raid0/20210112_CIF/'+p.lower()+'.cif')
            for c in C:
                CA = chains[c]['CA']
                SEQ = chains[c]['SEQ']
                #print( len(SEQ) )
                #print(SEQ)
                M = distM(CA)

                name = p+'_'+c
                np.save( path.join(save_dir,name), {'CA':CA, 'SEQ':SEQ})
                im = Image.fromarray( M.astype(np.int8) )
                im.save( path.join(save_dir, name+'.png') )

        except FileNotFoundError as fnf_error:
            problems.append( str(i+sidx)+' '+str(p)+ ' '+str(fnf_error) )
            ytil.file__list(problems, 'FNF_ERRORS_CIF')
            continue

