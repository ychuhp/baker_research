import sys
import os

root = '/raid0/20210112_PDB'
for d in os.listdir(root):
	for f in os.listdir( os.path.join( root,d) ):
		_f = f[:-4].upper() + '.pdb'
		os.system('ln -s ' + os.path.join(root,d,f)+ ' ' + os.path.join('PDB', _f) )
		#print( os.path.join(root,d,f), os.path.join('PDB', _f) )
