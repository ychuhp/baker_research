# -*- coding: utf-8 -*-
__author__ = "Manuel Alessandro Collazo"
__maintainer__ = "Manuel Alessandro Collazo"
__email__ = "manuel.collazo@pharmcadd.com"
__status__ = "Dev"

from collections import defaultdict
from datetime import datetime
import textdistance
import numpy as np
import pickle
import json
import bz2
import re
import os

# Initialize constant nan coord and backbone atom arrays as global vars
AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
    'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'HSE': 'H', 'HSD': 'H'
}
NAN_COORDS = np.full(shape=(4, 3), fill_value=np.nan)
BB_ATOMS = np.array([['N'], ['CA'], ['CB'], ['C']])
MOD_IND_ISSUE = []
TAKEN_CHAINS = defaultdict(dict)


def min_max_scale(X, full_data_range, scale_range):
    """
    Scales x, y, z coordinates to a range of scale_range using the minimum
    and maximum values for coordinates of proteins being parsed.

    Args:
        X: numpy array containing x, y, and z coordinates.
        full_data_range: tuple indicating min and max of all coordinates.
        scale_range: tuple indicating the min and max to use for scaling.

    Returns:
        numpy array: returns scaled x, y, z coordinate data.
    """

    # Set range vars for proper scaling among all protein coordinate data
    full_data_min, full_data_max = full_data_range
    scale_min, scale_max = scale_range

    # Scale coordinate data based on scaling variables defined
    X_std = (X - full_data_min) / (full_data_max - full_data_min)
    X_scaled = X_std * (scale_max - scale_min) + scale_min

    return X_scaled


def create_voxels(coordinates):
    """
    Creates voxel numpy array based on protein coordinates.

    Args:
        coordinates: x, y, z coordinates for protein atoms.

    Returns:
        numpy array: returns voxel array representing protein structure.
    """

    # Get rid of rows with any NAN values for processing
    coordinates = coordinates[~np.isnan(coordinates).any(axis=1)]

    # Initialize grid space with 0's
    grid_space = np.zeros((512, 512, 512, 1))

    # Scale coordinate information according to specified scale ranges
    coordinates = min_max_scale(
        coordinates,
        full_data_range=(-100, 100),
        scale_range=(0, 511)
    )

    # Round down each coordinate to specify voxel starting location
    coordinates = np.floor(coordinates).astype(int)
    last_coord = coordinates[0]

    # Set atom and line locations to 1 in grid space representing voxels
    for coord in coordinates:

        # Set coordinate locations to 1
        grid_space[tuple(coord) + (0,)] = 1

        # Set variables for indexing line coordinates
        diffs = np.subtract(coord, last_coord)
        steps = np.max(np.abs(diffs))
        max_step_inds = np.where(np.abs(diffs) == steps)[0]
        var_step_inds = np.where(np.abs(diffs) != steps)[0]

        # Avoid divide by 0 warnings as it doesn't affect desired outcome
        np.seterr(all='ignore')
        var_step = np.floor(np.divide(steps, np.abs(diffs)))

        # Create line between two atom coordinates
        for i in range(1, steps):
            for ind in max_step_inds:
                last_coord[ind] += np.sign(diffs[ind])
            for ind in var_step_inds:
                if i % var_step[ind] == 0:
                    last_coord[ind] += np.sign(diffs[ind])
                    var_step += var_step
            grid_space[tuple(last_coord) + (0,)] = 0.5

        # Reset last coordinate location
        last_coord = coord

    return grid_space.astype(np.float)


def update_pdb_info(tracking, pdb_info):
    """
    Updates pdb_info based on current tracking information.

    Args:
        tracking: list of rows for current residue (same seq no and chain).
        pdb_info: pdb rows with coordinates of protein backbone atoms.

    Returns:
        numpy array: returns pdb_info updated with necessary tracking rows.
    """

    # Get first residue mentioned with minimum alt loc indicator
    tracking = tracking[np.where(tracking[:, 5] == tracking[0][5])]
    tracking = tracking[np.where(tracking[:, 4] == min(tracking[:, 4]))]

    # Add available backbone atom rows in tracking to pdb_info
    pdb_info = np.vstack((
        pdb_info,
        tracking[np.isin(tracking[:, 6], BB_ATOMS)]
    ))

    # If missing backbone atoms, add with nan coordinates to pdb_info
    missing_atoms = BB_ATOMS[~np.isin(BB_ATOMS, tracking[:, 6])]
    ma_rows = len(missing_atoms)
    pdb_info = np.vstack((
        pdb_info,
        np.hstack((
            np.full((ma_rows, 6), tracking[-1][:6]),
            missing_atoms.reshape(ma_rows, 1),
            np.full((ma_rows, 3), np.nan)
        ))
    ))

    return pdb_info


def add_mods(tracking, mod_info, pdb_info):
    """
    Updates pdb_info based on chromophore residue tracking info.

    Args:
        tracking: list of chromophore rows related by seq number and chain.
        mod_info: pdb rows from header with info on chromophore residues.
        pdb_info: pdb rows with coordinates of protein backbone atoms.

    Returns:
        numpy array: returns pdb_info updated with necessary tracking rows.
    """

    # Create patterns to index into atom/residue index combos
    ind_p = re.compile(r'[A-Z]+(\d?)')
    atom_p = re.compile(r'([A-Z]+)\d?')

    # Append residue index as final column of pdb_info for mapping
    res_ind = np.vectorize(lambda x: re.search(ind_p, x).group(1).zfill(2))
    tracking = np.hstack((
        tracking,
        res_ind(tracking[:, 6]).reshape(-1, 1)
    ))

    # Strip atom column of residue index if available
    match_res = np.vectorize(lambda x: re.search(atom_p, x).group(1))
    tracking[:, 6] = match_res(tracking[:, 6])

    # Only take backbone atom rows to avoid indexing issues
    tracking = tracking[np.isin(tracking[:, 6], BB_ATOMS)]

    # Create res_maps list of unique residue indexes
    res_maps = np.sort(np.unique(tracking[:, -1]))

    # Iterate over and utilize residue indexes to map to mod_info residues
    for i, res_ind in enumerate(res_maps):
        res_track = tracking[np.where(tracking[:, -1] == res_ind)]
        try:
            res_track[:, 5] = mod_info[i, -1]
        except IndexError:
            MOD_IND_ISSUE.append(tracking[0][0])
        res_track[:, 3] = res_track[:, 3].astype(int) + i

        # Update pdb_info with each residue in chromophore
        pdb_info = update_pdb_info(res_track[:, :10], pdb_info)

    return pdb_info

def parse_pdb(file_loc, pdb, chain_dict, strict=True):
    """
    Creates numpy array with protein backbone atom data parsed from a
    specified pdb file.

    Args:
        file_loc: full directory location for pdb file to be parsed.
        pdb: pdb identification code that corresponds to pdb file.
        strict: skips the error checking

    Returns:
        numpy array: returns array with protein backbone atom data parsed
            from rows of a pdb file.

    NOTE: PDB row indexing is used for setting atom vars to ensure accuracy.
        Provided regex may be used if improper PDB formatting is suspected.
        The format for atom row parsing was evaluated from the site below:
        https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
    """

    # Create sub patterns for regex grouping
    mod_p = r'\d*?'
    res_let_p = r'[A-Z\d]?'
    res_p = r'[A-Z\d]{3,4}'
    atom_p = r'[A-Z\d\']{,3}'
    chain_p = r'[A-z\d]'
    seq_p = r'-?\d+'
    seq_let_p = r'[A-Z]?'
    coord_p = r'-?\d+\.\d{3}'

    # Create relevant patterns for parsing atom information from pdb rows
    missing_res_pattern = re.compile(
        fr'^REMARK 465\s+{mod_p}\s+({res_let_p})({res_p})\s*({chain_p})\s+'
        fr'({seq_p})({seq_let_p})'
    )
    modified_res_pattern = re.compile(
        fr'SEQADV\s+[A-Z\d]+\s*({res_p})\s*({chain_p})\s*({seq_p})'
        fr'({seq_let_p})\s*[A-Z\d]*\s*[A-Z\d]*\s*({res_let_p})({res_p})\s*'
        fr'{seq_p}{seq_let_p}\s*CHROMOPHORE'
    )
    atom_pattern = re.compile(
        fr'^(?:HETATM|ATOM)\s*\d+\s+({atom_p})\s*({res_let_p})({res_p})'
        fr'\s+({chain_p})\s*({seq_p})({seq_let_p})\s+({coord_p})\s*'
        fr'({coord_p})\s*({coord_p})'
    )
    protein_end_pattern = re.compile(
        fr'^TER\s+\d+\s+[A-Z\d\'\s]*?{res_let_p}{res_p}\s+{chain_p}'
    )
    model_end_pattern = re.compile(r'^ENDMDL')
    resolution_pattern = re.compile(
        fr'RESOLUTION.\s+([\d\.]+)'
    )
    technique_pattern = re.compile(
        fr'^EXPDTA\s+([A-Z])'
    )
    date_pattern = re.compile(
        r'^HEADER.*?(\d{2}-[A-Z]{3}-\d{2})'
    )

    # Initialize variables for data parsing
    tracking = np.empty((0, 10))
    pdb_info = np.empty((0, 10))
    mod_info = np.empty((0, 6))
    errors_available = False
    not_x_ray = False
    mod_flag = False
    resolution = None
    pdb_date = None

    # Open pdb file and save lines to pdb_rows var
    with open(file_loc, 'r') as f:
        pdb_rows = f.readlines()

    # Find index of final TER row to mark the end of protein rows
    for i, line in enumerate(pdb_rows):

        # Get PDB file date!
        if re.search(date_pattern, line):
            date = re.search(date_pattern, line).group(1)
            pdb_date = datetime.strptime(date, '%d-%b-%y')

        # Tag pdb if experimental technique was not X-Ray Diffraction
        if re.search(technique_pattern, line):
            first_let = re.search(technique_pattern, line).group(1)
            if first_let != 'X':
                not_x_ray = True

        # Extract resolution value if inside coordinate error section
        if re.search(resolution_pattern, line):
            reso = re.search(resolution_pattern, line).group(1)
            resolution = float(reso) if reso and reso[0].isdigit() else None

        # Update end_ind with each sequential TER row found
        if re.search(protein_end_pattern, line):
            end_ind = i

        # If models available, break after ENDMDL found
        if re.search(model_end_pattern, line):
            break

    # Iterate over all protein rows and parse out backbone atom information
    for line in pdb_rows[:end_ind]:

        # If X-Ray Diffraction not used or resolution is too large, break
        if not_x_ray or not pdb_date or not resolution or resolution >= 3.5:
            #return None
            a=1

        # If line contains missing residue, add rows with NAN
        # coordinates for each backbone atom (3) in pdb_info
        if re.search(missing_res_pattern, line):

            # Initialize missing residue groups and assign to respective vars
            missing_res_groups = re.search(missing_res_pattern, line)
            res_let = missing_res_groups.group(1)
            res = missing_res_groups.group(2)
            chain = missing_res_groups.group(3)
            seq = missing_res_groups.group(4)
            seq_let = missing_res_groups.group(5)

            # Create backbone atom data for residue and add to pdb_info
            details = np.repeat(
                [[pdb, chain, seq_let, seq, res_let, res]],
                repeats=4,
                axis=0
            )
            pdb_info = np.vstack((
                pdb_info,
                np.hstack((details, BB_ATOMS, NAN_COORDS))
            ))

        # If line contains modified residue, add info to mod_info
        if re.search(modified_res_pattern, line):

            # Initialize modified residue groups and assign to respective vars
            modified_res_groups = re.search(modified_res_pattern, line)
            code = modified_res_groups.group(1)
            chain = modified_res_groups.group(2)
            seq = modified_res_groups.group(3)
            seq_let = modified_res_groups.group(4)
            res_let = modified_res_groups.group(5)
            res = modified_res_groups.group(6)

            # Add chromophore residue information to mod_info
            mod_info = np.vstack((
                mod_info,
                np.hstack((code, chain, seq, seq_let, res_let, res))
            ))

        # If line contains HETATM/ATOM information, parse relevant info
        if re.search(atom_pattern, line):

            # Index line to pull respective var info and create vars row
            atom = line[12:16].strip()
            res_let = line[16].strip()
            res = line[17:20].strip()
            chain = line[21].strip()
            seq = line[22:26].strip()
            seq_let = line[26].strip()
            full_seq = seq + seq_let
            x = line[30:38].strip()
            y = line[38:46].strip()
            z = line[46:54].strip()
            row = np.hstack((
                pdb, chain, seq_let, seq, res_let, res, atom, x, y, z
            ))

            # If line is first atom of protein, initialize tracking var
            if tracking.size == 0:
                track_chain = chain
                track_seq = full_seq
                tracking = np.vstack((tracking, row))
            else:

                # If new residue row . . .
                if track_seq != full_seq or track_chain != chain:

                    # If tracking contains chromophore rows . . .
                    if mod_flag:
                        tracking[:, 3] = temp_seq
                        pdb_info = add_mods(tracking, mod_info, pdb_info)
                        track_seq = pdb_info[-1][3]
                        mod_flag = False

                    # Else update pdb_info with prior residue information
                    else:
                        pdb_info = update_pdb_info(tracking, pdb_info)

                    # If current row contains chromophore residue code,
                    # flag current tracking array
                    if mod_info[:, 0].size > 0 and res in mod_info[:, 0]:
                        temp_seq = int(track_seq) + 1
                        mod_flag = True

                    # Reinitialize track vars with new residue row info
                    track_chain = chain
                    track_seq = full_seq
                    tracking = row.reshape(1, 10)

                # Else update current residue tracking info
                else:
                    tracking = np.vstack((tracking, row))

    # Add information for final residue if not already added
    pdb_info = update_pdb_info(tracking, pdb_info)

    # Sort pdb_info by priority: chain, seq, seq_let, atom (Reversed: N, CA, C)
    # pdb_info = pdb_info[np.lexsort((
    #     pdb_info[::-1, 6],
    #     pdb_info[:, 2],
    #     pdb_info[:, 3].astype(int),
    #     pdb_info[:, 1]
    # ))]

    # For chain in PDB . . .
    for chain in np.unique(pdb_info[:, 1]):

        # Try / Except issued as chain info may not be present in chain_dict
        try:
            # Designate vars for evaluation
            chain_id = f'{pdb}_{chain}'
            chain_info = pdb_info[pdb_info[:, 1] == chain]
            if strict: chain_seq = chain_dict[chain_id]['seq']
            atom_sort = np.array([1 if atm == 'N' else 2 if atm == 'CA' else 3
                                  if atm == 'CB' else 4 for atm in
                                  chain_info[:, 6]]).reshape(-1,)
            chain_info = chain_info[np.lexsort((
                atom_sort,
                chain_info[:, 2],
                chain_info[:, 3].astype(int),
                chain_info[:, 1]
            ))]

            if strict:
                # Dump ligands if available
                if len(chain_seq) < 19:
                    continue

                # create pdb sequence based on parsed data
                pdb_seq = ''.join([AA_MAP[aa] if aa in AA_MAP.keys() else 'X'
                                   for aa in chain_info[:, 5][::4]])

                # Check if parsed data matches fasta information
                clean_parse = pdb_seq == chain_seq and len(chain_info) == \
                    len(chain_seq) * 4 and ~np.any(chain_info[chain_info[:, 6]
                                                              != 'CB'] == 'nan')

                # Check sequence against similar length sequences for heterogeneity
                temp_dict = {k: v for k, v in TAKEN_CHAINS.items() if
                             v['seq'] == chain_seq}
                non_homologous = not TAKEN_CHAINS or not temp_dict
                better_res_chain = temp_dict and clean_parse and resolution < \
                    temp_dict[list(temp_dict.keys())[0]]['res']

                # If pdb is parsed correctly and is heterogeneous, yield chain info
                if clean_parse and non_homologous:
                    TAKEN_CHAINS[chain_id]['seq'] = chain_seq
                    TAKEN_CHAINS[chain_id]['res'] = resolution
                    TAKEN_CHAINS[chain_id]['date'] = pdb_date
                    yield chain_info

                elif better_res_chain:
                    inferior_chain = list(temp_dict.keys())[0]
                    if TAKEN_CHAINS[inferior_chain]['date'] < pdb_date:
                        print(f'Inferior Resolution Chain {inferior_chain}')
                        print(f'Replaced by {chain_id}')
                        coord_loc = r'/home/sysguru/total_pdb/chain_coords'
                        coord_file = fr'{inferior_chain}.pbz2'
                        delete_file = os.path.join(coord_loc, coord_file)
                        # os.system(fr'rm {delete_file}')
                        TAKEN_CHAINS.pop(inferior_chain)
                        TAKEN_CHAINS[chain_id]['seq'] = chain_seq
                        TAKEN_CHAINS[chain_id]['res'] = resolution
                        TAKEN_CHAINS[chain_id]['date'] = pdb_date
                        yield chain_info
            else:
                yield chain_info

        except KeyError:
            continue


def process_files(pdb_files_loc):
    """
    Creates pickle documents with voxel data representing protein structure.

    Args:
        pdb_files_loc: directory that holds all folders and pdb files
            to be parsed.
    """

    # Utilizes json file to match sequence length with atom data parsed
    with open('chain_data.json') as json_file:
        chain_dict = json.load(json_file)

    # with open('ss_dict.json') as json_file:
    #     ss_dict = json.load(json_file)

    # Create list of processed files to skip files that have already processed
    # voxel_loc = '/home/sysguru/total_pdb/voxel_files'
    # done_files = [x.split('_')[0] for x in
    #               os.listdir(voxel_loc)]

    # Initialize variables to fill with relevant parsing information
    aa_dict = defaultdict(set)
    len_mismatch_list = []
    key_error_list = []
    # voxel_files_made = 0
    files_processed = 0
    # done_files = [x.split('.')[0] for x in
    #               os.listdir(os.path.join(pdb_files_loc, r'coordinates'))]

    # TAKEN_CHAINS = {k: v for k, v in chain_dict.items() if k in done_files}

    folders = ['6p', '5p', '4p', '3p', '2p', '1p', '7p', '8p', '9p']
    # 4pbc.pdb
    num = 0

    # Iterate over folders that hold pdb files (1p, 2p, ...)
    for folder in folders:

        # Get full folder dir and get list of files in dir
        pdb_folder = os.path.join(pdb_files_loc, folder)
        files = sorted([f for f in os.listdir(pdb_folder) if
                        os.path.splitext(f)[-1] == '.pdb'], reverse=True)
        # files = [f for f in files if f[0:4].upper() not in done_files]

        # Notify information on folder being parsed
        print(f'\nExtracting coordinates from {len(files)} pdb files'
              f' in folder: {folder}')

        # Parse relevant protein atom info from each pdb file in current
        # folder and create pickle file with related protein voxels

        # [files.index('5all.pdb'):]:
        for file in files:

            # Notify of processing file and increment processed var
            print(f'Processing file {file}')
            files_processed += 1

            # Initialize file_loc and structure_id vars for parsing
            file_loc = os.path.join(pdb_folder, file)
            structure_id = os.path.splitext(file)[0].upper()

            # Parse pdb file if structure_id available in count_dict
            for chain_info in parse_pdb(file_loc=file_loc,
                                        pdb=structure_id,
                                        chain_dict=chain_dict):
                # chain = chain_info[0][1]
                # # Create zipped file of voxel matrix for chain
                # coord_file_loc = os.path.join(pdb_files_loc, 'chain_coords')
                # z_file = f'{structure_id}_{chain}.pbz2'
                # z_file_save_loc = os.path.join(coord_file_loc, z_file)
                # with bz2.BZ2File(z_file_save_loc, 'wb') as f:
                #     pickle.dump(chain_info, f)

                print(len(TAKEN_CHAINS))

        # After all files parsed, notify of relevant parsing statistics
        print('\n\n')
        print('*' * 100 + '\n')
        print(f'Chromophore Errors: {len(MOD_IND_ISSUE)}\n{MOD_IND_ISSUE}\n\n')
        print(f'Files Processed: {files_processed}\n\n')
        print(f'Chain information files made: {len(TAKEN_CHAINS)}\n')
        print('*' * 100)


if __name__ == '__main__':

    # Set pdb_files_loc, currently running in directory needed
    pdb_files_loc = '/home/sysguru/total_pdb'

    # Process pdb files and create corresponding txt files with coordinate info
    process_files(pdb_files_loc)
