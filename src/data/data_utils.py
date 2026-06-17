import pandas as pd
from aaindex import aaindex1
import Bio.PDB as PDB
from Bio.PDB.Polypeptide import PPBuilder, is_aa
import numpy as np

def load_data(file: str, type: str ='single') -> pd.DataFrame:
    """
    Loads the DMS data from the given file path.
    args:
        file: file path
        sheet_name: name of the sheet to load, if None, loads the first sheet
    """
    if type not in ['single', 'pair']:
        raise ValueError(f"Invalid type: {type}. Must be 'single' or 'pair'.")
    
    try:
        if type == 'single':
            sheet_name = 'S2 Missense mutation fitnesses'
        else:
            sheet_name = 'S2. Fitness & Epistasis Values'
        data = pd.read_excel(file, sheet_name=sheet_name)
    except:
        data = pd.read_csv(file)
    prior_size = data.shape[0]
    
    if 'Ambler Position' not in data.columns:
        print(f"'Ambler Position' not found returning data as is")
        return data

    data.dropna(subset=['Ambler Position'], inplace=True)
    print(f"Data loaded. Prior size: {prior_size}, After dropna: {data.shape[0]}")
    return data

def load_aa_index(Id: str):
    """
    Loads the amino acid index data from the given record id
    args:
        Id: aaindex1 record
    returns:
        data: a dictionary of the aaindex1 record data
        values: a list of the aa properties in the aaindex1 record data
    """
    data = aaindex1[Id]
    values = data.values
    return data, values

def compare_single_fitness(single_df: pd.DataFrame, double_df: pd.DataFrame, err_dev: float = 1.0) -> tuple[pd.DataFrame, dict, list[int], list[int]]:
    """
    Compare single-mutant fitness values against double-mutant data.
    Supports:
    DF input with 'WT AA 1', 'WT AA 2', 'Mut AA 1', 'Mut AA 2',
    'Mut 1 Fitness', and 'Mut 2 Fitness'
    args:
        single_df: single-mutant data
        double_df: double-mutant data
    returns:
        merged_df: A merged DataFrame.
        stats: A dictionary with comparison statistics.
        drop_indices_single: Single-mutant row indices outside the error bounds.
        drop_indices_double: Original double-mutant row indices outside the error bounds.
    """
    single_df = single_df.copy()
    double_df = double_df.copy()

    if 'Code' not in single_df.columns:
        single_df['Code'] = single_df['WT AA'] + single_df['Ambler Position'].astype(int).astype(str) + single_df['Mutant AA']

    single_df = single_df.reset_index(drop=False).rename(columns={'index': 'Single Index'})
    double_df = double_df.reset_index(drop=False).rename(columns={'index': 'Source Index'})

    double_code_1 = double_df['WT AA 1'] + double_df['Ambler Position'].astype(int).astype(str) + double_df['Mut AA 1']
    double_code_2 = double_df['WT AA 2'] + double_df['Ambler Position'].astype(int).astype(str) + double_df['Mut AA 2']
    double_view = pd.concat(
        [
            pd.DataFrame({
                'Source Index': double_df['Source Index'],
                'Mutant Number': 1,
                'Code': double_code_1,
                'Double Fitness': double_df['Mut 1 Fitness'],
                'Double Fitness Error': double_df['Mut 1 Fitness Error'],
            }),
            pd.DataFrame({
                'Source Index': double_df['Source Index'],
                'Mutant Number': 2,
                'Code': double_code_2,
                'Double Fitness': double_df['Mut 2 Fitness'],
                'Double Fitness Error': double_df['Mut 2 Fitness Error'],
            }),
        ],
        ignore_index=True,
    )
    
    merged_df = pd.merge(single_df, double_view, on='Code', how='inner')

    # criteria: intersection of fitness error bars
    sf = merged_df['Fitness']
    sf_error = merged_df['Estimated error in fitness']
    dbf = merged_df['Double Fitness']
    dbf_error = merged_df['Double Fitness Error']

    merged_df['single_in_double'] =(sf >= dbf - dbf_error* err_dev) & (sf <= dbf + dbf_error*err_dev) 
    merged_df['double_in_single'] = (dbf >= sf - sf_error*err_dev) & (dbf <= sf + sf_error*err_dev)
    merged_df['error_intersection'] = merged_df['single_in_double'] | merged_df['double_in_single']

    # stats for data viewing
    stats = {'single_in_double': merged_df['single_in_double'].sum(), 'single_in_double_ratio': merged_df['single_in_double'].sum() / len(merged_df), 
             'double_in_single': merged_df['double_in_single'].sum(), 'double_in_single_ratio': merged_df['double_in_single'].sum() / len(merged_df), 
             'error_intersection': merged_df['error_intersection'].sum(), 'error_intersection_ratio': merged_df['error_intersection'].sum() / len(merged_df)}
    
    # indices to drop from original dms sets/dfs
    drop_indices_single = merged_df.loc[~merged_df['error_intersection'], 'Single Index'].drop_duplicates().tolist()
    drop_indices_double = merged_df.loc[~merged_df['error_intersection'], 'Source Index'].drop_duplicates().tolist()

    return merged_df, stats, drop_indices_single, drop_indices_double



def load_cif_structure(file_path:str, Id:str) -> PDB.Structure.Structure:
    """
    Loads mmCIF structure from the given file path
    args:
        file_path: path to the mmCIF file
        Id: structure PDBId
    """
    parser = PDB.MMCIFParser()
    structure = parser.get_structure(Id, file_path)
    return structure



def parse_structure(structure: PDB.Structure.Structure,):
    """
    Parse a PDB structure to extract sequence, and atomic corodinates for CA, N, C, O atoms.
    args:
        structure: a Bio.PDB structure object
    returns:
        sequence: amino acid sequence of the structure
        atomic_pos: array of shape (N, 4, 3) with atomic coordinates for CA, N, C, O atoms
    """
    # For Tem-1 beta, there is only one chain
    model = structure[0]
    chain = model['A']
    residues = [res for res in chain.get_residues() if is_aa(res, standard=True)]

    # might want to drop unresolved N-terminal residues if they are missing from the structure. (as in 1BTL)
    # Finberg does the whole thing, since it can effect expression and localization whihc affect fitness. 
    # do we want to decouple the active region from the N-terminal???

    # atom pos
    ca_atoms_pos = np.array([atom.get_coord() for res in residues for atom in res if atom.get_id() == 'CA'])
    n_atoms_pos = np.array([atom.get_coord() for res in residues for atom in res if atom.get_id() == 'N'])
    c_atoms_pos = np.array([atom.get_coord() for res in residues for atom in res if atom.get_id() == 'C'])
    o_atoms_pos = np.array([atom.get_coord() for res in residues for atom in res if atom.get_id() == 'O'])

    atomic_pos = np.stack([ca_atoms_pos, n_atoms_pos, c_atoms_pos, o_atoms_pos], axis=1)


    # sequence from the modeled chain
    sequence = ''.join(str(peptide.get_sequence()) for peptide in PPBuilder().build_peptides(chain))

    return sequence, atomic_pos,



"""def expand_double_mutants(data: pd.DataFrame) -> pd.DataFrame:
    #Expand each double-mutant row into two single-mutation-style rows.
    first_mutant = data.assign(
        Code=data['WT AA 1'] + data['Ambler Position'].astype(int).astype(str) + data['Mut AA 1'],
        Mutant_Number=1,
        Fitness=data['Mut 1 Fitness'],
        Fitness_Error=data['Mut 1 Fitness Error'],
        Partner_Fitness=data['Mut 2 Fitness'],
        Partner_Fitness_Error=data['Mut 2 Fitness Error'],
    )

    second_mutant = data.assign(
        Code=data['WT AA 2'] + data['Ambler Position'].astype(int).astype(str) + data['Mut AA 2'],
        Mutant_Number=2,
        Fitness=data['Mut 2 Fitness'],
        Fitness_Error=data['Mut 2 Fitness Error'],
        Partner_Fitness=data['Mut 1 Fitness'],
        Partner_Fitness_Error=data['Mut 1 Fitness Error'],
    )

    columns = [
        'Code',
        'Mutant_Number',
        'Fitness',
        'Fitness_Error',
        'Partner_Fitness',
        'Partner_Fitness_Error',
        'Double Mutant Fitness',
        'Double Mutant Fitness Error',
        'Epistasis',
    ]

    return pd.concat([first_mutant[columns], second_mutant[columns]], ignore_index=True)"""