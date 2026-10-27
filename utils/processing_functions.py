"""从 PMGen 提取的结构预测工具函数；不含肽生成流程。"""
import pandas as pd


import os


import re


import numpy as np


import subprocess


from Bio.PDB import PDBParser, PPBuilder, PDBIO


from Bio import pairwise2, PDB, SeqIO


from Bio.Align import substitution_matrices


from scipy.spatial.distance import cdist


import random


import sys


import Levenshtein


from typing import List


import warnings


from user_setting import netmhcipan_path, netmhciipan_path, pmgen_abs_dir


def get_aminoacid_sequences(pdb_file, exclude_residues=None):
    """
    Get the amino acid sequences from a PDB file, excluding specified residues.

    Parameters:
    - pdb_file (str): Path to the PDB file.
    - exclude_residues (list or None): List of residue names to exclude from the sequences. If None, no exclusions.

    Returns:
    - tuple: (Concatenated amino acid sequence from all chains, Dictionary mapping chain identifiers to their amino acid sequences.)
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    # Initialize an empty string to store the concatenated amino acid sequence
    concatenated_sequence = ''
    # Initialize a dictionary to store sequences for each chain
    chain_sequences = {}
    # Iterate through the structure to extract the sequences
    for model in structure:
        for chain in model:
            chain_id = chain.id
            chain_sequence = ''
            for residue in chain:

                # Check if the residue is an amino acid and not in the exclusion list
                if PDB.is_aa(residue) and (exclude_residues is None or residue.get_resname() not in exclude_residues):
                    try:
                        chain_sequence += PDB.Polypeptide.three_to_one(residue.get_resname())
                    except KeyError:
                        chain_sequence += 'C'
            # Update the concatenated sequence
            concatenated_sequence += chain_sequence
            # Update the dictionary with chain sequences
            chain_sequences[chain_id] = chain_sequence
            # Add separator between chains
            concatenated_sequence += '/'
    # Remove the trailing separator
    concatenated_sequence = concatenated_sequence.rstrip('/')
    return concatenated_sequence, chain_sequences


def process_string(input_string):
    """
    Processes a given string by extracting numbers and removing both numbers and spaces.
    Parameters:
    - input_string (str): The input string to be processed.
    Returns:
    - numbers_variable (list): A list containing the extracted numbers as integers.
    - final_result (str): The input string without both numbers and spaces.
    """
    # Find all numbers in the string using regular expression
    numbers = re.findall(r'\d+', input_string)
    # Extracted numbers as a new variable
    numbers_variable = list(map(int, numbers))
    # Remove numbers from the original string
    string_without_numbers = re.sub(r'\d+', '', input_string)
    # Remove spaces from the string without numbers
    final_result = string_without_numbers.replace(' ', '')

    return numbers_variable, final_result


def map_query_template(target_sequence, template_sequence, tar_dist=0, temp_dist=0):
    blosum62 = substitution_matrices.load("BLOSUM62")
    #alignments = pairwise2.align.localds(target_sequence, template_sequence, MatrixInfo.blosum62, -11, -1)
    alignments = pairwise2.align.localds(target_sequence, template_sequence, blosum62, -11, -1)
    alignment = alignments[0]
    list_alignment = list((pairwise2.format_alignment(*alignment)).split("\n"))
    start_target,_ = process_string(list_alignment[0])
    if len(start_target)==0: start_target=0
    else: start_target=int(start_target[0]) - 1
    start_template,_ = process_string(list_alignment[2])
    if len(start_template)==0: start_template=0
    else: start_template=int(start_template[0]) - 1
    signs = str(list_alignment[1]).replace(" ","")
    L = []
    for i,s in enumerate(signs):
        if s=="|":
            L.append(f"{i+start_target+tar_dist}:{i+start_template+temp_dist}")
    final = ""
    for l in L:
        if len(final)==0:
            final = l
        else:
            final = final + ";" + l

    return final


def prepare_alignment_file_without_peptide(template_id, mhc_seq, mhc_type, output_path, template_path,
                           template_csv_path=os.path.join(pmgen_abs_dir, "data/all_templates.csv")):
    """
    this function generates the alignment file needed as the input for AFfine
    template id: template id in *.pdb format
    mhc_seq: string mhc sequence, for mhc_2 each chain seperated by "/"
    mhc_type: int 1 or 2
    output_path: str where the alignment file to be saved
    template_path: str template fild pdb path
    template_csv_path(optional): str template csv database
    """
    template_csv = pd.read_csv(template_csv_path, sep="\t", header=0)
    template_df = template_csv[template_csv["pdb_file_name"]==template_id]

    if mhc_type==1:
        template_seq = str(template_df["concatenated_sequence"].tolist()[0]).split("/")[0]
        full_template_seq = str(template_df["concatenated_sequence"].tolist()[0]).replace("/","")
        df = pd.DataFrame({"template_pdbfile" : [template_path],
            "target_to_template_alignstring" : [map_query_template(target_sequence=mhc_seq, template_sequence=template_seq)],
            "identities":[len(map_query_template(target_sequence=mhc_seq, template_sequence=template_seq).split(";"))],
            "target_len" : [len(mhc_seq)],
            "template_len" : [len(full_template_seq)]})

    if mhc_type==2:
        template_seq_1 = str(template_df["concatenated_sequence"].tolist()[0]).split("/")[0]
        template_seq_2 = str(template_df["concatenated_sequence"].tolist()[0]).split("/")[1]
        full_template_seq = str(template_df["concatenated_sequence"].tolist()[0]).replace("/", "")
        mhc_seq_1 = mhc_seq.split("/")[0]
        mhc_seq_2 = mhc_seq.split("/")[1]
        align_seq_1 = map_query_template(target_sequence=mhc_seq_1, template_sequence=template_seq_1)
        align_seq_2 = map_query_template(target_sequence=mhc_seq_2, template_sequence=template_seq_2,
                                   temp_dist=len(template_seq_1), tar_dist=len(mhc_seq_1))
        align_seq = align_seq_1 + ";" + align_seq_2
        align_seq = align_seq.rstrip(";")
        df  = pd.DataFrame({"template_pdbfile":[template_path],
                            "target_to_template_alignstring":[align_seq],
                            "identities":[len(align_seq.split(";"))],
                            "target_len":[len(mhc_seq.replace("/",""))],
                            "template_len":[len(full_template_seq)]})
    if output_path:
        df.to_csv(output_path,sep="\t", index=False)
    return df


def prepare_alignment_file_with_peptide(pdb_file, target_seq, mhc_type,
                                        output_path, peptide=True):
    """
    This function generates the alignment files needed for peptide-MHC alphafold predictions
    :param pdb_file: str, path to template pdb files
    :param target_seq: str, target sequence, each chain and peptide separated by "/"
    :param mhc_type: int, 1 or 2
    :param output_path: str, the output file and directory path
    :peptide output_path: bool, if there is peptide in mhc, default is Flase
    :return: p-mhc dataframe and output csv
    """
    concat_seqs , _ = get_aminoacid_sequences(pdb_file)

    if mhc_type==1:
        #template part
        template_mhc_seq = concat_seqs.split("/")[0]
        full_template_seq = concat_seqs.replace("/","")
        peptide_seq_align = ""
        if peptide:
            tmpl_peptide_seq = concat_seqs.split("/")[2]
            peptide_seq_align = map_query_template(target_sequence=tmpl_peptide_seq, template_sequence=tmpl_peptide_seq,
                                                   tar_dist=(len(target_seq.replace("/",""))-len(tmpl_peptide_seq)),
                                                   temp_dist=len(concat_seqs.replace("/",""))-len(tmpl_peptide_seq))
        #Alignemnt part
        mhc_seq_align = map_query_template(target_sequence=target_seq.split("/")[0],
                                           template_sequence=template_mhc_seq)
        alignment = mhc_seq_align + ";" + peptide_seq_align
        alignment = alignment.rstrip(";")
        df = pd.DataFrame({"template_pdbfile" : [pdb_file],
            "target_to_template_alignstring" : [alignment],
            "identities":[len(alignment.split(";"))],
            "target_len" : [len(target_seq.replace("/",""))],
            "template_len" : [len(concat_seqs.replace("/",""))]})

    if mhc_type==2:
        template_mhc_seq_1 = concat_seqs.split("/")[0]
        template_mhc_seq_2 = concat_seqs.split("/")[1]
        peptide_seq_align = ""
        if peptide:
            tmpl_peptide_seq = concat_seqs.split("/")[2]
            peptide_seq_align = map_query_template(tmpl_peptide_seq,tmpl_peptide_seq,
                                                   len(target_seq.replace("/",""))-len(tmpl_peptide_seq),
                                                   len(template_mhc_seq_1)+len(template_mhc_seq_2))
        mhc_seq_align_1 = map_query_template(target_seq.split("/")[0], template_mhc_seq_1)
        mhc_seq_align_2 = map_query_template(target_seq.split("/")[1], template_mhc_seq_2,
                                             len(target_seq.split("/")[0]),
                                             len(template_mhc_seq_1))
        alignment = mhc_seq_align_1 + ";" + mhc_seq_align_2 + ";" + peptide_seq_align
        alignment = alignment.rstrip(";")
        df = pd.DataFrame({"template_pdbfile" : [pdb_file],
            "target_to_template_alignstring" : [alignment],
            "identities":[len(alignment.split(";"))],
            "target_len" : [len(target_seq.replace("/",""))],
            "template_len" : [len(concat_seqs.replace("/",""))]})
    if output_path:
        df.to_csv(output_path, sep="\t", index=False)
    return df


def rename_files(directory, num_template, copy_if_less_template=False):
    # Iterate through files in the specified directory
    i = 0
    for filename in os.listdir(directory):
        if filename.endswith(".pdb") and "BL" in filename:
            # Get the full path of the file
            file_path = os.path.join(directory, filename)

            # Create a new name based on your requirements
            new_filename = f"mod{i}.pdb"
            i = i + 1

            # Rename the file
            new_file_path = os.path.join(directory, new_filename)
            shutil.move(file_path, new_file_path)
    modified_files = os.listdir(directory)
    modified_files = [os.path.join(directory, i) for i in modified_files if '.pdb' in i and 'mod' in i]
    print(f'Only {len(modified_files)} PANDORA templates were generated')
    if copy_if_less_template: # if less templates are created, copies randomly
        print(f'copy_if_less_template==True, So, Duplicate Randomly')

        if len(modified_files) < num_template and len(modified_files) != 0:
            J = num_template - len(modified_files)
            for jj in range(J):
                source_file = os.path.join(directory, random.choice(modified_files))
                dest_file = os.path.join(directory, f'mod{num_template - 1 - jj}.pdb')
                shutil.copy(source_file, dest_file)
    return modified_files


from multiprocessing import Pool, cpu_count


from pathlib import Path


import tempfile


import shutil


def alignment_to_string(alignment):
    """
    Converts an alignment into a string format where ';' separates pairs
    and ':' denotes the index mapping (query:target).
    :param alignment: A tuple or list of two aligned sequences (query, target)
    :return: A formatted string showing index mappings
    """
    query, target = alignment
    mapping = []
    q_idx, t_idx = 0, 0  # Track original indices
    for q_res, t_res in zip(query, target):
        if q_res != "-" and t_res != "-":  # Both residues align
            mapping.append(f"{q_idx}:{t_idx}")
        if q_res != "-":  # Move query index if not a gap
            q_idx += 1
        if t_res != "-":  # Move target index if not a gap
            t_idx += 1
    return ";".join(mapping), q_idx, t_idx


def alignment_to_df(no_modelling_output_dict, output_dir):
    '''runs in case of no-modelling, when we have no pandora output'''
    template_pdbfile = []
    target_to_template_alignstring = []
    identities = []
    target_len = []
    template_len = []
    for i in range(len(no_modelling_output_dict['template_id'])):
        tp = no_modelling_output_dict['template_path'][i]
        aln, q_idx, t_idx = alignment_to_string([no_modelling_output_dict['aln_target'][i], no_modelling_output_dict['aln_template'][i]])
        template_pdbfile.append(tp)
        target_to_template_alignstring.append(aln)
        identities.append(len(aln.split(';')))
        target_len.append(q_idx)
        template_len.append(t_idx)
    df = pd.DataFrame({
        "template_pdbfile": template_pdbfile,
        "target_to_template_alignstring": target_to_template_alignstring,
        "identities": identities,
        "target_len": target_len,
        "template_len": template_len})
    df.to_csv(output_dir, sep='\t', index=False)


def anchor_combinations_mhc2(n, d1=3, d2=2, d3=3):
    quadruplets = []
    for i in range(1, n + 1):
        j = i + d1
        k = j + d2
        l = k + d3
        if l <= n:
            quadruplets.append([i, j, k, l])
    return quadruplets


def anchor_combinations_mhc1(n, min_distance=6):
    pairs = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if j - i >= min_distance:
                pairs.append([i, j])
    return pairs
