
import rdkit
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import Lipinski

def HDonors_in_substructure(mol, subs):
    HDonor_atoms = mol.GetSubstructMatches(Lipinski.HDonorSmarts)
    if HDonor_atoms == ():
        return 0
    atoms_in_substructure = mol.GetSubstructMatches(subs)[0]
    count = 0
    for HDonor_atom in HDonor_atoms:
        if HDonor_atom[0] in atoms_in_substructure:
            count += 1
    return count

def HAcceptors_in_substructure(mol, subs):
    HAcceptor_atoms = mol.GetSubstructMatches(Lipinski.HAcceptorSmarts)
    if HAcceptor_atoms == ():
        return 0
    atoms_in_substructure = mol.GetSubstructMatches(subs)[0]
    count = 0
    for HAcceptor_atom in HAcceptor_atoms:
        if HAcceptor_atom[0] in atoms_in_substructure:
            count += 1
    return count

def NHOH_in_substructure(mol, subs):
    NHOH_atoms = mol.GetSubstructMatches(Lipinski.NHOHSmarts)
    if NHOH_atoms == ():
        return 0
    atoms_in_substructure = mol.GetSubstructMatches(subs)[0]
    count = 0
    for NHOH_atom in NHOH_atoms:
        if NHOH_atom[0] in atoms_in_substructure:
            count += 1
    return count

def NumRotableBonds_NotStrict_in_substructure(mol, subs):
    RotableBonds_pairs = mol.GetSubstructMatches(Lipinski.RotatableBondSmarts)
    if RotableBonds_pairs == ():
        return 0
    atoms_in_substructure = mol.GetSubstructMatches(subs)[0]
    count = 0
    for RotableBonds_pair in RotableBonds_pairs:
        if RotableBonds_pair[0] in atoms_in_substructure and \
                RotableBonds_pair[1] in atoms_in_substructure:
            count += 1
    return count

def NumRotableBonds_minusAmides_in_substructure(mol, subs):
    patt1 = Chem.MolFromSmarts("[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]")
    patt2 = Chem.MolFromSmarts("[!$(C(=O)!@[NH])&!D1&!$(*#*)]-&!@[!$(C(=O)!@[NH])&!D1&!$(*#*)]")
    RotableBonds_pairs = mol.GetSubstructMatches(patt1) + mol.GetSubstructMatches(patt2)
    if RotableBonds_pairs == ():
        return 0
    RotableBonds_pairs = set(RotableBonds_pairs)
    atoms_in_substructure = mol.GetSubstructMatches(subs)[0]
    count = 0
    for RotableBonds_pair in RotableBonds_pairs:
        if RotableBonds_pair[0] in atoms_in_substructure and \
                RotableBonds_pair[1] in atoms_in_substructure:
            count += 1
    return count

def NumRotableBonds_minusAmides(mol):
    patt1 = Chem.MolFromSmarts("[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]")
    patt2 = Chem.MolFromSmarts("[!$(C(=O)!@[NH])&!D1&!$(*#*)]-&!@[!$(C(=O)!@[NH])&!D1&!$(*#*)]")
    RotableBonds_pairs = mol.GetSubstructMatches(patt1) + mol.GetSubstructMatches(patt2)
    if RotableBonds_pairs == ():
        return 0
    RotableBonds_pairs = set(RotableBonds_pairs)
    return len(RotableBonds_pairs)