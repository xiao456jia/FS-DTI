import networkx as nx
from dgl.nn.pytorch import GraphConv
from rdkit import Chem
from rdkit.Chem import BRICS, rdmolops
import re
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit.Chem import AllChem

def fragment_molecule(smiles):
    """Decompose the molecules and return the processed fragments"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    raw_frags = BRICS.BRICSDecompose(mol)

    frags_smiles, frags_atom_idx = [], []
    for frag in raw_frags:
        cleaned = re.sub(r'\[\d+\*\]', '[H]', frag)
        cleaned = re.sub(r'\(\)', '', cleaned)

        frag_mol = Chem.MolFromSmiles(cleaned)
        if frag_mol is None:
            continue

        for atom in frag_mol.GetAtoms():
            atom.SetIsotope(0)

        if not mol.HasSubstructMatch(frag_mol):
            continue

        matches = mol.GetSubstructMatches(frag_mol)
        frags_smiles.append(cleaned)
        frags_atom_idx.append(matches)

    return frags_smiles, frags_atom_idx, mol

def mol_to_dgl_graph(frag_mol, frag_id):
    """Directly construct a DGL graph from fragment molecules and add fragment number features"""
    if frag_mol is None:
        return None

    frag_mol_with_h = Chem.AddHs(frag_mol)

    adj = rdmolops.GetAdjacencyMatrix(frag_mol_with_h)

    src, dst = np.nonzero(adj)
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)))

    num_nodes = g.num_nodes()
    g.ndata['fragment_number'] = torch.full((num_nodes,), frag_id, dtype=torch.float)
    return g, frag_mol_with_h

def add_graph_features(g, mol):
    """Add features to DGL graph"""
    if mol is None or g is None:
        return None

    if g.num_nodes() == 0:

        g.add_nodes(1)
        g.ndata['h'] = torch.zeros((1, 8))
        g.ndata['fragment_number'] = torch.full((1,), -1, dtype=torch.float)
        return g


    existing_features = {}
    if 'fragment_number' in g.ndata:
        existing_features['fragment_number'] = g.ndata['fragment_number']

    # Ensure that the number of atoms in the molecule matches the number of nodes in the diagram
    if mol.GetNumAtoms() != g.num_nodes():

        adj = rdmolops.GetAdjacencyMatrix(mol)
        src, dst = np.nonzero(adj)
        g = dgl.graph((torch.tensor(src), torch.tensor(dst)))

        for key, value in existing_features.items():
            if value.shape[0] == g.num_nodes():
                g.ndata[key] = value
            else:

                g.ndata[key] = torch.full((g.num_nodes(),), value[0].item() if value.numel() > 0 else 0, dtype=value.dtype)


    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetMass(),
            atom.GetTotalNumHs(),
            int(atom.IsInRing())
        ]
        atom_features.append(features)


    if len(atom_features) != g.num_nodes():

        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetMass(),
                atom.GetTotalNumHs(),
                int(atom.IsInRing())
            ]
            atom_features.append(features)
        

        if len(atom_features) > g.num_nodes():
            atom_features = atom_features[:g.num_nodes()]
        else:
            while len(atom_features) < g.num_nodes():
                atom_features.append([0] * 8)  # 添加8维零特征

    g.ndata['h'] = torch.tensor(atom_features, dtype=torch.float32)
    

    if 'fragment_number' not in g.ndata and existing_features.get('fragment_number') is not None:
        g.ndata['fragment_number'] = torch.full((g.num_nodes(),), existing_features['fragment_number'][0].item(), dtype=torch.float)
    elif 'fragment_number' not in g.ndata:
        g.ndata['fragment_number'] = torch.full((g.num_nodes(),), -1, dtype=torch.float)
    
    return g

def process_drug_fragment(smiles, max_fragments=11):
    frags_smiles, frags_atom_idx, mol = fragment_molecule(smiles)
    if not frags_smiles:

        virtual_graph = dgl.graph(([], []))
        virtual_graph.add_nodes(1)
        virtual_graph.ndata['h'] = torch.zeros((1, 8))
        virtual_graph.ndata['fragment_number'] = torch.full((1,), -1, dtype=torch.float)
        return virtual_graph

    constructed_graphs = []

    for i, smi in enumerate(frags_smiles):
        if i >= max_fragments:
            break

        frag_mol = Chem.MolFromSmiles(smi)
        if frag_mol is None:
            continue

        g, frag_mol_with_h = mol_to_dgl_graph(frag_mol, frag_id=i)
        if g is None:
            continue

        g = add_graph_features(g, frag_mol_with_h)
        if g is None:
            continue

        constructed_graphs.append(g)


    for i, g in enumerate(constructed_graphs):

        if 'h' not in g.ndata:
            g.ndata['h'] = torch.zeros((g.num_nodes(), 8))
        elif g.ndata['h'].shape[1] != 8:

            if g.ndata['h'].shape[1] < 8:

                padding = torch.zeros((g.num_nodes(), 8 - g.ndata['h'].shape[1]))
                g.ndata['h'] = torch.cat([g.ndata['h'], padding], dim=1)
            else:

                g.ndata['h'] = g.ndata['h'][:, :8]
        

        if 'fragment_number' not in g.ndata:
            g.ndata['fragment_number'] = torch.full((g.num_nodes(),), i if i < len(frags_smiles) else -1, dtype=torch.float)

    while len(constructed_graphs) < max_fragments:
        virtual_graph = dgl.graph(([], []))
        virtual_graph.add_nodes(1)
        virtual_graph.ndata['h'] = torch.zeros((1, 8))
        virtual_graph.ndata['fragment_number'] = torch.full((1,), -1, dtype=torch.float)
        constructed_graphs.append(virtual_graph)

    batched_graph = dgl.batch(constructed_graphs)
    return batched_graph