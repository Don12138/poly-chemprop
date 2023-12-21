import csv
import os
import pickle
from typing import Tuple, Dict, List
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
import networkx as nx
import torch

def floyd_warshall_source_to_all(G, source, cutoff=None):
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))

    edges = {edge: i for i, edge in enumerate(G.edges())}

    level = 0  # the current level
    nextlevel = {source: 1}  # list of nodes to check at next level
    node_paths = {source: [source]}  # paths dictionary  (paths to key from source)
    edge_paths = {source: []}

    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G[v]:
                if w not in node_paths:
                    node_paths[w] = node_paths[v] + [w]
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                    nextlevel[w] = 1

        level = level + 1

        if (cutoff is not None and cutoff <= level):
            break
    return node_paths, edge_paths


def all_pairs_shortest_path(G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
    node_paths = {n: paths[n][0] for n in paths}
    edge_paths = {n: paths[n][1] for n in paths}
    return node_paths, edge_paths

def shortest_path_distance(data: Data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    G = to_networkx(data)
    node_paths, edge_paths = all_pairs_shortest_path(G)
    return node_paths, edge_paths

def get_distance_and_path(data,max_len=52,max_distance=100):
    '''
    获取邻接矩阵、最短距离、最短路径(节点和边)
    '''
    G = to_networkx(data)
    atom_num = data.x.size(0)
    bond_num = data.edge_index.size(1)

    node_paths_tensor = torch.zeros(atom_num,atom_num,atom_num).bool()
    edge_paths_tensor = torch.zeros(atom_num,atom_num,bond_num).bool()

    padded_distance = np.zeros((max_len, max_len), dtype=np.int32)# * max_distance
    distance_matrix = nx.floyd_warshall_numpy(G)
    distance_matrix[np.isnan(distance_matrix) | np.isinf(distance_matrix)] = 0
    padded_distance[:atom_num, :atom_num] = distance_matrix
    
    node_paths, edge_paths = shortest_path_distance(data)

    for i in range(atom_num):
        if i in node_paths.keys():
            for j in range(atom_num):
                if j in node_paths[i].keys():
                    node_paths_tensor[i,j,node_paths[i][j]] = True
            for j in range(bond_num):
                if j in edge_paths[i].keys():
                    edge_paths_tensor[i,j,edge_paths[i][j]] = True
    

    # 修正距离
    # padded_distance = [[8 if x >=8 and x <15 else x for x in row] for row in padded_distance]
    # padded_distance = [[9 if x < max_distance and x>=15 else x for x in row] for row in padded_distance]
    # padded_distance = [[-1 if x == max_distance else x for x in row] for row in padded_distance]
    
    return torch.LongTensor(padded_distance),node_paths_tensor,edge_paths_tensor




def save_features(path: str, features: List[np.ndarray]) -> None:
    """
    Saves features to a compressed :code:`.npz` file with array name "features".

    :param path: Path to a :code:`.npz` file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:

    * :code:`.npz` compressed (assumes features are saved with name "features")
    * .npy
    * :code:`.csv` / :code:`.txt` (assumes comma-separated features with a header and with one line per molecule)
    * :code:`.pkl` / :code:`.pckl` / :code:`.pickle` containing a sparse numpy array

    .. note::

       All formats assume that the SMILES loaded elsewhere in the code are in the same
       order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size :code:`(num_molecules, features_size)` containing the features.
    """
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        features = np.load(path)['features']
    elif extension == '.npy':
        features = np.load(path)
    elif extension in ['.csv', '.txt']:
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            features = np.array([[float(value) for value in row] for row in reader])
    elif extension in ['.pkl', '.pckl', '.pickle']:
        with open(path, 'rb') as f:
            features = np.array([np.squeeze(np.array(feat.todense())) for feat in pickle.load(f)])
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


def load_valid_atom_or_bond_features(path: str, smiles: List[str]) -> List[np.ndarray]:
    """
    Loads features saved in a variety of formats.

    Supported formats:

    * :code:`.npz` descriptors are saved as 2D array for each molecule in the order of that in the data.csv
    * :code:`.pkl` / :code:`.pckl` / :code:`.pickle` containing a pandas dataframe with smiles as index and numpy array of descriptors as columns
    * :code:'.sdf' containing all mol blocks with descriptors as entries

    :param path: Path to file containing atomwise features.
    :return: A list of 2D array.
    """

    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        container = np.load(path)
        features = [container[key] for key in container]

    elif extension in ['.pkl', '.pckl', '.pickle']:
        features_df = pd.read_pickle(path)
        if features_df.iloc[0, 0].ndim == 1:
            features = features_df.apply(lambda x: np.stack(x.tolist(), axis=1), axis=1).tolist()
        elif features_df.iloc[0, 0].ndim == 2:
            features = features_df.apply(lambda x: np.concatenate(x.tolist(), axis=1), axis=1).tolist()
        else:
            raise ValueError(f'Atom/bond descriptors input {path} format not supported')

    elif extension == '.sdf':
        features_df = PandasTools.LoadSDF(path).drop(['ID', 'ROMol'], axis=1).set_index('SMILES')

        features_df = features_df[~features_df.index.duplicated()]

        # locate atomic descriptors columns
        features_df = features_df.iloc[:, features_df.iloc[0, :].apply(lambda x: isinstance(x, str) and ',' in x).to_list()]
        features_df = features_df.reindex(smiles)
        if features_df.isnull().any().any():
            raise ValueError('Invalid custom atomic descriptors file, Nan found in data')

        features_df = features_df.applymap(lambda x: np.array(x.replace('\r', '').replace('\n', '').split(',')).astype(float))

        features = features_df.apply(lambda x: np.stack(x.tolist(), axis=1), axis=1).tolist()

    else:
        raise ValueError(f'Extension "{extension}" is not supported.')

    return features
