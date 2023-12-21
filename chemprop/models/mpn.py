from typing import List, Union, Tuple
from functools import reduce
from torch_geometric import nn as pnn
import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function
from torch_scatter import scatter_sum, scatter_mean
from .polygnn import polygnn_mp
import time
from onmt.modules.embeddings import PositionalEncoding
from onmt.modules.position_ffn import PositionwiseFeedForward
import networkx as nx
import math
from torch.nn.utils.rnn import pad_sequence
import pdb
from torch_geometric.nn.models import GCN,GAT,GraphSAGE,GIN
import torch.nn.functional as F
from .layers import AttnEncoderXL
from .graphformer import graphformerEncoder
from typing import Any, List, Optional, Tuple
from torch import Tensor
from torch_geometric.utils import cumsum
from torch_geometric.utils import to_dense_batch

def repeat_interleave(
    repeats: List[int],
    device: Optional[torch.device] = None,
) -> Tensor:
    outs = [torch.full((n, ), i, device=device) for i, n in enumerate(repeats)]
    return torch.cat(outs, dim=0)


class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.undirected = args.undirected
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm
        self.with_attn = args.with_attn
        if self.with_attn:
            self.attention_encoder = AttnEncoderXL(args)

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        # layer after concatenating the descriptors if args.atom_descriptors == descriptors
        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(self.hidden_size + self.atom_descriptors_size,
                                                    self.hidden_size + self.atom_descriptors_size,)

    def forward(self,
                mol_graph: BatchMolGraph,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atomic descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [np.zeros([1, atom_descriptors_batch[0].shape[1]])] + atom_descriptors_batch   # padding the first with 0 to match the atom_hiddens
            atom_descriptors_batch = torch.from_numpy(np.concatenate(atom_descriptors_batch, axis=0)).float().to(self.device)

        f_atoms, f_bonds, w_atoms, w_bonds, a2b, b2a, b2revb, \
        a_scope, b_scope, degree_of_polym,_ = mol_graph.get_components(atom_messages=self.atom_messages)

        f_atoms, f_bonds, w_atoms, w_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), \
                                                               w_atoms.to(self.device), w_bonds.to(self.device), \
                                                               a2b.to(self.device), b2a.to(self.device), \
                                                               b2revb.to(self.device)
        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for _ in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                nei_a_weight = index_select_ND(w_bonds, a2b)  # num_atoms x max_num_bonds
                # weight nei_a_message based on edge weights
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1) * weight(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = dot(nei_a_message,nei_a_weight)      rev_message
                nei_a_message = nei_a_message * nei_a_weight[..., None]  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message * w_bonds[..., None]  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        nei_a_weight = index_select_ND(w_bonds, a2x)  # num_atoms x max_num_bonds
        # weight messages
        nei_a_message = nei_a_message * nei_a_weight[..., None]  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # concatenate the atom descriptors
        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError(f'The number of atoms is different from the length of the extra atom features')

            atom_hiddens = torch.cat([atom_hiddens, atom_descriptors_batch], dim=1)     # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)                    # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.dropout_layer(atom_hiddens)                             # num_atoms x (hidden + descriptor size)
        
        if self.with_attn:
            lengths = torch.tensor([i[1] for i in a_scope],device=self.device)
            max_length = torch.max(lengths)
            padded_tensor = pad_sequence(torch.split(atom_hiddens[1:], lengths.tolist()), batch_first=True, padding_value=0)
            poly_vec = self.attention_encoder(padded_tensor,lengths)#distance.view(len(a_scope),distance.shape[-1],distance.shape[-1])[:,:max_length,:max_length]
            poly_vec = torch.cat([poly_vec[i, :a,:] for i,a in enumerate([i[1] for i in a_scope])], dim=0)        #确定是对的吗
            poly_vec = torch.cat([atom_hiddens[0].unsqueeze(0),poly_vec])
        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                w_atom_vec = w_atoms.narrow(0, a_start, a_size)
                # if input are polymers, weight atoms from each repeating unit according to specified monomer fractions
                # weight h by atom weights (weights are all 1 for non-polymer input)
                mol_vec = w_atom_vec[..., None] * mol_vec
                # weight each atoms at readout
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / w_atom_vec.sum(dim=0)  # if not --polymer, w_atom_vec.sum == a_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm

                # if input are polymers, multiply mol vectors by degree of polymerization
                # if not --polymer, Xn is 1
                mol_vec = degree_of_polym[i] * mol_vec

                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden

class mol_graph2data(nn.Module):
    def __init__(self,args):
        super(mol_graph2data,self).__init__()
        self.args = args

    def forward(self,
        mol_graph: BatchMolGraph,
        atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        f_atoms, f_bonds, w_atoms, w_bonds, a2b, b2a, b2revb, a_scope, b_scope, degree_of_polym, distances, node_paths, edge_paths,  ctype  = mol_graph.get_components()
        repeats = [i[1] for i in a_scope]
        batch = repeat_interleave(repeats, device=self.args.device)
        ptr = cumsum(torch.tensor(repeats, device=self.args.device))
        b2revb_individual = []
        edge_index = [[],[]]
        for idx, (a_scope_i, b_scope_i) in enumerate(zip(a_scope, b_scope)):
            b2a_i = b2a[b_scope_i[0]:b_scope_i[0] + b_scope_i[1]]
            b2revb_i = b2revb[b_scope_i[0]:b_scope_i[0] + b_scope_i[1]]

            edge_index[0].extend((b2a_i).tolist())
            edge_index[1].extend((b2a_i[b2revb_i - b_scope_i[0]]).tolist())
            # 如果使用pyG，那么就不需要b2revb_individual
            # b2revb_individual.append(b2revb[b_scope_i[0]:b_scope_i[0]+b_scope_i[1]]-b_scope_i[0])
            # b2revb_individual = torch.cat(b2revb_individual,dim=0)
        edge_index = torch.tensor(edge_index,dtype=torch.long,device=self.args.device)-1    #因为chemprop内部的设置，这里必须-1

        return f_atoms[1:,:].to(self.args.device),\
               edge_index, \
               f_bonds[1:,:].to(self.args.device),\
               w_atoms[1:].to(self.args.device), \
               w_bonds[1:].to(self.args.device), \
               b2revb_individual, \
               distances.to(self.args.device), \
               node_paths, \
               edge_paths, \
               batch, ptr

class GAIN(nn.Module):
    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        super(GAIN, self).__init__()
        self.gin = GIN(in_channels=atom_fdim,hidden_channels=args.hidden_size,num_layers=args.depth,dropout=args.dropout,act=args.activation)
        self.bn1 = torch.nn.BatchNorm1d(args.hidden_size)
        self.gat = GAT(in_channels=args.hidden_size,hidden_channels=args.hidden_size,num_layers=args.depth,dropout=args.dropout,act=args.activation)

    def forward(self,x,edge_index,edge_attr,edge_weight,batch):
        x = self.gin(x=x, edge_index=edge_index, edge_attr=edge_attr,edge_weight=edge_weight, batch=batch)
        x = F.relu(x)
        x = self.bn1(x)
        return F.relu(self.gat(x=x, edge_index=edge_index, edge_attr=edge_attr,edge_weight=edge_weight, batch=batch))


class pyG_helper(nn.Module):
    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        super(pyG_helper, self).__init__()
        self.mol_graph2data_layer = mol_graph2data(args)
        self.args = args
        self.encoder_type = args.encoder_type
        if args.encoder_type in ["gcn","gcn_attn","gcn_pe"]:
            self.mpn = GCN(in_channels=atom_fdim,hidden_channels=args.hidden_size,num_layers=args.depth,dropout=args.dropout,act=args.activation)
        elif args.encoder_type in ["gat","gat_attn","gat_pe"]:
            self.mpn = GAT(in_channels=atom_fdim,hidden_channels=args.hidden_size,num_layers=args.depth,dropout=args.dropout,act=args.activation)
        elif args.encoder_type in ["graphsage","graphsage_attn","graphsage_pe"]:
            self.mpn = GraphSAGE(in_channels=atom_fdim,hidden_channels=args.hidden_size,num_layers=args.depth,dropout=args.dropout,act=args.activation)
        elif args.encoder_type in ["gin","gin_attn","gin_pe"]:
            self.mpn = GIN(in_channels=atom_fdim,hidden_channels=args.hidden_size,num_layers=args.depth,dropout=args.dropout,act=args.activation)
        elif args.encoder_type in ["polygnn", "polygnn_attn", "polygnn_pe"]:
            hps={
                "readout_dim" : args.hidden_size,
                "depth" : args.depth,
                "activation" : get_activation_function(args.activation),
                "ffn_capacity": args.ffn_num_layers,
                "dropout":args.dropout
            }
            self.mpn = polygnn_mp(node_size= atom_fdim, edge_size= bond_fdim,hps=hps,normalize_embedding=True)
        elif args.encoder_type in ["gain", "gain_attn", "gain_pe"]:
            self.mpn = GAIN(args, atom_fdim, bond_fdim)
        elif args.encoder_type in ['graphformer']:
            self.mpn = graphformerEncoder(args, atom_fdim, bond_fdim)

        self.with_attn = args.with_attn
        if self.with_attn:
            self.attention_encoder = AttnEncoderXL(args,bond_fdim)
        self.layer_norm  = nn.LayerNorm(args.hidden_size, eps=1e-6,elementwise_affine=True)

    def forward(self,
        mol_graph: BatchMolGraph,
        atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        x, edge_index, edge_attr, w_atoms, w_bonds, b2revb_individual, distances, node_paths, edge_paths, batch, ptr = self.mol_graph2data_layer(mol_graph)
        if self.encoder_type in ["polygnn", "polygnn_attn", "polygnn_pe"]:
            poly_vec = self.mpn(x=x, edge_index=edge_index, edge_attr=edge_attr,edge_weight=w_bonds,w_atoms=w_atoms, batch=batch)
        elif self.encoder_type in ["graphformer"]:
             poly_vec = self.mpn(x, edge_index, edge_attr, distances, edge_paths, ptr, node_paths)
        else:
            poly_vec = self.mpn(x=x, edge_index=edge_index, edge_attr=edge_attr,edge_weight=w_bonds, batch=batch)
        
        # pdb.set_trace()
        if self.with_attn:
            lengths = ptr[1:] - ptr[:-1]
            dense_batch, mask = to_dense_batch(torch.arange(poly_vec.size(0),device=self.args.device), batch=batch)
            mask = ~(mask.unsqueeze(2) & mask.unsqueeze(1))        # [batch_size, max_size] => [batch_size,max_size,max_size]
            node_count = torch.bincount(batch)
            node_features_split = torch.split(poly_vec, node_count.tolist())
            max_length = dense_batch.size(1)
            padded_tensor = torch.stack([torch.cat([feat, torch.zeros(max_length - feat.size(0), feat.size(1),device=self.args.device)]) for feat in node_features_split])

            atom_message = self.attention_encoder(poly_vec,padded_tensor,mask,edge_index,edge_attr,ptr,batch,distances,node_paths,edge_paths)
            atom_message = torch.cat([atom_message[i, :a,:] for i,a in enumerate(lengths)], dim=0)
            # pdb.set_trace()
            poly_vec = self.layer_norm(poly_vec)+atom_message
        
        poly_vec = w_atoms.view(w_atoms.shape[0], 1) * poly_vec         # 这里的pooling方式可以改
        poly_vec = scatter_sum(poly_vec, batch, dim=0)
        w_atoms = scatter_sum(w_atoms, batch, dim=0)
        poly_vec = poly_vec / w_atoms.view(w_atoms.shape[0], 1)
        return poly_vec


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features)
        self.bond_fdim = bond_fdim or get_bond_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                    overwrite_default_bond=args.overwrite_default_bond_features,
                                                    atom_messages=args.atom_messages)

        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.atom_descriptors = args.atom_descriptors
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features

        if self.features_only:
            return

        if args.encoder_type in ["wDMPNN","wDMPNN_origin_attn","wDMPNN_attn","wDMPNN_pe"]:
            encoder = MPNEncoder
        else:
            encoder = pyG_helper

        if args.mpn_shared:
            self.encoder = nn.ModuleList([encoder(args, self.atom_fdim, self.bond_fdim)] * args.number_of_molecules)
        else:
            self.encoder = nn.ModuleList([encoder(args, self.atom_fdim, self.bond_fdim)
                                          for _ in range(args.number_of_molecules)])

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None,
                # start_time=None,
                # logger=None
                ) -> torch.FloatTensor:

        """
        Encodes a batch of molecules.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        # debug = logger.debug if logger is not None else print
        if type(batch[0]) != BatchMolGraph:
            # Group first molecules, second molecules, etc for mol2graph
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]

            # TODO: handle atom_descriptors_batch with multiple molecules per input
            if self.atom_descriptors == 'feature':
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        atom_features_batch=atom_features_batch,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            elif bond_features_batch is not None:
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            else:
                batch = [mol2graph(b) for b in batch]
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)

            if self.features_only:
                return features_batch

        if self.atom_descriptors == 'descriptor':
            if len(batch) > 1:
                raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                          'per input (i.e., number_of_molecules = 1).')

            encodings = [enc(ba, atom_descriptors_batch) for enc, ba in zip(self.encoder, batch)]
        else:
            encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]
        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)

            output = torch.cat([output, features_batch], dim=1)
        return output
