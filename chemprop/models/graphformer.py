from typing import Tuple

import torch
from torch import nn
from torch_geometric.utils import degree
import pdb
from typing import Tuple, Dict, List
from chemprop.args import TrainArgs
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import time

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


def batched_all_pairs_shortest_path(G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
    node_paths = {n: paths[n][0] for n in paths}
    edge_paths = {n: paths[n][1] for n in paths}
    return node_paths, edge_paths


def batched_shortest_path_distance(data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    graphs = [to_networkx(sub_data) for sub_data in data.to_data_list()]
    relabeled_graphs = []
    shift = 0
    for i in range(len(graphs)):
        num_nodes = graphs[i].number_of_nodes()
        relabeled_graphs.append(nx.relabel_nodes(graphs[i], {i: i + shift for i in range(num_nodes)}))
        shift += num_nodes

    paths = [batched_all_pairs_shortest_path(G) for G in relabeled_graphs]
    node_paths = {}
    edge_paths = {}

    for path in paths:
        for k, v in path[0].items():
            node_paths[k] = v
        for k, v in path[1].items():
            edge_paths[k] = v

    return node_paths, edge_paths


class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        """
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_index: edge_index of graph (adjacency list)
        :return: torch.Tensor, node embeddings after Centrality encoding
        """
        num_nodes = x.shape[0]

        x += self.z_in[degree(index=edge_index[1], num_nodes=num_nodes).long()] + \
             self.z_out[degree(index=edge_index[0], num_nodes=num_nodes).long()]

        return x

# 纯graphformer代码
# class SpatialEncoding(nn.Module):
#     def __init__(self, max_atom_len: int):
#         """
#         :param max_path_distance: max pairwise distance between nodes
#         """
#         super().__init__()
#         # self.b = nn.Parameter(torch.randn(self.max_path_distance))
#         self.b = nn.Embedding(num_embeddings=max_atom_len, embedding_dim=1,padding_idx=0)

#     def forward(self, paths, ptr) -> torch.Tensor:
#         """
#         :param x: node feature matrix
#         :param paths: pairwise node paths
#         :return: torch.Tensor, spatial Encoding matrix
#         """
#         # spatial_matrix = torch.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)
#         # spatial_matrix = torch.zeros((x.shape[0], x.shape[0]))
#         # for src in paths:
#         #     for dst in paths[src]:
#         #         spatial_matrix[src][dst] = self.b[min(len(paths[src][dst]), self.max_path_distance) - 1]
#         # spatial_matrix = spatial_matrix.to(next(self.parameters()).device)

#         spatial_matrix = torch.zeros((ptr[-1],ptr[-1])).to(next(self.parameters()).device)
#         spatial_partial_matrix = [self.b(path).squeeze() for path in paths]
#         len_ptr = ptr[1:] - ptr[:-1]
#         for i in range(len(ptr)-1):
#             spatial_matrix[ptr[i]:ptr[i+1],ptr[i]:ptr[i+1]] = spatial_partial_matrix[i][0:len_ptr[i],0:len_ptr[i]]
#         return spatial_matrix   #[atom_size,atom_size]  还能再快一点吗？
    

class SpatialEncoding(nn.Module):
    def __init__(self, max_atom_len: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
        self.b = nn.Embedding(num_embeddings=max_atom_len, embedding_dim=1,padding_idx=0)

    def forward(self, distances, ptr) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param paths: pairwise node paths
        :return: torch.Tensor, spatial Encoding matrix
        """
        lengths = ptr[1:] - ptr[:-1]
        max_a = torch.max(lengths).item()
        return self.b(distances).squeeze()[:,:max_a,:max_a]



# # 纯grapformer的代码
# class EdgeEncoding(nn.Module):
#     def __init__(self, edge_dim: int):
#         """
#         :param edge_dim: edge feature matrix number of dimension
#         """
#         super().__init__()
#         self.edge_lin = nn.Linear(edge_dim, 1, bias=False)      #为啥非要是False?
        
#     def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_paths, ptr) -> torch.Tensor:
#         pdb.set_trace()
#         edge_scores = self.edge_lin(edge_attr).squeeze()
#         edge_encoding_matrix = torch.zeros((ptr[-1],ptr[-1])).to(next(self.parameters()).device)

#         edge_ptr = [0] + [i.shape[2] for i in edge_paths]
#         edge_start_index = 0
#         for idx, path in enumerate(edge_paths):
#             path = path.to(next(self.parameters()).device)
#             edge_encoding_matrix[ptr[idx]:ptr[idx+1],ptr[idx]:ptr[idx+1]] = ((edge_scores[edge_start_index:edge_start_index+edge_ptr[idx+1]].unsqueeze(0).unsqueeze(0).expand_as(path) * path).sum(dim=-1) / (path.sum(dim=-1) + 1e-8))
#             edge_start_index += edge_ptr[idx+1]
#         return edge_encoding_matrix #[all_atom_size , all_atom_size]
    

class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim: int):
        """
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_lin = nn.Linear(edge_dim, 1)
        
    def forward(self, edge_attr: torch.Tensor, edge_paths, ptr) -> torch.Tensor:
        
        lengths = ptr[1:] - ptr[:-1]
        max_a = torch.max(lengths).item()
        edge_scores = self.edge_lin(edge_attr).squeeze()
        edge_encoding_matrix = torch.zeros((len(ptr)-1, max_a, max_a)).to(next(self.parameters()).device)

        edge_ptr = [0] + [i.shape[2] for i in edge_paths]
        edge_start_index = 0
        for idx, path in enumerate(edge_paths):
            path = path.to(next(self.parameters()).device)
            edge_encoding_matrix[idx,0:lengths[idx],0:lengths[idx]] = ((edge_scores[edge_start_index:edge_start_index+edge_ptr[idx+1]].unsqueeze(0).unsqueeze(0).expand_as(path) * path).sum(dim=-1) / (path.sum(dim=-1) + 1e-8))
            edge_start_index += edge_ptr[idx+1]
        return edge_encoding_matrix


class NodeEncoding(nn.Module):
    def __init__(self, node_dim: int):
        """
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.node_lin = nn.Linear(node_dim, 1)
        
    def forward(self, x: torch.Tensor, node_paths, ptr) -> torch.Tensor:
        
        lengths = ptr[1:] - ptr[:-1]
        max_a = torch.max(lengths).item()
        node_scores = self.node_lin(x).squeeze()
        node_encoding_matrix = torch.zeros((len(ptr)-1, max_a, max_a)).to(next(self.parameters()).device)
        node_ptr = [0] + lengths.tolist()
        node_start_index = 0
        for idx, path in enumerate(node_paths):
            path = path.to(next(self.parameters()).device)
            node_encoding_matrix[idx,:lengths[idx],:lengths[idx]] = ((node_scores[node_start_index:node_start_index+node_ptr[idx+1]].unsqueeze(0).unsqueeze(0).expand_as(path) * path).sum(dim=-1) / (path.sum(dim=-1) + 1e-8))
            node_start_index += node_ptr[idx+1]
        
        return node_encoding_matrix


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, edge_dim: int):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_encoding = EdgeEncoding(edge_dim)
        self.node_encoding = NodeEncoding(
            256
        )
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths,
                ptr,node_paths) -> torch.Tensor:
        """
        :param query: node feature matrix
        :param key: node feature matrix
        :param value: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
        self.node_encoding(query,node_paths,ptr)
        batch_mask = torch.zeros((query.shape[0], query.shape[0])).to(next(self.parameters()).device)
        # OPTIMIZE: get rid of slices: rewrite to torch 需要处理一下，能让他更快
        for i in range(len(ptr) - 1):
            batch_mask[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
        query = self.q(query)
        key = self.k(key)
        value = self.v(value)
        c = self.edge_encoding(edge_attr, edge_paths, ptr)   # 需要3s？
        a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        a = (a + b + c) * batch_mask
        softmax = torch.softmax(a, dim=-1)
        x = softmax.mm(value)
        return x


# FIX: sparse attention instead of regular attention, due to specificity of GNNs(all nodes in batch will exchange attention)
class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, edge_dim: int):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths,
                ptr,node_paths) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """
        return self.linear(
            torch.cat([
                attention_head(x, x, x, edge_attr, b, edge_paths, ptr,node_paths) for attention_head in self.heads
            ], dim=-1)
        )


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, n_heads):
        """
        :param node_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads, 
            edge_dim=edge_dim,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Linear(node_dim, node_dim)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch,
                edge_paths,
                ptr,node_paths) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths, ptr,node_paths) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime
        return x_new
    


class graphformerEncoder(nn.Module):
    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        super(graphformerEncoder, self).__init__()
        self.node_in_lin = nn.Linear(atom_fdim, args.hidden_size)
        self.edge_in_lin = nn.Linear(bond_fdim, bond_fdim)

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=args.max_degree,
            max_out_degree=args.max_degree,
            node_dim=args.hidden_size
        )

        self.spatial_encoding = SpatialEncoding(
            max_atom_len=args.max_atom_len,
        )

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(node_dim=args.hidden_size, edge_dim=bond_fdim, n_heads=args.attn_enc_heads) for _ in
            range(args.attn_enc_num_layers)
        ])

    def forward(self, x, edge_index, edge_attr, distances, edge_paths, ptr, node_paths):
        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)
        x = self.centrality_encoding(x, edge_index)
        b = self.spatial_encoding(distances, ptr)
        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr,node_paths)
        return x
    

