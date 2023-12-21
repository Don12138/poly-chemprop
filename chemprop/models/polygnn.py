import torch
import torch.nn as nn
import torch_geometric.nn as pnn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from .layers import ffn, hidden_xavier_bn, output
import pdb


class polygnn_mp_block(pnn.MessagePassing):  # Pseudo DeepChem layer
    def __init__(self, node_size, E, hps):
        super().__init__(aggr="add", node_dim=0)
        self.node_size = node_size
        self.E = E  # the edge mapper
        self.hps = hps

        # set up V
        self.V = [
            nn.Dropout(hps['dropout']),
            ffn(self.node_size, self.node_size, self.hps)
        ]
        self.V = nn.Sequential(*self.V)
        # set up U
        self.U = [
            nn.Dropout(hps['dropout']),
            ffn(
            (self.node_size * 2) + self.E.input_dim,
            self.node_size,
            self.hps,
        )
        ]
        self.U = nn.Sequential(*self.U)

    def forward(self, x, edge_index, edge_attr,w_atoms,w_bonds, batch):
        return self.propagate(
            edge_index=edge_index, x=x, edge_attr=edge_attr, w_atoms=w_atoms,w_bonds=w_bonds,batch=batch
        )

    def update(self, aggr_out, x,w_atoms,w_bonds):
        """
        aggr_out = output of self.aggregate( self.message(.) )
        """
        up = F.elu(torch.cat((aggr_out, x), 1))
        out = torch.mul(self.U(up),w_atoms.unsqueeze(1))
        # out = self.U(up)
        return out

    def message(self, x_i, x_j, edge_attr, edge_index,w_atoms,w_bonds):
        """
        x_j are the neighboring atoms that correspond to x
        """
        m_j = self.V(x_j)
        m_ij = self.E(edge_attr)
        # pdb.set_trace()
        return torch.mul(w_bonds.unsqueeze(1),torch.cat((m_j, m_ij), 1))
        # return torch.cat((m_j, m_ij), 1)

class polygnn_mp(nn.Module):
    def __init__(
        self,
        node_size,
        edge_size,
        hps,
        normalize_embedding
    ):
        super().__init__()

        self.node_size = node_size
        self.edge_size = edge_size
        self.normalize_embedding = normalize_embedding
        self.hps = hps

        # set up read-out layer
        self.readout_dim = hps['readout_dim']
        self.R = hidden_xavier_bn(
            self.node_size,
            self.readout_dim,
            hps=self.hps,
        )
        # set up message passing layers
        self.E = ffn(
            edge_size,
            edge_size,
            hps=self.hps
        )

        # set up message passing blocks
        # self.mp_layer = polygnn_mp_block(
        #             self.node_size,
        #             self.E,
        #             hps=self.hps,
        #         )

        self.mp_layers = nn.ModuleList(
            [
                polygnn_mp_block(
                    self.node_size,
                    self.E,
                    hps=self.hps,
                )
                for _ in range(self.hps['depth'])
            ]
        )

    def forward(self, x, edge_index, edge_attr,edge_weight,w_atoms, batch):
        # pdb.set_trace()
        x_clone = x.clone().detach()
        # message passes
        for i,layer in enumerate(self.mp_layers):
        # for i in range(self.hps['depth']):
            if i == 0 or i == 1:
                x = layer(x, edge_index, edge_attr,w_atoms,edge_weight, batch)
            else:
                skip_x = last_last_x + x  # skip connection
                x = layer(skip_x, edge_index, edge_attr,w_atoms,edge_weight, batch)
            if i > 0:
                last_last_x = last_x
            last_x = x

        x = self.R(
            x + x_clone
        )  # combine initial feature vector with updated feature vector and map. Skip connection.
        
        # readout
        # if self.normalize_embedding:
        #     x = scatter_mean(x, batch, dim=0)
        # else:
        #     x = scatter_sum(x, batch, dim=0)
        return x

# class polygnn(nn.Module):
#     def __init__(self,node_size,edge_size,hps,normalize_embedding="mean",target_mean=None):  #如果是多目标优化加了一个selector_dim
#         super().__init__()
#         self.node_size = node_size
#         self.edge_size = edge_size
#         # self.selector_dim = selector_dim
#         self.normalize_embedding = normalize_embedding
#         self.hps = hps
#         self.mpnn = polygnn_mp(
#             node_size,
#             edge_size,
#             hps,
#             normalize_embedding
#         )
#         self.activation = self.hps['activation']
#         self.output = output(self.hps['readout_dim'],target_mean=[2.82])

#     def forward(self, data):
#         x, edge_index, edge_attr,w_atoms,w_bonds, batch = (
#             data.x,
#             data.edge_index,
#             data.edge_attr,
#             data.w_atoms,
#             data.w_bonds,
#             data.batch,
#         )  # extract variables
#         x = self.output(self.activation(self.mpnn(x, edge_index, edge_attr,w_atoms,w_bonds, batch))).squeeze()
#         return x


