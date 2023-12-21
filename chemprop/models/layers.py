import torch
import torch.nn as nn
import torch_geometric.nn as pnn
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
import pdb
import math
from .graphformer import SpatialEncoding, EdgeEncoding, NodeEncoding


def get_unit_sequence(input_dim, output_dim, n_hidden):
    """
    Smoothly decay the number of hidden units in each layer.
    Start from 'input_dim' and end with 'output_dim'.

    Examples:
    get_unit_sequence(1,1024,4) = [1, 4, 16, 64, 256, 1024]
    get_unit_sequence(1024,1,4) = [1024, 256, 64, 16, 4, 1]
    """
    reverse = False
    if input_dim > output_dim:
        reverse = True
        input_dim,output_dim = output_dim,input_dim

    diff = abs(output_dim.bit_length() - input_dim.bit_length())
    increment = diff // (n_hidden+1)

    sequence = [input_dim] + [0] * (n_hidden) + [output_dim]

    for idx in range(n_hidden // 2):
        sequence[idx+1] = 2 ** ((sequence[idx]).bit_length() + increment-1)
        sequence[-2-idx] = 2 ** ((sequence[-1-idx]-1).bit_length() - increment)

    if n_hidden%2 == 1:
        sequence[n_hidden // 2 + 1] = (sequence[n_hidden // 2] + sequence[n_hidden // 2+2])//2

    if reverse: 
        sequence.reverse()

    return sequence

class output(nn.Module):
    """
    Output layer with xavier initialization on weights
    """

    def __init__(self, size_in, target_mean=[0]):
        super().__init__()
        self.size_in, self.size_out = size_in, len(target_mean)
        self.target_mean = target_mean

        self.linear = nn.Linear(self.size_in, self.size_out)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.target_mean != None:
            self.linear.bias.data = torch.tensor(target_mean)

    def forward(self, x):
        return self.linear(x)

class hidden_xavier_bn(nn.Module):
    """
    Hidden layer with xavier initialization and batch normalization
    """
    def __init__(self, size_in, size_out, hps):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.linear = nn.Linear(self.size_in, self.size_out)
        nn.init.xavier_uniform_(self.linear.weight)
        # self.bn = nn.BatchNorm1d(self.size_out)
        self.activation = hps['activation']

    def forward(self, x):
        # print(x)
        return self.activation(self.linear(x))
        # return self.activation(self.bn(self.linear(x)))

class ffn(nn.Module):
    """
    A Feed-Forward neural Network that uses DenseHidden layers
    """

    def __init__(self, input_dim, output_dim, hps):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hps = hps
        self.layers = nn.ModuleList()
        self.unit_sequence = get_unit_sequence(
            input_dim, output_dim, self.hps['ffn_capacity']
        )
        # set up hidden layers
        for ind, n_units in enumerate(self.unit_sequence[:-1]):
            size_out_ = self.unit_sequence[ind + 1]
            self.layers.append(
                hidden_xavier_bn(
                    size_in=n_units,
                    size_out=size_out_,
                    hps=self.hps,
                )
            )

    def forward(self, x):
        """
        Compute the forward pass of this model
        """
        for layer in self.layers:
            x = layer(x)

        return x

def get_sin_encodings(rel_pos_buckets, model_dim) -> torch.Tensor:
    pe = torch.zeros(rel_pos_buckets + 1, model_dim)
    position = torch.arange(0, rel_pos_buckets).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, model_dim, 2, dtype=torch.float) *
                          -(math.log(10000.0) / model_dim)))
    pe[:-1, 0::2] = torch.sin(position.float() * div_term)          # leaving last "position" as padding
    pe[:-1, 1::2] = torch.cos(position.float() * div_term)

    return pe


class polyMultiHeadedRelAttention(nn.Module):
    def __init__(self, args, head_count, model_dim, dropout, edge_dim):
        super().__init__()
        self.args = args
        self.with_pe = args.with_pe
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, model_dim)
        self.linear_values = nn.Linear(model_dim, model_dim)
        self.linear_query = nn.Linear(model_dim, model_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.distance_encoding = SpatialEncoding(args.max_atom_len)
        self.edge_encoding = EdgeEncoding(edge_dim)
        self.node_encoding = NodeEncoding(args.hidden_size)
        # self.u = nn.Parameter(torch.ones(head_count), requires_grad=True)
        # self.v = nn.Parameter(torch.randn(model_dim), requires_grad=True)

    def forward(self, poly_vec, inputs, mask,edge_index,edge_attr,ptr,batch,distances,node_paths,edge_paths):


        batch_size = inputs.size(0)
        max_a = inputs.size(1)
        dim_per_head = self.dim_per_head        # 256 / 8
        head_count = self.head_count            # 8

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query. Seems that we don't need layer_cache here
        query = self.linear_query(inputs)
        key = self.linear_keys(inputs)
        value = self.linear_values(inputs)

        key = shape(key)                # (b, max_a, h) -> (b, head, max_a, h/head)
        value = shape(value)
        query = shape(query)            # (b, max_a, h) -> (b, head, max_a, h/head)

        key_len = key.size(2)           # max_a
        query_len = query.size(2)       # max_a

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)

        scores = torch.matmul(query, key.transpose(2, 3))                 # (b, head, t_q, t_k)

        if self.with_pe:
            # [batch, max_a, max_a]
            a = to_dense_adj(edge_index=edge_index,batch=batch).unsqueeze(1).expand(-1, 2, -1, -1).float() # adj 这个理应编码，但是我还没写 
            
            b = self.distance_encoding(distances, ptr).unsqueeze(1).expand(-1, 2, -1, -1)                        # distance  [batch, max_a, max_a]

            c = self.edge_encoding(edge_attr, edge_paths,ptr).unsqueeze(1).expand(-1, 2, -1, -1)

            d = self.node_encoding(poly_vec, node_paths, ptr).unsqueeze(1).expand(-1, 2, -1, -1)

            addition_pe = torch.cat([a, b, c, d], dim=1)
            # pdb.set_trace()
            scores = scores + addition_pe
        
        scores = scores.float()

        mask = mask.unsqueeze(1)                            # (B, 1, 1, T_values)
        scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        drop_attn = drop_attn.masked_fill(mask, 0.)

        context_original = torch.matmul(drop_attn, value)   # -> (b, head, t_q, h/head)
        context = unshape(context_original)                 # -> (b, t_q, h)

        output = self.final_linear(context)
        attns = attn.view(batch_size, head_count, query_len, key_len)
        
        return output, attns




class MultiHeadedRelAttention(nn.Module):
    def __init__(self, args, head_count, model_dim, dropout, rel_pos_buckets):
        super().__init__()
        self.args = args
        self.with_pe = args.with_pe
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, model_dim)
        self.linear_values = nn.Linear(model_dim, model_dim)
        self.linear_query = nn.Linear(model_dim, model_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.rel_pos_buckets = rel_pos_buckets

        # self.u = nn.Parameter(torch.randn(model_dim), requires_grad=True)
        # self.v = nn.Parameter(torch.randn(model_dim), requires_grad=True)

    def forward(self, inputs, mask, ):
        """
        Compute the context vector and the attention vectors.

        Args:
           inputs (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
           distances: graph distance matrix (BUCKETED), ``(batch, key_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """
        # pdb.set_trace()
        batch_size = inputs.size(0)
        dim_per_head = self.dim_per_head        # 256 / 8
        head_count = self.head_count            # 8

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query. Seems that we don't need layer_cache here
        query = self.linear_query(inputs)
        key = self.linear_keys(inputs)
        value = self.linear_values(inputs)

        key = shape(key)                # (b, t_k, h) -> (b, head, t_k, h/head)
        value = shape(value)
        query = shape(query)            # (b, t_q, h) -> (b, head, t_q, h/head)

        key_len = key.size(2)           # max_len
        query_len = query.size(2)       # max_len

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)

        if not self.with_pe:
            scores = torch.matmul(
                query, key.transpose(2, 3))                 # (b, head, t_q, t_k)

        # else:
        #     # a + c
        #     u = self.u.reshape(1, head_count, 1, dim_per_head)     # [256] -> [1,8,1,32]
        #     a_c = torch.matmul(query + u, key.transpose(2, 3))
        #     a_c = F.layer_norm(a_c, normalized_shape=(a_c.size(-1),), eps=1e-6,\
        #          weight=torch.nn.Parameter(torch.ones(a_c.size(-1)).to("cuda")*5),\
        #             bias = torch.nn.Parameter(torch.zeros(a_c.size(-1)).to("cuda")))

        #     rel_emb = self.relative_pe(distances)           # (b, t_q, t_k) -> (b, t_q, t_k, h)
        #     rel_emb = rel_emb.reshape(                      # (b, t_q, t_k, h) -> (b, t_q, t_k, head, h/head)
        #         batch_size, query_len, key_len, head_count, dim_per_head)

        #     # W_kR = self.W_kR.reshape(1, 1, 1, head_count, dim_per_head, dim_per_head)
        #     # rel_emb = torch.matmul(rel_emb, W_kR)           # (b, t_q, t_k, head, 1, h/head)
        #     # rel_emb = rel_emb.squeeze(-2)                   # (b, t_q, t_k, head, h/head)
        #     #
        #     # b_kR = self.b_kR.reshape(1, 1, 1, head_count, dim_per_head)
        #     # rel_emb = rel_emb + b_kR                        # (b, t_q, t_k, head, h/head)

        #     # b + d
        #     # 此处开始注释
        #     # pdb.set_trace()
        #     query = query.unsqueeze(-2)                     # (b, head, t_q, h/head) -> (b, head, t_q, 1, h/head)
        #     rel_emb_t = rel_emb.permute(0, 3, 1, 4, 2)      # (b, t_q, t_k, head, h/head) -> (b, head, t_q, h/head, t_k)

        #     v = self.v.reshape(1, head_count, 1, 1, dim_per_head)
        #     b_d = torch.matmul(query + v, rel_emb_t
        #                     ).squeeze(-2)                # (b, head, t_q, 1, t_k) -> (b, head, t_q, t_k)
        #     b_d = F.layer_norm(b_d, normalized_shape=(b_d.size(-1),), eps=1e-6,\
        #          weight=torch.nn.Parameter(torch.ones(b_d.size(-1)).to("cuda")),\
        #             bias = torch.nn.Parameter(torch.zeros(b_d.size(-1)).to("cuda")))
        #     scores = a_c + b_d
        
        
        
        scores = scores.float()

        mask = mask.unsqueeze(1)                            # (B, 1, 1, T_values)
        scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)   # -> (b, head, t_q, h/head)
        context = unshape(context_original)                 # -> (b, t_q, h)

        output = self.final_linear(context)
        attns = attn.view(batch_size, head_count, query_len, key_len)

        return output, attns


class SALayerXL(nn.Module):     # TransformerEncoderLayer
    """
    A single layer of the self-attention encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout: dropout probability(0-1.0).
    """

    def __init__(self, args, d_model, heads, d_ff, attention_dropout, edge_dim: int):
        super().__init__()

        self.self_attn = polyMultiHeadedRelAttention(
            args,
            heads, d_ff, dropout=attention_dropout,
            edge_dim=edge_dim
        )
        self.layer_norm_0 = nn.LayerNorm(d_model, eps=1e-6,elementwise_affine=True)
        # self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6,elementwise_affine=True)
        # self.linear_0 = nn.Linear(d_model, d_ff)
        # self.linear_1 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, poly_vec, inputs, mask,edge_index,edge_attr,ptr,batch,distances,node_paths,edge_paths ):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``
            distances (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        # 下面的方法是把注意力的隐藏层进行放缩
        # mid = self.linear_0(self.layer_norm(inputs))
        # context, attn = self.self_attn(mid, mask=mask, distances=distances)

        # out = self.linear_1(self.dropout(context)) + inputs              # skip connection
        
        # 下面的方法是不进行放缩，且设置了残差,此时d_ff 一定等于d_model
        # mid = self.layer_norm_0(inputs)
        context, attn = self.self_attn(poly_vec, inputs, mask, edge_index,edge_attr,ptr,batch,distances,node_paths,edge_paths)
        # pdb.set_trace()
        out = context + inputs
        # out = self.linear_0(mid) - mid + inputs +  self.dropout(context)
        return out

class AttnEncoderXL(nn.Module):     # TransformerEncoder
    def __init__(self, args, edge_dim):
        super().__init__()
        self.args = args

        self.num_layers = args.attn_enc_num_layers
        self.d_model = args.hidden_size            # hidden_size的超参数
        self.heads = args.attn_enc_heads                    # 多少个头
        self.d_ff = args.attn_enc_filter_size
        self.attention_dropout = args.attn_dropout
        self.max_path_distance = args.max_path_distance

        # if args.with_attn:
            # self.encoder_pe = None
            # self.encoder_pe = PositionalEncoding(
            #     dim=self.d_model,
            #     max_len=1024,        # temporary hard-code. Seems that onmt fix the denominator as 10000.0
            #     dropout=args.dropout
            #     # enc_type=args.enc_type
            # )

        self.attention_layers = nn.ModuleList(
            [SALayerXL(
                args, self.d_model, self.heads, self.d_ff, self.attention_dropout,
                edge_dim)
             for i in range(self.num_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6,elementwise_affine=True)
        self.node_in_lin = nn.Linear(self.d_model, self.d_model)
        self.node_out_lin = nn.Linear(self.d_model, self.d_model)

    def forward(self, poly_vec, src, mask,edge_index, edge_attr,ptr,batch,distances,node_paths,edge_paths):
        """adapt from onmt TransformerEncoder
            src: (b, t, h)
            lengths: (b,)
            distances: (b, t, t)
        """
        # pdb.set_trace()
        # if self.encoder_pe:
        #     emb = self.encoder_pe(src)                      # 使用了PositionalEncoding这个绝对位置编码
        # else:
        #     emb = src
        # out = self.layer_norm(src.transpose(0, 1).contiguous())
        mid = self.node_in_lin(src)
        # pdb.set_trace()
        for layer in self.attention_layers:
            mid = self.layer_norm(mid)
            mid = layer(poly_vec, mid, mask, edge_index,edge_attr,ptr,batch,distances,node_paths,edge_paths)
            # out [batch_size, max_len, hidden_size]
            # mask [batch_size, 1, max_len]
            # distances [batch_size, max_len, max_len]
        # out = self.node_out_lin(mid)

        return mid #+ src