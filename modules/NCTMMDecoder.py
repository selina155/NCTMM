import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.modules.MultiEmbedding import MultiEmbedding, Embedding
import math

class SpatioTemporalGraphAttentionLayer(pl.LightningModule):
    """
    X_input:(batch,num_graphs, lag + 1, num_nodes, data_dim)
    A: (batch,num_graphs, lag+1, num_nodes, num_nodes)
    output: (batch,num_graphs, num_nodes, data_dim)
    """

    def __init__(self,
                 data_dim: int,
                 lag: int,
                 num_nodes: int,
                 num_graphs: int,
                 embedding_dim: int,
                 num_heads: int = 4,
                 dropout=0
                 ):
        super(SpatioTemporalGraphAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.lag = lag
        self.data_dim = data_dim
        self.num_heads = num_heads

        assert self.embedding_dim % self.num_heads == 0
        self.head_dim = self.embedding_dim // self.num_heads

        self.proj_query = nn.Linear(self.data_dim + self.embedding_dim, self.embedding_dim, bias=False)
        self.proj_key = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.proj_value = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.att_q=nn.Parameter(torch.Tensor(1,1,1,num_heads,self.head_dim))
        self.att_k=nn.Parameter(torch.Tensor(1,1,1,num_heads,self.head_dim))
        nn.init.xavier_uniform_(self.att_q)
        nn.init.xavier_uniform_(self.att_k)


        self.leakyrelu = nn.LeakyReLU(0.2)
        self.layernorm = nn.LayerNorm(self.data_dim + self.embedding_dim)
        self.agg_norm = nn.LayerNorm(self.embedding_dim)
        self.temporal_fc=nn.Linear(self.lag,1)

        # self.layernorm = nn.LayerNorm(self.in_features)
        self.input_mapping = self._feature_mapping(input_dim=self.data_dim + self.embedding_dim,
                                                   output_dim=self.embedding_dim)
        # self.t_feat_mapping = self._feature_mapping(input_dim=2 * self.embedding_dim)
        self.out_mapping = self._feature_mapping(input_dim=2 * self.embedding_dim, use_activation=True)
        # single graph setting
        if self.num_graphs is None:
            self.embeddings = Embedding(num_nodes=num_nodes, lag=self.lag, embedding_dim=self.embedding_dim)
        # mixture graph setting
        else:
            self.embeddings = MultiEmbedding(num_nodes=self.num_nodes,
                                             lag=self.lag,
                                             num_graphs=self.num_graphs,
                                             embedding_dim=self.embedding_dim)

    def forward(self, h, adj):
        batch, num_graphs, L, num_nodes, data_dim = h.shape
        E = self.embeddings.get_embeddings()
        E = E.unsqueeze(0).expand((h.shape[0], -1, -1, -1, -1))
        X_in = torch.cat((h, E), dim=-1)
        X_in = self.layernorm(X_in)
        X_enc = self.input_mapping(X_in)
        K = self.proj_key(X_enc).view(batch, num_graphs, L, num_nodes, self.num_heads, self.head_dim)
        V = self.proj_value(X_enc).view(batch, num_graphs, L, num_nodes, self.num_heads, self.head_dim)
        E_target = E[:, :, -1:, ...]
        X_prev = h[:, :, -2:-1, ...]
        X_in_query = self.layernorm(torch.cat([X_prev, E_target], dim=-1))
        Q = self.proj_query(X_in_query).view(batch, num_graphs, 1, num_nodes, self.num_heads, self.head_dim)

        f_q=(Q*self.att_q).sum(dim=-1)
        f_k=(K*self.att_k).sum(dim=-1)
        logits=self.leakyrelu(f_q.unsqueeze(4)+f_k.unsqueeze(3))
        logits = logits / math.sqrt(self.head_dim)                        
        degree=adj.sum(dim=(2,4)).unsqueeze(-1)

        masked_logits = logits + (1.0 - adj.unsqueeze(-1)) * -1e9
        masked_logits_flat = masked_logits.permute(0, 1, 3, 5, 2, 4).reshape(batch, num_graphs, num_nodes,
                                                                             self.num_heads, -1)
        attn_weights = F.softmax(masked_logits_flat, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        V_flat = V.permute(0, 1, 4, 2, 3, 5).reshape(batch, num_graphs, self.num_heads, -1, self.head_dim)
        X_agg = torch.einsum("bkthl, bkhld -> bkthd", attn_weights, V_flat).reshape(batch, num_graphs, num_nodes, self.embedding_dim)
        X_agg=self.agg_norm(X_agg*(degree+1e-6))
        pred = self.out_mapping(torch.cat((X_agg, E[:, :, -1, ...]), dim=-1))
        return pred,attn_weights
       

    def _feature_mapping(self, input_dim, hidden_dim=128, output_dim=1, use_activation=False):
        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_activation:
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def _graphsoftmax(self, attention, A):
        causal_att = attention * A
        causal_att = causal_att.float()
        mask = (causal_att != 0).float()
        softmax_causal_att = F.softmax(causal_att + (1.0 - mask) * -1e9, dim=-1)
        masked_att = softmax_causal_att * mask
        masked_att = F.dropout(masked_att, self.dropout, training=self.training)
        return masked_att

    def _prepare_attentional_mechanism_input(self, Wh, t_proxy):
        N, K, T, D, C = Wh.shape
        Wh = Wh.reshape(N * K, T, D, C)
        t_proxy = t_proxy.reshape(N * K, T, D, C)
        Wh_neighbor = Wh.unsqueeze(2).expand(-1, -1, D, -1, -1)
        Wh_current = t_proxy.view(N * K, T, 1, D, C).expand(-1, -1, D, -1, -1)
        Wh_current = Wh_current.permute(0, 1, 3, 2, 4)
        a_input = torch.cat([Wh_neighbor, Wh_current], dim=-1)
        return a_input.view(N, K, T, D, D, 2 * C)


class SpatiotemporalGAT(nn.Module):
    """
    X_input:(batch,num_graphs, lag + 1, num_nodes, data_dim)
    A: (batch,num_graphs, lag+1, num_nodes, num_nodes)
    output: (batch,num_graphs, num_nodes, data_dim)
    Args:
    num_nodes: the number of the nodes
    num_graphs: the number of the causal graphs
    num_layers: the number of spatiotemporal graph attention layers
    num_heads: the number of attention heads
    embedding_dim: dimension of the learnable embedding
    """

    def __init__(self, data_dim: int,
                 lag: int,
                 num_nodes: int,
                 num_graphs: int = None,
                 num_layers: int = 1,
                 num_heads: int = 4,
                 embedding_dim: int = 32,
                 dropout=0):
        super(SpatiotemporalGAT, self).__init__()
        self.embedding_dim = embedding_dim
        self.data_dim = data_dim
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.lag = lag
        self.num_graphs = num_graphs
        self.num_layers = num_layers
        self.input_dim = self.data_dim
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                SpatioTemporalGraphAttentionLayer(
                    data_dim=self.data_dim,
                    lag=self.lag,
                    num_nodes=self.num_nodes,
                    num_graphs=self.num_graphs,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
            )
            
        self.BN = nn.BatchNorm2d(self.data_dim)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, A):
        if x.dim()==5:
            B,S,L,D,F=x.shape
            x=x.reshape(B*S,L,D,F)
        x = self.BN(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        if self.num_graphs is None:
            adj = A.unsqueeze(1)
            h = x.unsqueeze(1)
        else:
            adj = A.unsqueeze(0).expand((x.shape[0], -1, -1, -1, -1))
            h = x.unsqueeze(1).expand(
                (-1, self.num_graphs, -1, -1, -1))
        batch, K, L, num_nodes, data_dim = h.shape
        adj = adj.flip([2])
        out = None
        for layer in self.layers:
            out,att = layer(h, adj)
            if self.num_layers > 1:
                h_prev = h[:, :, :-1, :, :]
                h_curr = out.view(batch, K, 1, num_nodes, data_dim)
                h = torch.cat([h_prev, h_curr], dim=1)
        out = out.view(batch, K, num_nodes, data_dim)
        return out

