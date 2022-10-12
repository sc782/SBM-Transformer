import torch
import torch.nn as nn
import math
import json
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from fastRG import fastRG
from STE import *
import time
from dgl.nn.functional import edge_softmax
import dgl.function as fn
import dgl.ops as DF
import dgl


@torch.no_grad()
def block_diag(m):
    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n, device=m.device).unsqueeze(-2), d - 3, 1)
    out = (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )
    
    return out

@torch.no_grad()
def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))
    
class SBMAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]
        self.type = config["sbm_type"]
        self.num_head = config["num_head"]
        self.num_clusters = config["num_clusters"]
        
        self.clusters = nn.Parameter(torch.empty(self.num_head, self.num_clusters, self.head_dim))

        self.proj = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, self.head_dim)
        )

        nn.init.kaiming_normal_(self.clusters)
    
    def forward(self, Q, K, V, mask):
        
        b, h, n, d = Q.shape
        _, _, m, _ = V.shape
        k = self.num_clusters

        dist = torch.matmul(self.clusters, torch.transpose(self.clusters, -1, -2)) 
            
        # Activation for inter-cluster correlations
        S = nn.Softmax(dim=-1)(dist.reshape(self.num_head, self.num_clusters**2)).reshape(self.num_head,k,k).unsqueeze(0).repeat((b,1,1,1))
        
        Qhat = nn.Sigmoid()(torch.matmul(self.proj(Q), self.clusters.transpose(-1, -2))) # Original
        Khat = nn.Sigmoid()(torch.matmul(self.proj(K), self.clusters.transpose(-1, -2)))

        
        if self.type == 'fastRG':
            
            src, dst = fastRG(block_diag(Qhat.view(b*h,n,k)), 
                              block_diag(S.view(b*h,k,k)), 
                              block_diag(Khat.view(b*h,n,k)))

            graph = dgl.graph((src, dst), num_nodes=b*h*n)
            
            graph.dstdata.update({'v':V.reshape(b*h*n, d)})
            edata = DF.v_dot_u(graph, Q.reshape(b*h*n, d), K.reshape(b*h*n, d))

            # Compute probs of sampled edges
            eprobs = DF.u_dot_v(graph, 
                                Qhat.reshape(b*h*n, k), 
                                torch.matmul(Khat, S.transpose(-1,-2)).reshape(b*h*n,k))

            # Pass through STE
            edata = EdgeSample.apply(eprobs, edata)

            # Compute attention per edge
            graph.edata['a'] = edge_softmax(graph, edata, norm_by='dst')

            # Attention via Message Passing
            graph.update_all(fn.u_mul_e('v','a','m'), fn.sum('m', 'y'))
            
            del src, dst
            
            return graph.dstdata['y'].view(b,h,n,d), torch.sum(torch.ones_like(eprobs))/(b*h*(n**2))
            
        else:
            
            expA = torch.matmul(Qhat, torch.matmul(S, Khat.transpose(-1, -2)))
            
            graph = SampleGraphSparseGraph.apply(expA)
            
            dot = torch.matmul(Q, torch.transpose(K, -2, -1))
            dot = dot / math.sqrt(self.head_dim)
        
            dot.masked_fill_(mask[:,None,None,:] == 0, float('-inf')) # first apply user-provided mask
            
            attn = F.normalize(nn.Softmax(dim=-1)(dot)*graph, p=1, dim=-1)
            X = torch.matmul(self.drop_attn(attn), V) # apply dropout then matmul
            sparsity = torch.sum(graph, dim=(0,-1,-2))/(b*n*m) # head-wise sparsity
            
            return X, sparsity