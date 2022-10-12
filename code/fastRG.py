import torch
import numpy as np
import time
import torch.nn.functional as F
from torch import LongTensor, Tensor
from typing import Generator, Iterable, List, Optional, Tuple

@torch.no_grad()
def batched_bincount(inp: Tensor, max_num: int):
    
    batch_shape, num_samples = inp.shape[:-1], inp.shape[-1]
    num_batch = np.prod(list(batch_shape))
    
    out = torch.zeros(num_batch, max_num+1, device=inp.device, dtype=torch.int)
    aux = torch.ones(num_batch, num_samples, device=inp.device, dtype=torch.int)
    out = out.scatter_add_(1, inp.view(-1, num_samples), aux).view(*batch_shape, max_num+1)
    
    del aux
    
    return out[..., :-1].int() # cut out last dummy column


###
# weights and m must have equal batch_shape
#
@torch.no_grad()
def batched_multinomial(weights: Tensor, m: Tensor, replacement: bool = False, flatten: bool = False) -> LongTensor:

    batch_shape, n_categories = weights.shape[:-1], weights.size(-1)
    num_batch = np.prod(list(batch_shape))
    num_samples = torch.max(m)
    m = m.view(-1).int()
    batch_num = len(m)

    mask = torch.tensor([0, 1]*batch_num, device=weights.device, dtype=torch.bool)

    mask = mask.repeat_interleave(torch.stack([m, num_samples-m], dim=1).view(-1)).view(num_batch, num_samples)

    flat_samples = torch.multinomial(
        input=weights.view(-1, n_categories),
        num_samples=num_samples,
        replacement=replacement,
        generator=None,
        out=None)
    
    out = flat_samples
    if flatten:
        result = out[~mask]
        return result
    else:
        out[mask] = n_categories
        return out

@torch.no_grad()
def fastRG(X, S, Y):

    N, K = X.shape
    M, _ = Y.shape
    device = X.device

    # normalize to column-stochastic
    X_sum = torch.sum(X, axis=-2, keepdim=True)  # [1, K]
    Y_sum = torch.sum(Y, axis=-2, keepdim=True)  # [1, K]
    Xn = (X / X_sum).transpose(-2,-1) # [K, N]
    Yn = (Y / Y_sum).transpose(-2,-1) # [K, N]

    # gather normalization and sample number of edges
    Sn = X_sum.transpose(-1,-2) * S * Y_sum  # [K, K]
    print(Sn.shape)
    m = torch.poisson(torch.sum(Sn, (-1,-2))).int()  # [1]

    if m == 0:
        return torch.tensor([]), torch.tensor([])
    
    # prepare indices
    src = torch.zeros(m, dtype=torch.int, device=device)  # [n_edges,]
    dst = torch.zeros(m, dtype=torch.int, device=device)  # [n_edges,]

    # sample number of edges for each cluster-cluster pair
    logits = torch.flatten(Sn)/Sn.sum()
    samples = torch.multinomial(input=logits, num_samples=m, replacement=True)
    tabUVs = torch.bincount(samples, minlength=K*K).view(K,K)

    blockDegreesU = torch.sum(tabUVs, axis=-1)  # [K,]
    blockDegreesV = torch.sum(tabUVs, axis=-2)  # [K,]

    src = batched_multinomial(Xn, blockDegreesU, replacement=True, flatten=True)
    dst = batched_multinomial(Yn, blockDegreesV, replacement=True, flatten=True)
    
    del X_sum, Y_sum, Xn, Yn
    del Sn, m, logits, samples, tabUVs
    del blockDegreesU, blockDegreesV

    return src, dst
    
###
# Inputs
#   - X: [B H N K] tensor
#   - S: [B H K K] tensor
#   - Y: [B H M K] tensor
# Outputs
#   - Mask: [BxHxN BxHxM] block-diagonal sparse binary tensor 
#   - NOTE1: Bipartite graph Mask[b,h,:,:] has expectation X[b,h]*S[b,h]*Y[b,h].T
#   - NOTE2: Similar formatting used in torch-geometric
#
@torch.no_grad()
def fastRG_batched(X, S, Y):

    B, H, N, K = X.shape
    _, _, M, _ = Y.shape
    device = X.device

    # normalize to column-stochastic
    X_sum = torch.sum(X, axis=-2, keepdim=True)  # [B, H, 1, K]
    Y_sum = torch.sum(Y, axis=-2, keepdim=True)  # [B, H, 1, K]
    Xn = (X / X_sum).transpose(-2,-1)
    Yn = (Y / Y_sum).transpose(-2,-1)
    Xn_flat = Xn.reshape(B*H*K, N)
    Yn_flat = Yn.reshape(B*H*K, M)

    # gather normalization and sample number of edges
    Sn = X_sum.transpose(-1,-2) * S * Y_sum  # [B, H, K, K]
    m = torch.poisson(torch.sum(Sn, (-1,-2))).int()  # [B, H]
    m_sum = m.sum()
    m_flat = m.view(-1)  # [B*H,]

    # prepare indices
    indices = torch.zeros(2, m_sum, dtype=torch.int, device=device)  # [2, sum(n_edges)]

    # sample number of edges for each cluster-cluster pair
    logits = torch.flatten(Sn,start_dim=2)/torch.sum(Sn,(-1,-2)).unsqueeze(2)
    sample = batched_multinomial(logits, m, replacement=True) 
    
    tabUVs = batched_bincount(sample, K*K).reshape(B, H, K, K)

    tabUVs_flat = tabUVs.reshape(B*H*K*K)

    mapping = torch.arange(0,B*H*K*K).reshape(B,H,K,K).transpose(-1,-2).reshape(B*H*K*K)

    permuted = tabUVs_flat[mapping]
    nnz = torch.nonzero(permuted).squeeze()
    nums = permuted[nnz]
    
    sorted_heads = tabUVs_flat.cumsum(dim=0)[(mapping[nnz]-1)] ### BOTTLENECK
    sorted_heads[0] = 0
    setup = sorted_heads.repeat_interleave(nums, output_size=m_sum)
    
    begin_idxes = nums.cumsum(dim=0).roll(1)
    begin_idxes[0] = 0
    result = torch.arange(nums.sum(), device=device) - begin_idxes.repeat_interleave(nums, output_size=m_sum)
    ofs = result + setup

    blockDegreesU = torch.sum(tabUVs, axis=-1)  # [B, H, K]
    blockDegreesV = torch.sum(tabUVs, axis=-2)  # [B, H, K]
    blockDegreesU_flat = blockDegreesU.view(B*H*K)
    blockDegreesV_flat = blockDegreesV.view(B*H*K)

    indices[0,:] = batched_multinomial(Xn_flat, blockDegreesU_flat, replacement=True, flatten=True)
    indices[1,:] = batched_multinomial(Yn_flat, blockDegreesV_flat, replacement=True, flatten=True)

    indices[1,:] = indices[1,ofs]

    b_flat = torch.arange(B, device=device, dtype=torch.int)[:, None].expand(B, H).reshape(B*H)
    h_flat = torch.arange(H, device=device, dtype=torch.int)[None, :].expand(B, H).reshape(B*H)
    bh_offset_flat_edgewise = torch.cat([torch.ones(e, device=device, dtype=torch.int) * (H * b + h) for b, h, e in zip(b_flat, h_flat, m_flat)])
    indices[0].add_(N*bh_offset_flat_edgewise)
    indices[1].add_(M*bh_offset_flat_edgewise)

    ## construct attention-mask and return
    return indices[0], indices[1]