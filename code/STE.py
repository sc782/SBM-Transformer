import torch
import torch.nn as nn
import torch.nn.functional as F

class SampleGraphSparseGraph(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        A = torch.bernoulli(torch.clamp(input+0.01, min=0, max=1)).requires_grad_(True)
        ctx.save_for_backward(A)
        return A
        
    def backward(ctx, grad_output):
        A, = ctx.saved_tensors
        return F.hardtanh(A*grad_output)