"""Exact first-order CPU sparse autograd without per-edge activation tapes.

CSR orientation: rows are destinations; columns are sources. No N-by-N dense
matrix or E-by-B-by-T activation is constructed. SciPy computes sparse matmul;
Numba computes the derivative of each allowed edge directly.
"""
import numpy as np
import torch
from numba import njit
from scipy import sparse


@njit(cache=True)
def edge_gradient(h, grad, indices, indptr):
    result = np.empty(indices.size, dtype=h.dtype)
    for dst in range(indptr.size - 1):
        for edge in range(indptr[dst], indptr[dst + 1]):
            value = 0.0
            for batch in range(h.shape[0]):
                value += h[batch, indices[edge]] * grad[batch, dst]
            result[edge] = value
    return result


class CSRMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, weights, indices, indptr):
        if h.device.type != 'cpu':
            raise ValueError('scipy backend is CPU only')
        ctx.save_for_backward(h, weights)
        ctx.indices, ctx.indptr = indices, indptr
        matrix = sparse.csr_matrix((weights.detach().numpy(), indices, indptr), shape=(h.shape[1], h.shape[1]))
        return torch.from_numpy(np.stack([matrix.dot(row) for row in h.detach().numpy()]))

    @staticmethod
    def backward(ctx, grad):
        h, weights = ctx.saved_tensors
        g = grad.detach().contiguous().numpy()
        matrix = sparse.csr_matrix((weights.detach().numpy(), ctx.indices, ctx.indptr), shape=(h.shape[1], h.shape[1]))
        dh = torch.from_numpy(np.stack([matrix.T.dot(row) for row in g]))
        dw = torch.from_numpy(edge_gradient(h.detach().numpy(), g, ctx.indices, ctx.indptr))
        return dh, dw, None, None
