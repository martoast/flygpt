import torch
from torch import nn

class SparseGraphRNN(nn.Module):
    def __init__(self,n_nodes,src,dst,input_dim,output_dim,leak=.5,edge_scale=.05,input_fraction=.25,output_fraction=.25,inner_steps=3,backend='scatter',population_seed=None,degree_normalize=False):
        super().__init__(); self.n=n_nodes; self.leak=leak; self.inner_steps=inner_steps
        if n_nodes < 2 or not 0 < input_fraction < 1 or not 0 < output_fraction < 1:
            raise ValueError('Require at least two nodes and fractions strictly between zero and one')
        if inner_steps < 1 or not 0 <= leak < 1:
            raise ValueError('Invalid inner_steps or leak')
        self.backend=backend
        if backend == 'scipy':
            import numpy as np
            order=np.lexsort((src,dst)); src=np.asarray(src)[order]; dst=np.asarray(dst)[order]
            self.csr_indices=np.asarray(src,dtype=np.int32)
            self.csr_indptr=np.r_[0,np.cumsum(np.bincount(dst,minlength=n_nodes))].astype(np.int32)
        self.register_buffer('src',torch.as_tensor(src,dtype=torch.long)); self.register_buffer('dst',torch.as_tensor(dst,dtype=torch.long))
        initial=torch.randn(len(src))*edge_scale
        if degree_normalize:
            degree=torch.bincount(self.dst,minlength=n_nodes).clamp_min(1)
            initial=initial/degree[self.dst].sqrt()
        self.edge_w=nn.Parameter(initial)
        self.bias=nn.Parameter(torch.zeros(n_nodes))
        n_in=max(1,int(n_nodes*input_fraction)); n_out=max(1,int(n_nodes*output_fraction))
        if n_in+n_out>n_nodes: raise ValueError('Input and output populations must be disjoint')
        self.n_in=n_in; self.n_out=n_out
        pop=torch.arange(n_nodes) if population_seed is None else torch.randperm(n_nodes,generator=torch.Generator().manual_seed(population_seed))
        self.register_buffer('input_nodes',pop[:n_in])
        self.register_buffer('output_nodes',pop[-n_out:])
        self.in_proj=nn.Linear(input_dim,n_in,bias=False)
        self.out_proj=nn.Linear(n_out,output_dim)
        self.gain=nn.Parameter(torch.ones(n_nodes))
    def recurrent(self,h):
        if self.backend == 'scipy':
            from .sparse_ops import CSRMultiply
            return CSRMultiply.apply(h,self.edge_w,self.csr_indices,self.csr_indptr)
        msg=h[:,self.src]*self.edge_w
        rec=torch.zeros_like(h); rec.index_add_(1,self.dst,msg)
        return rec
    def step(self,h,x=None):
        z=self.recurrent(h)+self.bias
        if x is not None:
            injected=torch.zeros_like(h); injected[:,self.input_nodes]=self.in_proj(x); z=z+injected
        new=torch.tanh(z*self.gain)
        return self.leak*h+(1-self.leak)*new
    def token_step(self,h,x):
        h=self.step(h,x)
        for _ in range(self.inner_steps-1): h=self.step(h,None)
        return h
    def readout(self,h): return self.out_proj(h[:,self.output_nodes])
    def forward(self,x,h=None):
        B,T,_=x.shape
        if h is None: h=torch.zeros(B,self.n,device=x.device,dtype=x.dtype)
        outs=[]
        for t in range(T):
            h=self.token_step(h,x[:,t]); outs.append(self.readout(h))
        return torch.stack(outs,1),h
