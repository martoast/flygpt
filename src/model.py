import torch
from torch import nn

class SparseGraphRNN(nn.Module):
    def __init__(self,n_nodes,src,dst,input_dim,output_dim,leak=.5,edge_scale=.05,input_fraction=.25,output_fraction=.25,inner_steps=3):
        super().__init__(); self.n=n_nodes; self.leak=leak; self.inner_steps=inner_steps
        self.register_buffer('src',torch.as_tensor(src,dtype=torch.long)); self.register_buffer('dst',torch.as_tensor(dst,dtype=torch.long))
        self.edge_w=nn.Parameter(torch.randn(len(src))*edge_scale)
        self.bias=nn.Parameter(torch.zeros(n_nodes))
        n_in=max(1,int(n_nodes*input_fraction)); n_out=max(1,int(n_nodes*output_fraction))
        if n_in+n_out>n_nodes: n_out=max(1,n_nodes-n_in)
        self.n_in=n_in; self.n_out=n_out
        self.register_buffer('input_nodes',torch.arange(0,n_in,dtype=torch.long))
        self.register_buffer('output_nodes',torch.arange(n_nodes-n_out,n_nodes,dtype=torch.long))
        self.in_proj=nn.Linear(input_dim,n_in,bias=False)
        self.out_proj=nn.Linear(n_out,output_dim)
        self.gain=nn.Parameter(torch.ones(n_nodes))
    def recurrent(self,h):
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
