import torch
from torch import nn

class TinyGPT(nn.Module):
    def __init__(self,vocab_size=256,d_model=128,n_head=4,n_layer=3,max_len=256,dropout=0.0):
        super().__init__(); self.max_len=max_len
        self.tok=nn.Embedding(vocab_size,d_model); self.pos=nn.Embedding(max_len,d_model)
        layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=n_head,dim_feedforward=4*d_model,dropout=dropout,batch_first=True,norm_first=True,activation='gelu')
        self.blocks=nn.TransformerEncoder(layer,num_layers=n_layer)
        self.ln=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,vocab_size,bias=False); self.head.weight=self.tok.weight
    def forward(self,idx):
        B,T=idx.shape
        if T>self.max_len: raise ValueError('sequence exceeds max_len')
        x=self.tok(idx)+self.pos(torch.arange(T,device=idx.device))[None]
        mask=torch.triu(torch.ones(T,T,device=idx.device,dtype=torch.bool),diagonal=1)
        x=self.blocks(x,mask=mask,is_causal=True)
        return self.head(self.ln(x))
