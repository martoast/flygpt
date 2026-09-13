from torch import nn
from .model import SparseGraphRNN
class GraphLanguageModel(nn.Module):
    def __init__(self,n_nodes,src,dst,vocab_size=256,embed_dim=64,leak=.8,edge_scale=.02,input_fraction=.25,output_fraction=.25,inner_steps=3):
        super().__init__(); self.embed=nn.Embedding(vocab_size,embed_dim)
        self.core=SparseGraphRNN(n_nodes,src,dst,embed_dim,vocab_size,leak=leak,edge_scale=edge_scale,input_fraction=input_fraction,output_fraction=output_fraction,inner_steps=inner_steps)
    def forward(self,idx,h=None): return self.core(self.embed(idx),h)
