import torch

def delayed_bit_batch(batch,seq_len=24,delay=12,device='cpu'):
    # input channel 0 is random bit during first half; channel 1 is recall cue at end.
    x=torch.zeros(batch,seq_len,2,device=device); y=torch.zeros(batch,dtype=torch.long,device=device)
    bit=torch.randint(0,2,(batch,),device=device); y=bit
    x[:,0,0]=bit.float()*2-1
    x[:,-1,1]=1
    return x,y
