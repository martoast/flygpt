import argparse, torch
from pathlib import Path
from .tinygpt import TinyGPT
from .train_language import bytes_from_file, sample_batch

def main():
 p=argparse.ArgumentParser(); p.add_argument('--text',required=True); p.add_argument('--steps',type=int,default=2000); p.add_argument('--block',type=int,default=64); p.add_argument('--batch',type=int,default=16); p.add_argument('--out',default='results/tinygpt.pt'); a=p.parse_args(); dev='cuda' if torch.cuda.is_available() else 'cpu'; data=bytes_from_file(a.text); cut=int(.9*len(data)); tr=data[:cut]
 m=TinyGPT(max_len=a.block).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
 for _ in range(a.steps):
  x,y=sample_batch(tr,a.batch,a.block,dev); z=m(x); loss=torch.nn.functional.cross_entropy(z.reshape(-1,256),y.reshape(-1)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); torch.save(m.state_dict(),a.out); print(a.out)
if __name__=='__main__': main()
