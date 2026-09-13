"""Interpolated byte n-gram baselines; training bytes and evaluation match files.
These baselines expose the minimal-language task's limited complexity.
"""
from collections import Counter,defaultdict
from pathlib import Path
import math
import numpy as np
from src.provenance import manifest,save_json


def main():
    tr=Path('data/raw/grammar_v1/train.txt').read_bytes();va=Path('data/raw/grammar_v1/validation.txt').read_bytes()
    starts=np.random.default_rng(9100).integers(0,len(va)-33,size=8).tolist()
    unigram=Counter(tr);models=[]
    for order in [1,2,3,4,8]:
        counts=defaultdict(Counter)
        for t in range(1,len(tr)):
            for k in range(1,min(order,t)+1):counts[tr[t-k:t]][tr[t]]+=1
        loss=0.;correct=0;total=0
        for s in starts:
            window=va[s:s+33]
            for t in range(1,len(window)):
                prob=np.array([unigram[i]+1 for i in range(256)],float);prob/=prob.sum()
                for k in range(1,min(order,t)+1):
                    c=counts.get(window[t-k:t],{});n=sum(c.values())
                    # Fixed add-1 total interpolation mass, not selected on validation.
                    if n:prob=(np.array([c.get(i,0) for i in range(256)])+prob)/(n+1)
                loss-=math.log(prob[window[t]]);correct+=int(prob.argmax()==window[t]);total+=1
        models.append({'context_bytes':order,'ce':loss/total,'bits_per_byte':loss/total/math.log(2),'accuracy':correct/total,'n_evaluation_bytes':total})
    result=manifest({'smoothing':'recursive interpolation, 1 total pseudocount','starts':starts},['data/raw/grammar_v1/train.txt','data/raw/grammar_v1/validation.txt'])
    result.update(evidence_domain='synthetic text; conventional statistical model, no connectome',training_bytes=len(tr),models=models)
    save_json('results/malecns_v1/ngram_baselines.json',result);print(models)


if __name__=='__main__':main()
