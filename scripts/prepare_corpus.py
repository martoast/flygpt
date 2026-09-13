"""Deterministic minimal-language corpus with independently sampled splits.
This corpus is synthetic text; graph provenance is independently MaleCNS.
"""
from pathlib import Path
import numpy as np
from src.provenance import save_json, sha256

SUBJECTS=['the cat','the dog','a bird','the child','a fox','the girl','a boy','the mouse']
VERBS=['sees','finds','likes','follows','hears','helps']
OBJECTS=['a bird','the dog','a cat','the child','a fox','the girl','a boy','the mouse']


def main():
    root=Path('data/raw/grammar_v1');root.mkdir(parents=True,exist_ok=True)
    records={}
    for split,seed,count in [('train',101,2000),('validation',202,100),('test',303,100)]:
        rng=np.random.default_rng(seed)
        text=''.join(f'{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)}.\n' for _ in range(count))
        path=root/f'{split}.txt';path.write_text(text)
        records[split]={'seed':seed,'sentences':count,'bytes':len(text.encode()),'sha256':sha256(path)}
    save_json(root/'manifest.json',{'kind':'synthetic compositional grammar, not natural-language understanding',
                                   'splits':'independent draws; sentence overlap allowed by design; no shared byte positions',
                                   'records':records})


if __name__=='__main__':main()
