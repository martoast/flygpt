"""Prospective G2b deterministic transduction data; never launches training."""
import json
from pathlib import Path
import random
from src.provenance import save_json, sha256

SIZES = 'small large tiny huge short tall wide slim'.split()
COLORS = 'red blue green yellow black white pink gray'.split()
ANIMALS = 'dog cat bird fox wolf bear mole hare'.split()
VERBS = 'sees finds likes follows hears helps watches chases'.split()


def partition(left,right):
    offsets = {'train':set(range(5)), 'validation':{5}, 'qualification':{6}, 'test':{7}}
    return {name:[(a,b) for i,a in enumerate(left) for j,b in enumerate(right) if (j-i)%8 in values]
            for name,values in offsets.items()}


def build():
    rng = random.Random(20260914)
    attributes = partition(SIZES,COLORS); relations = partition(ANIMALS,VERBS)
    data = {}
    for split in ('train','validation','qualification','test'):
        for axis in (('familiar',) if split=='train' else ('attribute','relation','combined')):
            attr = attributes[split] if axis in ('attribute','combined') else attributes['train']
            rel = relations[split] if axis in ('relation','combined') else relations['train']
            cases = set(); count = 4096 if split=='train' else 64
            while len(cases)<count:
                size,color = rng.choice(attr); animal,verb = rng.choice(rel)
                osize,ocolor = rng.choice(attributes['train'])
                # Prevent the training answer's reversed role from leaking a
                # held-out animal/verb pair through the original object.
                oanimal = rng.choice([a for a,v in relations['train'] if v==verb])
                cases.add((size,color,animal,verb,osize,ocolor,oanimal))
            records = []
            for case in sorted(cases):
                size,color,animal,verb,osize,ocolor,oanimal = case
                subject=f'the {size} {color} {animal}'; obj=f'the {osize} {ocolor} {oanimal}'
                for operation in ('swap','roles'):
                    prompt=f'{operation}|{subject} {verb} {obj}.\n>'
                    response=f'{obj} {verb} {subject}.\n' if operation=='swap' else f'{subject} | {verb} | {obj}.\n'
                    records.append({'prompt':prompt,'response':response,'response_start_byte':len(prompt.encode()),
                                    'operation':operation,'subject_attribute':[size,color],
                                    'subject_relation':[animal,verb],'object_attribute':[osize,ocolor],
                                    'object_relation':[oanimal,verb]})
            rng.shuffle(records); data[f'{split}_{axis}'] = records
    return data, {'attributes':attributes,'relations':relations}


def audit(data,partitions):
    training = data['train_familiar']
    attrs = {tuple(r[k]) for r in training for k in ('subject_attribute','object_attribute')}
    rels = {tuple(r[k]) for r in training for k in ('subject_relation','object_relation')}
    assert attrs==set(partitions['attributes']['train'])
    assert rels==set(partitions['relations']['train'])
    seen_prompts=set(); seen_responses=set(); results={}
    for name,records in data.items():
        split,axis=name.split('_'); prompts={r['prompt'] for r in records}; responses={r['response'] for r in records}
        assert len(prompts)==len(records) and len(responses)==len(records)
        assert not prompts & seen_prompts and not responses & seen_responses
        seen_prompts |= prompts; seen_responses |= responses
        for row in records:
            assert len((row['prompt']+row['response']).encode())<=128
            assert row['response_start_byte']==len(row['prompt'].encode())
            for field,kind in [('subject_attribute','attributes'),('subject_relation','relations')]:
                holdout = split!='train' and (axis=='combined' or (axis=='attribute')==(kind=='attributes'))
                assert tuple(row[field]) in partitions[kind][split if holdout else 'train']
            assert tuple(row['object_attribute']) in attrs
            assert tuple(row['object_relation']) in rels
        results[name]={'records':len(records),'unique_prompts':len(prompts),'unique_responses':len(responses),
                       'prompt_overlap_with_other_splits':0,'answer_overlap_with_other_splits':0,
                       'max_sequence_bytes':max(len((r['prompt']+r['response']).encode()) for r in records)}
    return results


def main():
    data,partitions=build(); results=audit(data,partitions)
    root=Path('data/raw/grammar_g2b_v1'); root.mkdir(parents=True,exist_ok=True)
    for name,records in data.items():
        text=''.join(json.dumps(r)+'\n' for r in records); p=root/f'{name}.jsonl'
        if p.exists() and p.read_text()!=text: raise RuntimeError('Refusing to overwrite different G2b dataset')
        p.write_text(text); results[name]['sha256']=sha256(p)
    save_json(root/'manifest.json',{'experiment':'G2b — Teacher-qualified compositional transduction',
              'status':'data only; no trained result','seed':20260914,'partitions':partitions,
              'audit':results,'generator_sha256':sha256(__file__),
              'qualification_protocol_sha256':sha256('configs/g2b_qualification_v1.json')})
    print({k:v['records'] for k,v in results.items()})


if __name__=='__main__':main()
