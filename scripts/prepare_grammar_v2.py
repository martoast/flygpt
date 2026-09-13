"""G2: disjoint composition splits, with all lexical pieces seen in training.

Corpus construction only. This module never starts or modifies an experiment.
"""
import argparse
import itertools
import json
import random
from pathlib import Path
from src.provenance import save_json, sha256

SIZES = ('small', 'large', 'tiny', 'huge')
COLORS = ('red', 'blue', 'green', 'yellow')
ANIMALS = ('dog', 'cat', 'bird', 'fox')
VERBS = ('sees', 'follows', 'helps', 'likes')


def partition(left, right):
    """Every word occurs in every partition; pair membership is disjoint."""
    return {name: [(a, b) for i, a in enumerate(left) for j, b in enumerate(right)
                   if (j-i) % 4 in offsets]
            for name, offsets in [('train', (0, 2)), ('validation', (1,)), ('test', (3,))]}


def build(seed=20260913, evaluation_count=64):
    attributes = partition(SIZES, COLORS)
    relations = partition(ANIMALS, VERBS)

    def pool(attribute_split, relation_split):
        rows = []
        for (size, color), (animal, verb), (obj_size, obj_color), obj_animal in itertools.product(
                attributes[attribute_split], relations[relation_split], attributes['train'], ANIMALS):
            sentence = f'the {size} {color} {animal} {verb} the {obj_size} {obj_color} {obj_animal}.\n'
            rows.append({'text': sentence, 'subject_attribute': [size, color],
                         'subject_relation': [animal, verb], 'object_attribute': [obj_size, obj_color]})
        return rows

    rng = random.Random(seed)
    corpora = {'train': pool('train', 'train')}
    rng.shuffle(corpora['train'])
    for split in ('validation', 'test'):
        for name, attr, relation in [('attribute', split, 'train'),
                                      ('relation', 'train', split), ('combined', split, split)]:
            candidates = pool(attr, relation)
            if not 1 <= evaluation_count <= len(candidates):
                raise ValueError('Evaluation count exceeds distinct composition pool')
            corpora[f'{split}_{name}'] = rng.sample(candidates, evaluation_count)
    return corpora, {'attributes': attributes, 'relations': relations}


def audit(corpora, partitions):
    train = corpora['train']
    train_strings = {r['text'] for r in train}
    train_words = set(' '.join(train_strings).replace('.', '').split())
    training_attributes = {tuple(r[k]) for r in train for k in ['subject_attribute', 'object_attribute']}
    training_relations = {tuple(r['subject_relation']) for r in train}
    assert training_attributes == set(map(tuple, partitions['attributes']['train']))
    assert training_relations == set(map(tuple, partitions['relations']['train']))
    seen_strings = set()
    report = {}
    for name, rows in corpora.items():
        strings = {r['text'] for r in rows}
        assert len(strings) == len(rows), f'Duplicate sentences in {name}'
        assert not seen_strings & strings, f'Split overlap in {name}'
        seen_strings |= strings
        unseen_words = set(' '.join(strings).replace('.', '').split())-train_words
        assert not unseen_words, f'Lexical confound in {name}'
        if name != 'train':
            split, axis = name.split('_')
            for row in rows:
                attr, rel = tuple(row['subject_attribute']), tuple(row['subject_relation'])
                assert attr in map(tuple, partitions['attributes'][split if axis != 'relation' else 'train'])
                assert rel in map(tuple, partitions['relations'][split if axis != 'attribute' else 'train'])
                assert (attr not in training_attributes) == (axis != 'relation')
                assert (rel not in training_relations) == (axis != 'attribute')
        report[name] = {'sentences': len(rows), 'unique_sentences': len(strings),
                        'unseen_words': sorted(unseen_words),
                        'verbatim_training_overlap': len(strings & train_strings) if name != 'train' else None}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/raw/grammar_v2')
    args = parser.parse_args()
    corpora, partitions = build()
    report = audit(corpora, partitions)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in corpora.items():
        for suffix, content in [('txt', ''.join(r['text'] for r in rows)),
                                ('jsonl', ''.join(json.dumps(r)+'\n' for r in rows))]:
            path = root/f'{name}.{suffix}'
            if path.exists() and path.read_text() != content:
                raise RuntimeError(f'Refusing to overwrite different frozen dataset: {path}')
            path.write_text(content)
        report[name].update(bytes=(root/f'{name}.txt').stat().st_size,
                            sha256=sha256(root/f'{name}.txt'), records_sha256=sha256(root/f'{name}.jsonl'))
    save_json(root/'manifest.json', {'experiment': 'G2 — Finite-Grammar Compositional Generalization',
              'version': 2, 'seed': 20260913, 'partitions': partitions, 'splits': report,
              'generator_sha256': sha256(__file__),
              'status': 'dataset prepared and audited; no G2 models trained or evaluated',
              'rule': 'Train on allowed pairs only. Validation and test use separate unseen pair sets. All words known. Object attributes remain in training support to isolate subject composition.'})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
