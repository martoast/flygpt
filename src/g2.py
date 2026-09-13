"""G2-only full-sentence training and composition evaluation. G1 is untouched."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .provenance import manifest, save_json, sha256
from .train_memory import build_model
from .tinygpt import TinyGPT
from .query_flygpt import load_model

ROOT = Path('results/g2_v1')
DATA = Path('data/raw/grammar_v2')
AXES = ('attribute', 'relation', 'combined')


def rows(name):
    dataset = json.loads((DATA/'manifest.json').read_text())
    path = DATA/f'{name}.jsonl'
    assert sha256(path) == dataset['splits'][name]['records_sha256'], 'Dataset changed'
    return [json.loads(line) for line in path.read_text().splitlines()]


def batch(records, indices):
    encoded = [torch.tensor(list(records[i]['text'].encode()), dtype=torch.long) for i in indices]
    length = max(len(s)-1 for s in encoded)
    x = torch.zeros(len(encoded), length, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, s in enumerate(encoded):
        x[i, :len(s)-1] = s[:-1]
        y[i, :len(s)-1] = s[1:]
    return x, y


def logits(model, x):
    value = model(x)
    return value[0] if isinstance(value, tuple) else value


def span_indices(text, word_index):
    """Indices into NEXT-byte targets; include the word-ending separator."""
    words = text.split(' ')
    start = sum(len(w)+1 for w in words[:word_index])
    end = start+len(words[word_index])+1
    return list(range(start-1, min(end-1, len(text)-1)))


class References:
    """Additive-smoothed byte baselines, fitted only to training sentences."""
    def __init__(self, training):
        self.counts = [defaultdict(Counter) for _ in range(5)]
        for row in training:
            sentence = row['text'].encode()
            for i in range(1, len(sentence)):
                for order in range(min(i, 4)+1):
                    self.counts[order][sentence[i-order:i]][sentence[i]] += 1

    def losses(self, text, order):
        data = text.encode(); losses = []
        for i in range(1, len(data)):
            k = min(order, i)
            while k and data[i-k:i] not in self.counts[k]:
                k -= 1
            counts = self.counts[k][data[i-k:i]]
            losses.append(-math.log((counts[data[i]]+.1)/(sum(counts.values())+25.6)))
        return np.array(losses)


def evaluate(model, datasets, teacher=None, ablate=False):
    """Full prefixes; no cross-sentence hidden state; first fixed 't' is context."""
    model.eval()
    references = References(rows('train'))
    report = {}
    with torch.no_grad():
        for name, records in datasets.items():
            details = []
            for record in records:
                x, y = batch([record], [0]); value = model(x)
                z = value[0] if isinstance(value, tuple) else value
                losses = nn.functional.cross_entropy(z[0], y[0], reduction='none').numpy()
                spans = {'attribute': span_indices(record['text'], 2),
                         'relation': span_indices(record['text'], 4)}
                selected = spans['attribute'] if name.endswith('attribute') else spans['relation'] if name.endswith('relation') else spans['attribute']+spans['relation']
                u, n4 = references.losses(record['text'], 0), references.losses(record['text'], 4)
                item = {'bytes': y.numel(), 'ce': float(losses.mean()),
                        'span_bytes': len(selected), 'attribute_span_bytes': len(spans['attribute']),
                        'relation_span_bytes': len(spans['relation']), 'span_ce': float(losses[selected].mean()),
                        'attribute_span_ce': float(losses[spans['attribute']].mean()),
                        'relation_span_ce': float(losses[spans['relation']].mean()),
                        'unigram_ce': float(u.mean()), 'ngram4_ce': float(n4.mean()),
                        'unigram_span_ce': float(u[selected].mean()), 'ngram4_span_ce': float(n4[selected].mean()),
                        'subject_attribute': record['subject_attribute'], 'subject_relation': record['subject_relation']}
                if hasattr(model.core if hasattr(model,'core') else None, 'edge_w'):
                    h = value[1]
                    item['final_hidden_rms'] = float(h.square().mean().sqrt())
                    item['final_hidden_saturation'] = float((h.abs()>.95).float().mean())
                # Unrestricted symbolic grammar has 7 independent uniform four-way
                # lexical choices. This is an external oracle, not a learned model.
                item['unrestricted_grammar_oracle_ce'] = 7*math.log(4)/y.numel()
                item['unrestricted_grammar_oracle_span_ce'] = (2 if name.endswith('combined') or name=='train_familiar' else 1)*math.log(4)/len(selected)
                if teacher is not None:
                    tz = teacher(x)
                    item.update(teacher_ce=float(nn.functional.cross_entropy(tz[0], y[0])),
                                teacher_kl=float(nn.functional.kl_div(nn.functional.log_softmax(z,-1), nn.functional.softmax(tz,-1), reduction='sum')/y.numel()),
                                teacher_agreement=float((z.argmax(-1)==tz.argmax(-1)).float().mean()))
                details.append(item)
            keys = [k for k in details[0] if k.endswith('_ce') or k in ('ce', 'teacher_kl', 'teacher_agreement')]
            def weight_key(key):
                if key in ('attribute_span_ce','relation_span_ce'): return key.replace('_ce','_bytes')
                return 'span_bytes' if key in ('span_ce','unigram_span_ce','ngram4_span_ce','unrestricted_grammar_oracle_span_ce') else 'bytes'
            means = {k: float(np.average([d[k] for d in details], weights=[d[weight_key(k)] for d in details])) for k in keys}
            for key in ('final_hidden_rms','final_hidden_saturation'):
                if key in details[0]: means[key] = float(np.mean([d[key] for d in details]))
            means['bits_per_byte'] = means['ce']/math.log(2)
            report[name] = {'metrics': means, 'sentences': len(details), 'bytes': sum(d['bytes'] for d in details), 'rows': details}
        if ablate:
            weights = model.core.edge_w.detach().clone()
            try:
                model.core.edge_w.zero_()
                for name, records in datasets.items():
                    total = count = 0
                    for record in records:
                        x, y = batch([record], [0]); z = logits(model,x)
                        total += float(nn.functional.cross_entropy(z[0],y[0],reduction='sum')); count += y.numel()
                    report[name]['metrics']['zero_edge_ce'] = total/count
            finally:
                model.core.edge_w.copy_(weights)
    return report


def datasets(split, count):
    return {f'{split}_{axis}': rows(f'{split}_{axis}')[:count] for axis in AXES}


def train(condition, seed, cfg):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    is_teacher = condition == 'teacher'
    path = ROOT/f'{condition}_{seed}.pt'
    result_path = path.with_suffix('.json')
    if path.exists() != result_path.exists():
        raise RuntimeError(f'Incomplete checkpoint/manifest pair {path}; inspect before retry')
    graph = f'data/processed/controls/rewired_{777+seed}.npz' if condition.startswith('rewired') else 'data/processed/malecns.npz'
    if condition.startswith('rewired'):
        control = json.loads(Path(f'results/malecns_v1/controls/rewired_{777+seed}.json').read_text())
        assert sha256(graph) == control['graph_sha256'], 'Control differs from audited rewiring'
    start = time.perf_counter()
    model = TinyGPT(**cfg['teacher_model']) if is_teacher else build_model(graph, 'gru' if condition=='gru_ce' else 'real', seed, cfg['model'])
    teacher = None
    if not is_teacher:
        t = torch.load(ROOT/'teacher_0.pt', weights_only=True, map_location='cpu')
        teacher = TinyGPT(**t['config']).eval(); teacher.load_state_dict(t['model'])
        for parameter in teacher.parameters(): parameter.requires_grad_(False)
    lr = cfg['teacher_lr'] if is_teacher else cfg['student_lr']
    opt = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg['weight_decay'],foreach=False)
    generator = torch.Generator().manual_seed(cfg['teacher_sampling_seed'] if is_teacher else 30000+seed)
    training = rows('train'); trace = []; evaluations = []; total_bytes = 0
    steps = cfg['teacher_steps'] if is_teacher else cfg['student_steps']
    size = cfg['teacher_batch'] if is_teacher else cfg['student_batch']
    inputs = [DATA/'manifest.json', DATA/'train.jsonl', 'configs/g2_v1.json']
    if not is_teacher: inputs += [graph, ROOT/'teacher_0.pt']
    result = manifest({**cfg, 'condition': condition, 'seed': seed}, inputs)
    done = 0; previous_wall = 0
    if path.exists():
        previous = json.loads(result_path.read_text())
        assert previous['config'] == result['config'], 'Resume protocol changed'
        assert previous['source_sha256'] == result['source_sha256'], 'Resume training source changed'
        assert previous['inputs'] == result['inputs'], 'Resume inputs changed'
        assert sha256(path) == previous['checkpoint_sha256']
        if previous['complete']: return
        ck = torch.load(path,weights_only=True,map_location='cpu')
        missing, unexpected = model.load_state_dict(ck['model'],strict=False)
        assert set(missing) == ({'core.src','core.dst'} if ck['topology_buffers_external'] else set()) and not unexpected
        opt.load_state_dict(ck['optimizer']); generator.set_state(ck['data_rng'])
        done = ck['completed_steps']; total_bytes = ck['bytes_seen']
        trace = previous['trace']; evaluations = previous['evaluations']; previous_wall = previous['runtime_seconds']
        result = previous; del ck
    if not is_teacher and done==0:
        initial = evaluate(model,datasets('validation',cfg['validation_sentences_per_axis']),teacher,condition!='gru_ce')
        evaluations.append({'step':0,'bytes_seen':0,'metrics':initial,'wall_seconds':time.perf_counter()-start})
    for step in range(done+1, steps+1):
        model.train()
        indices = torch.randint(len(training), (size,), generator=generator).tolist()
        x, y = batch(training, indices); z = logits(model,x)
        ce = nn.functional.cross_entropy(z.reshape(-1,256), y.flatten())
        loss = ce; kl = None
        if condition.endswith('_kd'):
            with torch.no_grad(): tz = teacher(x)
            temperature = cfg['temperature']
            kl = nn.functional.kl_div(nn.functional.log_softmax(z/temperature,-1), nn.functional.softmax(tz/temperature,-1),reduction='none').sum(-1)
            kl = kl[y!=-100].mean()*temperature**2
            loss = cfg['alpha']*ce+(1-cfg['alpha'])*kl
        opt.zero_grad(set_to_none=True); loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
        if not torch.isfinite(loss) or not torch.isfinite(grad): raise RuntimeError('Nonfinite training state')
        opt.step(); total_bytes += int((y!=-100).sum())
        trace.append({'step': step, 'ce': float(ce.detach()), 'kl': None if kl is None else float(kl.detach()),
                      'gradient_norm': float(grad), 'bytes_seen': total_bytes})
        if step % 16 == 0: print(condition,seed,trace[-1],flush=True)
        if step == steps or (not is_teacher and step in cfg['evaluation_steps']):
            metrics = evaluate(model,datasets('validation',cfg['validation_sentences_per_axis']),teacher,not is_teacher and condition!='gru_ce')
            if not is_teacher:
                for axis, item in metrics.items():
                    initial_ce = evaluations[0]['metrics'][axis]['metrics']['ce']
                    denominator = initial_ce-item['metrics']['teacher_ce']
                    item['metrics']['normalized_teacher_gap_closure'] = (initial_ce-item['metrics']['ce'])/denominator if denominator>0 else None
            evaluations.append({'step': step, 'bytes_seen': total_bytes, 'metrics': metrics,
                                'wall_seconds': previous_wall+time.perf_counter()-start})
            checkpoint = {'model': {k:v for k,v in model.state_dict().items() if k not in ('core.src','core.dst')},
                          'config': cfg['teacher_model'] if is_teacher else cfg['model'],
                          'condition': condition, 'seed': seed, 'completed_steps': step,
                          'optimizer': opt.state_dict(), 'data_rng': generator.get_state(),
                          'bytes_seen': total_bytes, 'code_commit': result['code_commit'],
                          'protocol_sha256': sha256('configs/g2_v1.json'),
                          'topology_buffers_external': not is_teacher and condition!='gru_ce'}
            if not is_teacher: checkpoint.update(graph=graph,graph_sha256=sha256(graph))
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp = path.with_suffix('.pt.tmp'); torch.save(checkpoint,tmp); tmp.replace(path)
            result.update(trace=trace,evaluations=evaluations,complete=step==steps,
                          checkpoint_sha256=sha256(path),bytes_seen=total_bytes,
                          runtime_seconds=previous_wall+time.perf_counter()-start)
            save_json(result_path,result)
            print('VALIDATION',condition,step,{k:v['metrics']['ce'] for k,v in metrics.items()},flush=True)


def final_evaluate(condition, seed, cfg):
    torch.set_num_threads(4)
    schedule = [('teacher',cfg['teacher_seed'])]+[(c,s) for s in cfg['seeds'] for c in cfg['conditions']]
    for c,s in schedule:
        progress = ROOT/f'{c}_{s}.json'
        if not progress.exists(): raise RuntimeError('Test evaluation is locked until all scheduled training finishes')
        completed = json.loads(progress.read_text())
        if not completed.get('complete') or completed['inputs']['configs/g2_v1.json']!=sha256('configs/g2_v1.json'):
            raise RuntimeError('Test evaluation is locked: unfinished or changed protocol')
    path = ROOT/f'{condition}_{seed}.pt'
    ck = torch.load(path,weights_only=True,map_location='cpu')
    assert ck['protocol_sha256'] == sha256('configs/g2_v1.json')
    if condition == 'teacher':
        model = TinyGPT(**ck['config']); model.load_state_dict(ck['model']); teacher = None
    else:
        if condition == 'gru_ce':
            model = build_model(ck['graph'],'gru',seed,ck['config']); model.load_state_dict(ck['model'])
        else: model = load_model(ck['graph'],path,'cpu')
        t = torch.load(ROOT/'teacher_0.pt',weights_only=True,map_location='cpu')
        teacher = TinyGPT(**t['config']).eval(); teacher.load_state_dict(t['model'])
    del ck
    data = {**datasets('validation',cfg['final_validation_sentences_per_axis']),
            **datasets('test',cfg['test_sentences_per_axis'])}
    # Familiar-composition score is explicitly training resubstitution, not an IID holdout.
    data['train_familiar'] = rows('train')[:24]
    start = time.perf_counter()
    output = manifest({'condition':condition,'seed':seed,'selection':'fixed final budget; no test selection'},
                      [path,'configs/g2_v1.json',DATA/'manifest.json'])
    output.update(metrics=evaluate(model,data,teacher,condition not in ('teacher','gru_ce')),
                  runtime_seconds=time.perf_counter()-start)
    save_json(ROOT/'final'/f'{condition}_{seed}.json',output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode',choices=['train','evaluate'])
    parser.add_argument('--condition',required=True); parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args(); config = json.loads(Path('configs/g2_v1.json').read_text())
    assert args.condition in config['conditions']+['teacher']
    (train if args.mode=='train' else final_evaluate)(args.condition,args.seed,config)
