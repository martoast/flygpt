"""Queue seed-zero rewiring behind baseline A; replay its completed budgets.

This first pair is exploratory. Five paired seeds remain necessary before any
topology-advantage inference. No decisions depend on control performance.
"""
import json
import os
import time
from pathlib import Path
from scripts.run_screen import run
from scripts.trajectory import inspect
from src.provenance import save_json, sha256

ROOT = Path('results/malecns_v1')


def main():
    status = ROOT/'target/rewired_queue.json'
    save_json(status, {'status': 'waiting for baseline conclusion', 'seed': 0,
                      'policy': '64 CE pilot then identical 512/1024/2048 continuation stages completed by real; no control-dependent stopping'})
    while not (ROOT/'target/conclusion.json').exists():
        time.sleep(10)
    final_step = json.loads((ROOT/'target/conclusion.json').read_text())['final_step']
    original = json.loads((ROOT/'target/real_0.json').read_text())
    for source in ['results/malecns_v1/teacher.pt', 'data/raw/grammar_v1/train.txt',
                   'data/raw/grammar_v1/validation.txt']:
        assert sha256(source) == original['inputs'][source]
    graph = 'data/processed/controls/rewired_777.npz'
    control = json.loads((ROOT/'controls/rewired_777.json').read_text())
    assert sha256(graph) == control['graph_sha256']
    pilot = ROOT/'language/rewired_0.json'
    if not pilot.exists():
        save_json(status, {'status': 'training matched CE pilot', 'seed': 0, 'final_step': final_step})
        run(['-m', 'src.train_language', '--graph', graph, '--condition', 'rewired',
             '--seed', '0', '--steps', '64', '--block', '8', '--batch', '1',
             '--lr', '.003', '--out', str(pilot)], pilot.with_suffix('.log'))
    p = json.loads(pilot.read_text())
    assert sha256(p['checkpoint']['path']) == p['checkpoint']['sha256']
    assert p['config']['seed'] == 0 and p['config']['steps'] == 64
    assert p['config']['model'] == original['config']['model']
    out = ROOT/'target/rewired_0.json'
    cumulative = p['runtime_seconds']
    for step in [s for s in [512, 1024, 2048] if s <= final_step]:
        snapshot = ROOT/'target/snapshots'/f'rewired_0_step_{step:04d}.pt'
        diagnostic = ROOT/'target/rewired_trajectory'/f'step_{step:04d}.json'
        if diagnostic.exists():
            cumulative = json.loads(diagnostic.read_text())['training_wall_seconds']
            continue
        if snapshot.exists():
            raise RuntimeError('Incomplete archived stage requires inspection before restart')
        if out.exists() and json.loads(out.read_text())['completed_steps'] >= step:
            raise RuntimeError('Unarchived completed stage requires inspection before restart')
        save_json(status, {'status': 'training matched continuation', 'seed': 0,
                          'target_step': step, 'final_step': final_step})
        command = ['-m', 'scripts.train_target', '--graph', graph,
                   '--initial', p['checkpoint']['path'], '--condition', 'rewired',
                   '--seed', '0', '--steps', str(step), '--out', str(out)]
        if step > 512:
            command.append('--resume')
        run(command, ROOT/f'target/rewired_extension_{step}.log')
        result = json.loads(out.read_text())
        assert result['completed_steps'] == step
        cumulative += result['runtime_seconds_this_session']
        os.link(out.with_suffix('.pt'), snapshot)
        inspect(snapshot, diagnostic, step, cumulative, result['trace'], graph, 'rewired')
    pairs = []
    for path in sorted((ROOT/'target/rewired_trajectory').glob('step_*.json')):
        rewired = json.loads(path.read_text())
        real = json.loads((ROOT/'target/trajectory'/path.name).read_text())
        pairs.append({'step': real['continuation_step'], 'real_ce': real['validation_nats_per_byte'],
                      'rewired_ce': rewired['validation_nats_per_byte'],
                      'A_bio_negative_ce': rewired['validation_nats_per_byte']-real['validation_nats_per_byte']})
    save_json(ROOT/'target/paired_seed0.json', {'pairs': pairs, 'seeds': [0],
              'interpretation': 'One exploratory pair; no topology-superiority conclusion. Equal updates/bytes, runtime reported separately.'})
    save_json(status, {'status': 'complete', 'seed': 0, 'final_step': final_step})


if __name__ == '__main__':
    main()
