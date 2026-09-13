import argparse
from pathlib import Path
import torch
from .graphs import load_npz
from .graph_lm import GraphLanguageModel


def sample_next(logits, temperature=0.8, top_k=40):
    logits = logits / max(temperature, 1e-6)
    if top_k and top_k < logits.numel():
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = -float('inf')
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def load_model(graph_path, ckpt_path, device):
    n, src, dst, _ = load_npz(graph_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('config', {})
    model = GraphLanguageModel(
        n, src, dst,
        vocab_size=cfg.get('vocab_size', 256),
        embed_dim=cfg.get('embed_dim', 64),
        leak=cfg.get('leak', 0.8),
        edge_scale=cfg.get('edge_scale', 0.02),
        input_fraction=cfg.get('input_fraction', 0.25),
        output_fraction=cfg.get('output_fraction', 0.25),
        inner_steps=cfg.get('inner_steps', 3),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def generate(model, prompt: bytes, max_new=200, temperature=.8, top_k=40, device='cpu'):
    h = None
    if len(prompt):
        x = torch.tensor([list(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            _, h = model(x, h)
    out = bytearray(prompt)
    current = prompt[-1] if prompt else ord('\n')
    for _ in range(max_new):
        x = torch.tensor([[current]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, h = model(x, h)
        nxt = sample_next(logits[0, -1], temperature, top_k)
        out.append(nxt)
        current = nxt
    return bytes(out)


def main():
    p = argparse.ArgumentParser(description='Query a trained connectome-constrained byte language model.')
    p.add_argument('--graph', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prompt', default=None)
    p.add_argument('--max-new', type=int, default=200)
    p.add_argument('--temperature', type=float, default=.8)
    p.add_argument('--top-k', type=int, default=40)
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(a.graph, a.checkpoint, dev)

    def run(text):
        raw = generate(model, text.encode('utf-8'), a.max_new, a.temperature, a.top_k, dev)
        print(raw.decode('utf-8', errors='replace'))

    if a.prompt is not None:
        run(a.prompt)
        return
    print('FlyGPT interactive query. Ctrl-D/Ctrl-C to exit.')
    while True:
        try:
            q = input('\n> ')
        except (EOFError, KeyboardInterrupt):
            print(); break
        run(q)

if __name__ == '__main__':
    main()
