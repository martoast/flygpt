"""Resumable official MaleCNS v1.0 download; raw data are never committed."""
import subprocess
from pathlib import Path
from src.provenance import sha256, save_json

BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES = ['connectome-weights-male-cns-v1.0-minconf-0.5.feather',
         'body-annotations-male-cns-v1.0-minconf-0.5.feather',
         'body-neurotransmitters-male-cns-v1.0.feather']


def main():
    records = []
    for name in FILES:
        path = Path('data/raw') / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            part = path.with_suffix(path.suffix + '.part')
            subprocess.run(['curl', '-fL', '--retry', '5', '-C', '-', '-o', str(part), BASE + name], check=True)
            part.replace(path)
        records.append({'file': str(path), 'url': BASE + name, 'bytes': path.stat().st_size,
                        'sha256': sha256(path)})
    save_json('results/malecns_v1/download.json', {
        'dataset': 'MaleCNS v1.0 minconf 0.5', 'license': 'CC-BY',
        'source': 'https://male-cns.janelia.org/download/', 'files': records})


if __name__ == '__main__':
    main()
