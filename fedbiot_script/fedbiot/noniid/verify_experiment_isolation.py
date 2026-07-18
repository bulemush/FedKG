"""Snapshot or verify hashes of protected IID experiment inputs."""

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(paths):
    result = {}
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f'Protected file not found: {path}')
        result[path.as_posix()] = {
            'size_bytes': path.stat().st_size,
            'sha256': _sha256(path),
        }
    return {'schema_version': 1, 'files': result}


def verify(payload):
    failures = []
    for value, expected in payload.get('files', {}).items():
        path = Path(value)
        if not path.is_file():
            failures.append(f'{value}: missing')
            continue
        actual_size = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_size != expected['size_bytes'] or \
                actual_hash != expected['sha256']:
            failures.append(f'{value}: changed')
    if failures:
        raise RuntimeError('Isolation verification failed:\n' +
                           '\n'.join(failures))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    snapshot_parser = subparsers.add_parser('snapshot')
    snapshot_parser.add_argument('--output', required=True)
    snapshot_parser.add_argument('paths', nargs='+')
    verify_parser = subparsers.add_parser('verify')
    verify_parser.add_argument('--snapshot', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == 'snapshot':
        payload = snapshot(args.paths)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w', encoding='utf-8', newline='\n') as fout:
            json.dump(payload, fout, ensure_ascii=False,
                      indent=2, sort_keys=True)
            fout.write('\n')
        print(f'Wrote {output}')
    else:
        with open(args.snapshot, encoding='utf-8') as fin:
            verify(json.load(fin))
        print('Isolation verification passed.')


if __name__ == '__main__':
    main()
