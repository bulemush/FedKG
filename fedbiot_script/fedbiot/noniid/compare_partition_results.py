"""Create the IID versus non-IID paper table from evaluation JSON files."""

import argparse
import csv
import json
from pathlib import Path


def _load(path):
    with open(path, encoding='utf-8') as fin:
        payload = json.load(fin)
    if payload.get('schema_version') != 1:
        raise ValueError(f'Unsupported summary schema in {path}.')
    if payload.get('diagnostic_only'):
        raise ValueError(f'Diagnostic-only result cannot enter table: {path}')
    return payload


def _validate_pair(dataset, iid, noniid):
    for payload, expected in [(iid, 'iid'), (noniid, 'noniid')]:
        if payload.get('partition_type') != expected:
            raise ValueError(
                f'{dataset}: expected {expected} partition, got '
                f'{payload.get("partition_type")}.')
        if payload.get('checkpoint_distribution') != expected:
            raise ValueError(
                f'{dataset}: {expected} partition was not evaluated with an '
                f'{expected} checkpoint.')
    for key in ['dataset', 'split', 'match_mode']:
        if iid.get(key) != noniid.get(key):
            raise ValueError(
                f'{dataset}: IID/non-IID {key} values do not match.')
    canonical = 'kqa_pro' if dataset == 'kqapro' else dataset
    if iid.get('dataset') != canonical:
        raise ValueError(
            f'Pair label {dataset} does not match {iid.get("dataset")}.')


def _drop(iid_value, noniid_value):
    absolute = noniid_value - iid_value
    relative = None if iid_value == 0 else \
        (iid_value - noniid_value) / iid_value * 100.0
    return absolute, relative


def _partition_disclosure(summary):
    manifest_path = summary.get('partition_manifest')
    if not manifest_path or not Path(manifest_path).is_file():
        return '', '', ''
    with open(manifest_path, encoding='utf-8') as fin:
        manifest = json.load(fin)
    split = manifest['splits'][summary['split']]
    sizes = ';'.join(
        f'{client}:{size}'
        for client, size in sorted(split['client_sizes'].items())
    )
    histograms = json.dumps(split['category_histograms'],
                            ensure_ascii=False, sort_keys=True,
                            separators=(',', ':'))
    js = json.dumps(split['diagnostics']['pairwise_js_divergence'],
                    sort_keys=True, separators=(',', ':'))
    return sizes, histograms, js


def compare_pair(dataset, iid_path, noniid_path):
    iid = _load(iid_path)
    noniid = _load(noniid_path)
    _validate_pair(dataset, iid, noniid)
    avg_drop, avg_relative = _drop(iid['macro_average'],
                                   noniid['macro_average'])
    worst_drop, worst_relative = _drop(iid['worst_client'],
                                       noniid['worst_client'])
    iid_sizes, iid_histograms, iid_js = _partition_disclosure(iid)
    noniid_sizes, noniid_histograms, noniid_js = \
        _partition_disclosure(noniid)
    return {
        'dataset': iid['dataset'],
        'split': iid['split'],
        'iid_average': iid['macro_average'],
        'noniid_average': noniid['macro_average'],
        'average_absolute_drop_pp': avg_drop,
        'average_relative_degradation_pct': avg_relative,
        'iid_worst_client': iid['worst_client'],
        'noniid_worst_client': noniid['worst_client'],
        'worst_absolute_drop_pp': worst_drop,
        'worst_relative_degradation_pct': worst_relative,
        'iid_weighted_global': iid['weighted_global'],
        'noniid_weighted_global': noniid['weighted_global'],
        'iid_client_sizes': iid_sizes,
        'noniid_client_sizes': noniid_sizes,
        'iid_category_histograms': iid_histograms,
        'noniid_category_histograms': noniid_histograms,
        'iid_pairwise_js_divergence': iid_js,
        'noniid_pairwise_js_divergence': noniid_js,
        'iid_summary': str(Path(iid_path).resolve()),
        'noniid_summary': str(Path(noniid_path).resolve()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', action='append', nargs=3,
                        metavar=('DATASET', 'IID_JSON', 'NONIID_JSON'),
                        required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-json', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [compare_pair(*pair) for pair in args.pair]
    for output in [args.output_csv, args.output_json]:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.output_json, 'w', encoding='utf-8', newline='\n') as fout:
        json.dump({'schema_version': 1, 'rows': rows}, fout,
                  ensure_ascii=False, indent=2, sort_keys=True)
        fout.write('\n')
    print(f'Wrote {args.output_csv}')
    print(f'Wrote {args.output_json}')


if __name__ == '__main__':
    main()
