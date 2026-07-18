"""Build reproducible IID and semantic non-IID client partitions.

This script intentionally works on raw dataset metadata only.  It neither
tokenizes examples nor rewrites any source dataset file.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


DATASET_ALIASES = {
    'cwq': 'cwq',
    'kqa_pro': 'kqapro',
    'kqapro': 'kqapro',
    'graphquestions': 'graphquestions',
}

LABEL_RULES = {
    'cwq': 'raw.compositionality_type',
    'kqapro': 'last non-empty raw.program[*].function',
    'graphquestions': 'raw.function',
}


def _read_json(path):
    with open(path, encoding='utf-8') as fin:
        value = json.load(fin)
    if isinstance(value, dict):
        value = value.get('questions', value.get('Questions',
                                                  value.get('data', [])))
    if not isinstance(value, list):
        raise ValueError(f'Expected a list of records in {path}.')
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_entry(path):
    resolved = Path(path).resolve()
    try:
        portable_path = resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        portable_path = resolved.as_posix()
    return {
        'path': portable_path,
        'size_bytes': resolved.stat().st_size,
        'sha256': _sha256(path),
    }


def _cwq_label(record):
    value = record.get('compositionality_type')
    if value in [None, '']:
        raise ValueError('CWQ record is missing compositionality_type.')
    return str(value)


def _cwq_answers(record):
    """Mirror task_datasets._extract_cwq_answers exactly.

    In particular, an existing ``answer`` key whose value is ``None`` does
    not fall back to ``answer_id`` in the current loader.  The manifest must
    reproduce that behavior to keep its indices aligned with LLMDataset.
    """
    raw_answers = record.get('answers', record.get('answer', []))
    answers = []
    if isinstance(raw_answers, list):
        for answer in raw_answers:
            if isinstance(answer, dict):
                value = answer.get(
                    'answer',
                    answer.get('answer_id', answer.get('entity_name', None)))
            else:
                value = answer
            if value not in [None, '']:
                answers.append(str(value))
    elif raw_answers not in [None, '']:
        answers.append(str(raw_answers))
    return list(dict.fromkeys(answers))


def _filter_cwq_records(records, split_name):
    filtered = []
    for record in records:
        question = record.get(
            'question',
            record.get('machine_question',
                       record.get('webqsp_question', None)))
        if question is None:
            continue
        if not _cwq_answers(record) and split_name != 'test':
            continue
        filtered.append(record)
    return filtered


def _kqapro_label(record):
    for step in reversed(record.get('program', [])):
        if isinstance(step, dict) and step.get('function') not in [None, '']:
            return str(step['function'])
    return 'kqa_pro'


def _graphquestions_label(record):
    value = record.get('function')
    if value in [None, '']:
        raise ValueError('GraphQuestions record is missing function.')
    return str(value)


def _derived_train_val(records, labels, val_ratio, seed=12345):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    val_size = max(1, int(len(records) * val_ratio)) \
        if len(records) > 1 and val_ratio > 0 else 0
    val_indices = set(int(index) for index in indices[:val_size])
    train_labels, val_labels = [], []
    for index, label in enumerate(labels):
        (val_labels if index in val_indices else train_labels).append(label)
    return train_labels, val_labels


def load_dataset_labels(dataset, data_root):
    dataset = DATASET_ALIASES.get(dataset, dataset)
    root = Path(data_root)
    if dataset == 'openbookqa':
        raise ValueError(
            'OpenBookQA is intentionally unsupported: its current '
            'preprocessed loader assigns category 0 to every sample. '
            'Answer-position labels are not an acceptable semantic non-IID '
            'definition.')

    if dataset == 'cwq':
        paths = {
            'train': root / 'CWQ' / 'ComplexWebQuestions_train.json',
            'val': root / 'CWQ' / 'ComplexWebQuestions_dev.json',
            'test': root / 'CWQ' / 'ComplexWebQuestions_test.json',
        }
        raw_records = {
            name: _read_json(path) for name, path in paths.items()
        }
        records = {
            name: _filter_cwq_records(split_records, name)
            for name, split_records in raw_records.items()
        }
        labels = {
            name: [_cwq_label(record) for record in split_records]
            for name, split_records in records.items()
        }
        filtered_counts = {
            name: len(raw_records[name]) - len(records[name])
            for name in raw_records
        }
        notes = {
            'train': f'Applied the runtime CWQ formatter filter; excluded '
                     f'{filtered_counts["train"]} source records.',
            'val': f'Applied the runtime CWQ formatter filter; excluded '
                   f'{filtered_counts["val"]} source records.',
            'test': 'Official test is unlabeled and is not reported; applied '
                    f'the runtime question filter and excluded '
                    f'{filtered_counts["test"]} source records.',
        }
    elif dataset == 'kqapro':
        paths = {
            'train': root / 'kqa_pro' / 'train.json',
            'val': root / 'kqa_pro' / 'val.json',
            'test': root / 'kqa_pro' / 'test.json',
        }
        records = {name: _read_json(path) for name, path in paths.items()}
        labels = {
            name: [_kqapro_label(record) for record in split_records]
            for name, split_records in records.items()
        }
        notes = {
            'test': 'Official test has neither programs nor gold answers; '
                    'fallback assignment only, never report hit@1.'
        }
    elif dataset == 'graphquestions':
        paths = {
            'train_source': root / 'GraphQuestions' /
                            'graphquestions.training.json',
            'test': root / 'GraphQuestions' /
                    'graphquestions.testing.json',
        }
        train_records = _read_json(paths['train_source'])
        train_source_labels = [
            _graphquestions_label(record) for record in train_records
        ]
        train_labels, val_labels = _derived_train_val(
            train_records, train_source_labels, val_ratio=0.1 / 0.9)
        labels = {
            'train': train_labels,
            'val': val_labels,
            'test': [
                _graphquestions_label(record)
                for record in _read_json(paths['test'])
            ],
        }
        notes = {
            'val': 'Deterministically derived from the training source with '
                   'seed 12345 and val_ratio=0.1/(0.8+0.1).'
        }
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    sources = {
        name: _source_entry(path)
        for name, path in paths.items()
    }
    return dataset, labels, sources, notes


def _largest_remainder(total, proportions):
    proportions = np.asarray(proportions, dtype=float)
    if proportions.sum() <= 0:
        proportions = np.ones_like(proportions)
    proportions = proportions / proportions.sum()
    raw = proportions * total
    counts = np.floor(raw).astype(int)
    remaining = int(total - counts.sum())
    order = np.argsort(-(raw - counts), kind='stable')
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def iid_partition(lengths, client_num, seed):
    rng = np.random.RandomState(seed)
    result = {}
    for split_name in ('train', 'val', 'test'):
        indices = np.arange(lengths[split_name])
        rng.shuffle(indices)
        result[split_name] = [
            [int(index) for index in part]
            for part in np.array_split(indices, client_num)
        ]
    return result


def _partition_with_proportions(labels, proportions_by_label, client_num,
                                rng, fallback, min_client_size=0):
    clients = [[] for _ in range(client_num)]
    labels_array = np.asarray(labels, dtype=object)
    for label in sorted(set(labels)):
        label_indices = np.where(labels_array == label)[0]
        rng.shuffle(label_indices)
        proportions = proportions_by_label.get(label, fallback)
        counts = _largest_remainder(len(label_indices), proportions)
        cursor = 0
        for client_id, count in enumerate(counts):
            clients[client_id].extend(
                int(index) for index in label_indices[cursor:cursor + count])
            cursor += count
    if min_client_size > 0:
        _repair_min_client_size(clients, labels, proportions_by_label,
                                min_client_size)
    for client in clients:
        rng.shuffle(client)
    return clients


def _repair_min_client_size(clients, labels, proportions_by_label,
                            min_client_size):
    """Meet an evaluation-size floor with minimum label-prior distortion."""
    if len(labels) < len(clients) * min_client_size:
        raise ValueError('Split is too small for the requested client floor.')

    category_totals = Counter(labels)
    current = [Counter(labels[index] for index in client)
               for client in clients]
    desired = {
        label: np.asarray(proportions, dtype=float) * category_totals[label]
        for label, proportions in proportions_by_label.items()
        if label in category_totals
    }

    while min(len(client) for client in clients) < min_client_size:
        target = min(range(len(clients)), key=lambda idx: (len(clients[idx]),
                                                           idx))
        donors = [idx for idx in range(len(clients))
                  if len(clients[idx]) > min_client_size]
        if not donors:
            raise ValueError('Cannot satisfy minimum client size.')

        best = None
        for donor in donors:
            for position, sample_index in enumerate(clients[donor]):
                label = labels[sample_index]
                label_desired = desired.get(label)
                if label_desired is None:
                    delta = 0.0
                else:
                    before = (abs(current[donor][label] -
                                  label_desired[donor]) +
                              abs(current[target][label] -
                                  label_desired[target]))
                    after = (abs(current[donor][label] - 1 -
                                 label_desired[donor]) +
                             abs(current[target][label] + 1 -
                                 label_desired[target]))
                    delta = float(after - before)
                candidate = (delta, -len(clients[donor]), sample_index,
                             donor, position, label)
                if best is None or candidate < best:
                    best = candidate

        _, _, sample_index, donor, position, label = best
        clients[donor].pop(position)
        clients[target].append(sample_index)
        current[donor][label] -= 1
        current[target][label] += 1


def noniid_partition(labels_by_split, client_num, alpha, seed,
                     min_eval_sizes=None):
    rng = np.random.RandomState(seed)
    train_labels = labels_by_split['train']
    train_array = np.asarray(train_labels, dtype=object)
    train_clients = [[] for _ in range(client_num)]

    for label in sorted(set(train_labels)):
        label_indices = np.where(train_array == label)[0]
        rng.shuffle(label_indices)
        proportions = rng.dirichlet(np.repeat(alpha, client_num))
        counts = _largest_remainder(len(label_indices), proportions)
        cursor = 0
        for client_id, count in enumerate(counts):
            train_clients[client_id].extend(
                int(index) for index in label_indices[cursor:cursor + count])
            cursor += count
    for client in train_clients:
        rng.shuffle(client)
    if any(len(client) == 0 for client in train_clients):
        raise ValueError('Fixed non-IID partition produced an empty client.')

    proportions_by_label = {}
    for label in sorted(set(train_labels)):
        counts = np.array([
            sum(train_labels[index] == label for index in client)
            for client in train_clients
        ], dtype=float)
        proportions_by_label[label] = counts / counts.sum()
    fallback = np.array([len(client) for client in train_clients], dtype=float)
    fallback /= fallback.sum()

    min_eval_sizes = min_eval_sizes or {}
    result = {'train': train_clients}
    for split_name in ('val', 'test'):
        result[split_name] = _partition_with_proportions(
            labels_by_split[split_name], proportions_by_label, client_num,
            rng, fallback, min_client_size=min_eval_sizes.get(split_name, 0))
    return result


def _js_divergence(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first / first.sum()
    second = second / second.sum()
    mean = (first + second) / 2.0

    def kl(left, right):
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * kl(first, mean) + 0.5 * kl(second, mean)


def _split_payload(labels, clients):
    flattened = [index for client in clients for index in client]
    if len(flattened) != len(labels) or \
            sorted(flattened) != list(range(len(labels))):
        raise ValueError('Partition does not cover every sample exactly once.')
    categories = sorted(set(labels))
    histograms = {}
    vectors = []
    entropies = {}
    for client_id, indices in enumerate(clients, start=1):
        histogram = Counter(labels[index] for index in indices)
        histograms[str(client_id)] = dict(sorted(histogram.items()))
        vector = np.array([histogram.get(label, 0) for label in categories],
                          dtype=float)
        vectors.append(vector)
        probabilities = vector[vector > 0] / vector.sum()
        entropy = -float(np.sum(probabilities * np.log2(probabilities)))
        normalizer = math.log2(len(categories)) if len(categories) > 1 else 1
        entropies[str(client_id)] = entropy / normalizer
    js = {}
    for first in range(len(clients)):
        for second in range(first + 1, len(clients)):
            js[f'{first + 1}-{second + 1}'] = _js_divergence(
                vectors[first], vectors[second])
    return {
        'num_samples': len(labels),
        'clients': {
            str(client_id): indices
            for client_id, indices in enumerate(clients, start=1)
        },
        'client_sizes': {
            str(client_id): len(indices)
            for client_id, indices in enumerate(clients, start=1)
        },
        'category_histograms': histograms,
        'diagnostics': {
            'normalized_category_entropy': entropies,
            'pairwise_js_divergence': js,
        },
    }


def build_manifest(dataset,
                   data_root='data',
                   partition_type='noniid',
                   client_num=3,
                   alpha=0.5,
                   seed=12345):
    dataset, labels, sources, notes = load_dataset_labels(dataset, data_root)
    if partition_type == 'iid':
        partitions = iid_partition(
            {name: len(values) for name, values in labels.items()},
            client_num, seed)
    elif partition_type == 'noniid':
        min_eval_sizes = {'val': 50} if dataset == 'graphquestions' else {}
        partitions = noniid_partition(labels, client_num, alpha, seed,
                                      min_eval_sizes=min_eval_sizes)
    else:
        raise ValueError(f'Unsupported partition type: {partition_type}')

    manifest = {
        'schema_version': 1,
        'dataset': dataset,
        'partition_type': partition_type,
        'client_num': client_num,
        'seed': seed,
        'alpha': alpha if partition_type == 'noniid' else None,
        'label_rule': LABEL_RULES[dataset],
        'source_files': sources,
        'split_notes': notes,
        'constraints': {
            'min_val_client_size': 50 if dataset == 'graphquestions' and
            partition_type == 'noniid' else None,
        },
        'splits': {
            name: _split_payload(labels[name], partitions[name])
            for name in ('train', 'val', 'test')
        },
    }
    if dataset == 'graphquestions' and partition_type == 'noniid':
        min_val = min(manifest['splits']['val']['client_sizes'].values())
        if min_val < 50:
            raise ValueError(
                f'GraphQuestions validation client has only {min_val} '
                'samples; minimum is 50.')
    return manifest


def _alpha_tag(alpha):
    return str(alpha).replace('.', 'p')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+',
                        default=['cwq', 'kqapro', 'graphquestions'])
    parser.add_argument('--partition-types', nargs='+',
                        choices=['iid', 'noniid'], default=['iid', 'noniid'])
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output-dir', default=(
        'fedbiot_script/fedbiot/noniid/manifests'))
    parser.add_argument('--client-num', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for requested_dataset in args.datasets:
        dataset = DATASET_ALIASES.get(requested_dataset, requested_dataset)
        for partition_type in args.partition_types:
            manifest = build_manifest(
                dataset,
                data_root=args.data_root,
                partition_type=partition_type,
                client_num=args.client_num,
                alpha=args.alpha,
                seed=args.seed)
            if partition_type == 'noniid':
                filename = (f'{dataset}_noniid_alpha{_alpha_tag(args.alpha)}_'
                            f'seed{args.seed}.json')
            else:
                filename = f'{dataset}_iid_seed{args.seed}.json'
            output_path = output_dir / filename
            if output_path.exists() and not args.force:
                raise FileExistsError(
                    f'{output_path} exists; pass --force to replace it.')
            with open(output_path, 'w', encoding='utf-8', newline='\n') as fout:
                json.dump(manifest, fout, ensure_ascii=False,
                          indent=2, sort_keys=True)
                fout.write('\n')
            print(f'Wrote {output_path}')
            for split_name in ('train', 'val', 'test'):
                split = manifest['splits'][split_name]
                print(f'  {split_name}: {split["client_sizes"]}')


if __name__ == '__main__':
    main()
