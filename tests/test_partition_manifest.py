import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from federatedscope.core.splitters.generic.partition_manifest_splitter import \
    PartitionManifestSplitter


REPO_ROOT = Path(__file__).resolve().parents[1]
NONIID_ROOT = REPO_ROOT / 'fedbiot_script' / 'fedbiot' / 'noniid'


def _load_builder():
    path = NONIID_ROOT / 'build_partition_manifests.py'
    spec = importlib.util.spec_from_file_location('partition_builder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_comparator():
    path = NONIID_ROOT / 'compare_partition_results.py'
    spec = importlib.util.spec_from_file_location('partition_comparator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_isolation_checker():
    path = NONIID_ROOT / 'verify_experiment_isolation.py'
    spec = importlib.util.spec_from_file_location('isolation_checker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(path, splits, client_num=3):
    payload = {
        'schema_version': 1,
        'dataset': 'toy',
        'partition_type': 'noniid',
        'client_num': client_num,
        'source_files': {},
        'splits': {},
    }
    for split_name, clients in splits.items():
        payload['splits'][split_name] = {
            'num_samples': sum(len(indices) for indices in clients),
            'clients': {
                str(index): values
                for index, values in enumerate(clients, start=1)
            },
        }
    path.write_text(json.dumps(payload), encoding='utf-8')


def test_manifest_splitter_replays_all_three_splits(tmp_path):
    path = tmp_path / 'manifest.json'
    _write_manifest(path, {
        'train': [[0, 3], [1], [2]],
        'val': [[2], [0], [1]],
        'test': [[1], [2], [0]],
    })
    splitter = PartitionManifestSplitter(3, str(path), strict=False)
    assert splitter(list('abcd')) == [['a', 'd'], ['b'], ['c']]
    assert splitter(list('xyz')) == [['z'], ['x'], ['y']]
    assert splitter(list('123')) == [['2'], ['3'], ['1']]
    with pytest.raises(RuntimeError):
        splitter([0, 1, 2])


def test_manifest_splitter_rejects_duplicate_or_missing_indices(tmp_path):
    path = tmp_path / 'manifest.json'
    _write_manifest(path, {
        'train': [[0], [0], [2]],
        'val': [[0], [1], [2]],
        'test': [[0], [1], [2]],
    })
    splitter = PartitionManifestSplitter(3, str(path), strict=False)
    with pytest.raises(ValueError, match='exactly once'):
        splitter([0, 1, 2])


def test_manifest_splitter_rejects_stale_source_hash(tmp_path):
    import hashlib
    source = tmp_path / 'source.json'
    source.write_text('original', encoding='utf-8')
    path = tmp_path / 'manifest.json'
    _write_manifest(path, {
        'train': [[0], [1], [2]],
        'val': [[0], [1], [2]],
        'test': [[0], [1], [2]],
    })
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['source_files'] = {
        'source': {
            'path': source.name,
            'sha256': hashlib.sha256(b'original').hexdigest(),
        }
    }
    path.write_text(json.dumps(payload), encoding='utf-8')
    PartitionManifestSplitter(3, str(path), strict=True,
                              source_root=str(tmp_path))
    source.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        PartitionManifestSplitter(3, str(path), strict=True,
                                  source_root=str(tmp_path))


def test_noniid_partition_is_deterministic_and_complete():
    builder = _load_builder()
    labels = {
        'train': ['a'] * 60 + ['b'] * 60 + ['c'] * 60,
        'val': ['a'] * 15 + ['b'] * 15 + ['c'] * 15,
        'test': ['a'] * 12 + ['b'] * 12 + ['c'] * 12,
    }
    first = builder.noniid_partition(labels, 3, 0.5, 12345)
    second = builder.noniid_partition(labels, 3, 0.5, 12345)
    assert first == second
    for split_name, clients in first.items():
        flattened = [index for client in clients for index in client]
        assert sorted(flattened) == list(range(len(labels[split_name])))
        assert all(clients)


def test_openbookqa_partition_is_explicitly_rejected(tmp_path):
    builder = _load_builder()
    with pytest.raises(ValueError, match='intentionally unsupported'):
        builder.load_dataset_labels('openbookqa', tmp_path)


def test_comparison_rejects_diagnostic_results_and_computes_drops(tmp_path):
    comparator = _load_comparator()

    def write(name, partition_type, average, worst, diagnostic=False):
        path = tmp_path / name
        path.write_text(json.dumps({
            'schema_version': 1,
            'dataset': 'cwq',
            'split': 'val',
            'match_mode': 'contains',
            'partition_type': partition_type,
            'checkpoint_distribution': partition_type,
            'diagnostic_only': diagnostic,
            'macro_average': average,
            'worst_client': worst,
            'weighted_global': average + 1,
        }), encoding='utf-8')
        return path

    iid = write('iid.json', 'iid', 60.0, 50.0)
    noniid = write('noniid.json', 'noniid', 54.0, 40.0)
    row = comparator.compare_pair('cwq', iid, noniid)
    assert row['average_absolute_drop_pp'] == -6.0
    assert row['average_relative_degradation_pct'] == 10.0
    assert row['worst_absolute_drop_pp'] == -10.0
    diagnostic = write('diagnostic.json', 'noniid', 54.0, 40.0, True)
    with pytest.raises(ValueError, match='Diagnostic-only'):
        comparator.compare_pair('cwq', iid, diagnostic)


def test_isolation_snapshot_detects_changes(tmp_path):
    checker = _load_isolation_checker()
    protected = tmp_path / 'protected.txt'
    protected.write_text('original', encoding='utf-8')
    payload = checker.snapshot([protected])
    checker.verify(payload)
    protected.write_text('changed', encoding='utf-8')
    with pytest.raises(RuntimeError, match='changed'):
        checker.verify(payload)


@pytest.mark.parametrize('dataset,expected_categories', [
    ('cwq', 4),
    ('kqapro', 12),
    ('graphquestions', 8),
])
def test_committed_manifests_are_complete(dataset, expected_categories):
    path = NONIID_ROOT / 'manifests' / (
        f'{dataset}_noniid_alpha0p5_seed12345.json')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    assert manifest['partition_type'] == 'noniid'
    assert manifest['alpha'] == 0.5
    assert manifest['seed'] == 12345
    assert manifest['client_num'] == 3
    train = manifest['splits']['train']
    categories = set()
    for histogram in train['category_histograms'].values():
        categories.update(histogram)
    assert len(categories) == expected_categories
    for split in manifest['splits'].values():
        indices = [
            index for values in split['clients'].values() for index in values
        ]
        assert sorted(indices) == list(range(split['num_samples']))
        assert all(size > 0 for size in split['client_sizes'].values())
    if dataset == 'graphquestions':
        assert min(manifest['splits']['val']['client_sizes'].values()) >= 50


@pytest.mark.parametrize('dataset,original,new', [
    ('cwq',
     'fedbiot_script/fedbiot/cwq/cwq_client_iid_webqsp_align_kg_adpt2_dp2.yaml',
     'fedbiot_script/fedbiot/noniid/cwq_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml'),
    ('kqapro',
     'fedbiot_script/fedbiot/kqapro/kqapro_client_iid_webqsp_align_kg_adpt2_dp2.yaml',
     'fedbiot_script/fedbiot/noniid/kqapro_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml'),
    ('graphquestions',
     'fedbiot_script/fedbiot/graphquestions/graphquestions_client_iid_webqsp_align_kg_adpt2_dp2.yaml',
     'fedbiot_script/fedbiot/noniid/graphquestions_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml'),
])
def test_noniid_config_only_changes_partition_and_outputs(dataset, original,
                                                           new):
    original_cfg = yaml.safe_load((REPO_ROOT / original).read_text(
        encoding='utf-8'))
    new_cfg = yaml.safe_load((REPO_ROOT / new).read_text(encoding='utf-8'))
    new_cfg.pop('outdir')
    new_cfg['federate']['save_to'] = original_cfg['federate']['save_to']
    new_cfg['data']['splitter'] = original_cfg['data']['splitter']
    new_cfg['data'].pop('splitter_args')
    new_cfg['expname'] = original_cfg['expname']
    assert new_cfg == original_cfg
