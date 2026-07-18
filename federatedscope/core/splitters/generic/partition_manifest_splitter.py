import hashlib
import json
import os

from federatedscope.core.splitters import BaseSplitter


class PartitionManifestSplitter(BaseSplitter):
    """Replay a precomputed train/val/test client partition manifest."""

    _SPLIT_ORDER = ('train', 'val', 'test')

    def __init__(self,
                 client_num,
                 manifest,
                 strict=True,
                 source_root='.'):
        super(PartitionManifestSplitter, self).__init__(client_num)
        self.manifest = manifest
        self.manifest_path = os.path.abspath(os.path.expanduser(manifest))
        self.strict = bool(strict)
        self.source_root = os.path.abspath(os.path.expanduser(source_root))
        self._call_index = 0

        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(
                f'Partition manifest not found: {self.manifest_path}')
        with open(self.manifest_path, encoding='utf-8') as fin:
            self.manifest_data = json.load(fin)
        self._validate_header()
        if self.strict:
            self._verify_sources()

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as fin:
            for chunk in iter(lambda: fin.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_header(self):
        if self.manifest_data.get('schema_version') != 1:
            raise ValueError('Only partition manifest schema_version=1 is '
                             'supported.')
        manifest_clients = int(self.manifest_data.get('client_num', -1))
        if manifest_clients != self.client_num:
            raise ValueError(
                f'Manifest has {manifest_clients} clients but config requests '
                f'{self.client_num}.')
        missing = [
            name for name in self._SPLIT_ORDER
            if name not in self.manifest_data.get('splits', {})
        ]
        if missing:
            raise ValueError(f'Manifest is missing splits: {missing}')

    def _verify_sources(self):
        for source in self.manifest_data.get('source_files', {}).values():
            relpath = source.get('path')
            expected = source.get('sha256')
            if not relpath or not expected:
                raise ValueError('Each source file must contain path and '
                                 'sha256 in strict mode.')
            path = relpath if os.path.isabs(relpath) else os.path.join(
                self.source_root, relpath)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f'Manifest source file not found: {path}')
            actual = self._sha256(path)
            if actual != expected:
                raise ValueError(
                    f'Source hash mismatch for {path}: expected {expected}, '
                    f'got {actual}. Refusing to replay a stale partition.')

    def _indices_for_split(self, split_name, length):
        split = self.manifest_data['splits'][split_name]
        expected_length = int(split.get('num_samples', -1))
        if expected_length != length:
            raise ValueError(
                f'{split_name} dataset length is {length}, but manifest '
                f'expects {expected_length}.')

        clients = split.get('clients', {})
        indices = []
        for client_id in range(1, self.client_num + 1):
            key = str(client_id)
            if key not in clients:
                raise ValueError(
                    f'{split_name} manifest is missing client {client_id}.')
            client_indices = [int(index) for index in clients[key]]
            if not client_indices:
                raise ValueError(
                    f'{split_name} client {client_id} has no samples.')
            indices.append(client_indices)

        flattened = [index for client in indices for index in client]
        if len(flattened) != length or sorted(flattened) != list(range(length)):
            raise ValueError(
                f'{split_name} manifest indices must cover [0, {length}) '
                'exactly once.')
        return indices

    def __call__(self, dataset, prior=None, **kwargs):
        if self._call_index >= len(self._SPLIT_ORDER):
            raise RuntimeError('PartitionManifestSplitter was called more '
                               'than three times.')
        split_name = self._SPLIT_ORDER[self._call_index]
        self._call_index += 1
        indices = self._indices_for_split(split_name, len(dataset))

        try:
            from torch.utils.data import Dataset, Subset
        except ImportError:
            Dataset, Subset = None, None
        if Dataset is not None and isinstance(dataset, Dataset):
            return [Subset(dataset, client_indices)
                    for client_indices in indices]
        return [[dataset[index] for index in client_indices]
                for client_indices in indices]
