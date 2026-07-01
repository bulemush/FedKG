import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_task_datasets():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if 'torch' not in sys.modules:
        torch_module = types.ModuleType('torch')
        utils_module = types.ModuleType('torch.utils')
        data_module = types.ModuleType('torch.utils.data')

        class Dataset:
            pass

        data_module.Dataset = Dataset
        utils_module.data = data_module
        torch_module.utils = utils_module
        sys.modules['torch'] = torch_module
        sys.modules['torch.utils'] = utils_module
        sys.modules['torch.utils.data'] = data_module

    path = repo_root / 'federatedscope/llm/dataloader/task_datasets.py'
    spec = importlib.util.spec_from_file_location('task_datasets_under_test',
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyTokenizer:
    eos_token = '</s>'
    pad_token_id = 0
    model_max_length = 128

    def __call__(self,
                 text,
                 return_tensors=None,
                 padding=None,
                 max_length=None,
                 truncation=True,
                 add_special_tokens=True):
        ids = [(ord(ch) % 89) + 1 for ch in str(text)]
        if max_length is not None and truncation:
            ids = ids[:max_length]
        return {'input_ids': ids or [self.pad_token_id]}


def _cfg():
    return SimpleNamespace(
        llm=SimpleNamespace(
            kg_adapter=SimpleNamespace(
                use=True,
                entity_vocab_size=50000,
                edge_vocab_size=50000,
                gnn_backend='paper',
                num_relations=4096,
                max_node_num_per_batch=2500,
                use_trips=True,
            ), ), )


def test_graphquestions_formatter_builds_graph_query_sg():
    task_datasets = _load_task_datasets()
    item = {
        'qid': 1,
        'question': 'find terrorist organizations involved in september 11 attacks.',
        'answer': ['al-Qaeda'],
        'function': 'none',
        'graph_query': {
            'nodes': [{
                'nid': 0,
                'node_type': 'class',
                'id': 'base.terrorism.terrorist_organization',
                'friendly_name': 'Terrorist organization',
                'question_node': 1,
            }, {
                'nid': 1,
                'node_type': 'entity',
                'id': 'en.september_11_2001_attacks',
                'friendly_name': 'September 11 attacks',
                'question_node': 0,
            }],
            'edges': [{
                'start': 0,
                'end': 1,
                'relation':
                'base.terrorism.terrorist_organization.involved_in_attacks',
            }],
        },
    }

    record = task_datasets._format_graphquestions_item(
        item,
        tokenizer=DummyTokenizer(),
        config=_cfg(),
        split_name='train')

    assert record['target'] == 'al-Qaeda'
    assert record['category'] == 'none'
    assert record['sg']['edge_index'] == [[0], [1]]
    assert len(record['sg']['edge_type']) == 1


def test_kqapro_formatter_uses_sparql_or_program_sg():
    task_datasets = _load_task_datasets()
    item = {
        'question': 'who directed inception?',
        'answer': 'Christopher Nolan',
        'program': [{
            'function': 'Find',
            'inputs': ['Inception'],
            'dependencies': [],
        }, {
            'function': 'Relate',
            'inputs': ['director'],
            'dependencies': [0],
        }],
        'sparql':
        'SELECT ?x WHERE { wd:Q25188 wdt:P57 ?x . ?x wdt:P31 wd:Q5 . }',
    }

    record = task_datasets._format_kqapro_item(
        item,
        tokenizer=DummyTokenizer(),
        config=_cfg(),
        split_name='train')

    assert record['target'] == 'Christopher Nolan'
    assert record['category'] == 'Relate'
    assert record['sg']['edge_index'][0]
    assert len(record['sg']['edge_type']) >= 1


def test_kqapro_public_test_allows_empty_target_with_program_fallback():
    task_datasets = _load_task_datasets()
    item = {
        'question': 'is inception a film?',
        'program': [{
            'function': 'Verify',
            'inputs': ['film'],
            'dependencies': [],
        }],
    }

    record = task_datasets._format_kqapro_item(
        item,
        tokenizer=DummyTokenizer(),
        config=_cfg(),
        split_name='test')

    assert record['target'] == ''
    assert record['category'] == 'Verify'
    assert record['sg']['edge_index'][0]
