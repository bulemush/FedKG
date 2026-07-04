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


def _cfg(use_graph_query_fallback=False, max_answers_per_sample=None):
    data_args = {}
    if max_answers_per_sample is not None:
        data_args['max_answers_per_sample'] = max_answers_per_sample
    return SimpleNamespace(
        data=SimpleNamespace(args=[data_args]),
        llm=SimpleNamespace(
            kg_adapter=SimpleNamespace(
                use=True,
                entity_vocab_size=50000,
                edge_vocab_size=50000,
                gnn_backend='paper',
                num_relations=4096,
                max_node_num_per_batch=2500,
                use_trips=True,
                use_graph_query_fallback=use_graph_query_fallback,
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
        config=_cfg(use_graph_query_fallback=True),
        split_name='train')

    assert record['target'] == 'al-Qaeda'
    assert record['category'] == 'none'
    assert record['sg']['edge_index'] == [[0], [1]]
    assert len(record['sg']['edge_type']) == 1


def test_graphquestions_formatter_prefers_sparql_query_sg():
    task_datasets = _load_task_datasets()
    item = {
        'qid': 1,
        'question': 'find terrorist organizations involved in september 11 attacks.',
        'answer': ['al-Qaeda'],
        'function': 'none',
        'graph_query': {
            'nodes': [{
                'nid': idx,
                'node_type': 'entity',
                'id': f'unrelated.entity.{idx}',
                'friendly_name': f'unrelated entity {idx}',
            } for idx in range(20)],
            'edges': [{
                'start': idx,
                'end': idx + 1,
                'relation': f'unrelated.relation.{idx}',
            } for idx in range(19)],
        },
        'sparql_query':
        'PREFIX : <http://rdf.freebase.com/ns/> '
        'SELECT (?x0 AS ?value) WHERE { '
        '?x0 :type.object.type :base.terrorism.terrorist_organization . '
        'VALUES ?x1 { :en.september_11_2001_attacks } '
        '?x0 :base.terrorism.terrorist_organization.involved_in_attacks ?x1 . '
        '}',
    }

    record = task_datasets._format_graphquestions_item(
        item,
        tokenizer=DummyTokenizer(),
        config=_cfg(),
        split_name='train')

    assert len(record['sg']['node_ids']) == 3
    assert len(record['sg']['edge_type']) == 2
    assert record['sg']['edge_index'] == [[0, 0], [1, 2]]


def test_graphquestions_formatter_uses_minimal_sg_without_sparql_by_default():
    task_datasets = _load_task_datasets()
    item = {
        'qid': 1,
        'question': 'find terrorist organizations involved in september 11 attacks.',
        'answer': ['al-Qaeda'],
        'graph_query': {
            'nodes': [{
                'nid': idx,
                'node_type': 'entity',
                'id': f'unrelated.entity.{idx}',
                'friendly_name': f'unrelated entity {idx}',
            } for idx in range(20)],
            'edges': [{
                'start': idx,
                'end': idx + 1,
                'relation': f'unrelated.relation.{idx}',
            } for idx in range(19)],
        },
    }

    record = task_datasets._format_graphquestions_item(
        item,
        tokenizer=DummyTokenizer(),
        config=_cfg(),
        split_name='train')

    assert len(record['sg']['node_ids']) == 1
    assert record['sg']['edge_index'] == [[], []]


def test_graphquestions_formatter_limits_multi_answer_target():
    task_datasets = _load_task_datasets()
    item = {
        'qid': 1,
        'question': 'list matching entities.',
        'answer': [{
            'answer_argument': 'm.a0',
            'entity_name': 'a0',
        }, {
            'answer_argument': 'm.a1',
            'entity_name': 'a1',
        }, {
            'answer_argument': 'm.a2',
            'entity_name': 'a2',
        }, {
            'answer_argument': 'm.a3',
            'entity_name': 'a3',
        }],
        'sparql_query':
        'PREFIX : <http://rdf.freebase.com/ns/> '
        'SELECT (?x0 AS ?value) WHERE { '
        '?x0 :type.object.type :some.class . '
        '}',
    }

    record = task_datasets._format_graphquestions_item(
        item,
        tokenizer=DummyTokenizer(),
        config=_cfg(max_answers_per_sample=2),
        split_name='train')

    assert record['target'] == 'a0; a1'
    assert record['answer_aliases'] == [
        'm.a0', 'a0', 'm.a1', 'a1', 'm.a2', 'a2', 'm.a3', 'a3'
    ]


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
