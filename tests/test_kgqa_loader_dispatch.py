import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')


class DummyTokenized(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids


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
        if not ids:
            ids = [self.pad_token_id]
        if return_tensors == 'pt':
            return DummyTokenized(torch.tensor([ids], dtype=torch.long))
        return DummyTokenized(ids)


def _cfg(root, data_type, args):
    return SimpleNamespace(
        data=SimpleNamespace(
            root=str(root),
            type=data_type,
            splits=[0.8, 0.1, 0.1],
            args=[args],
        ),
        model=SimpleNamespace(type='dummy@huggingface_llm'),
        llm=SimpleNamespace(
            tok_len=128,
            kg_adapter=SimpleNamespace(
                use=True,
                entity_vocab_size=50000,
                edge_vocab_size=50000,
                gnn_backend='paper',
                num_relations=4096,
                max_node_num_per_batch=2500,
                use_trips=True,
            ),
        ),
    )


def _graphquestions_item(qid=1):
    return {
        'qid': qid,
        'question': f'question {qid}?',
        'answer': ['answer'],
        'function': 'none',
        'graph_query': {
            'nodes': [{
                'nid': 0,
                'node_type': 'class',
                'id': 'type.answer',
                'friendly_name': 'Answer type',
                'question_node': 1,
            }, {
                'nid': 1,
                'node_type': 'entity',
                'id': 'entity.topic',
                'friendly_name': 'topic',
                'question_node': 0,
            }],
            'edges': [{
                'start': 0,
                'end': 1,
                'relation': 'type.answer.relation',
            }],
        },
    }


def _kqapro_item(question='who directed inception?'):
    return {
        'question': question,
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
        'sparql': 'SELECT ?x WHERE { wd:Q25188 wdt:P57 ?x . }',
    }


def test_load_llm_dataset_dispatches_kqapro(tmp_path, monkeypatch):
    for name, payload in [('train.json', [_kqapro_item('train?')]),
                          ('val.json', [_kqapro_item('val?')]),
                          ('test.json', [_kqapro_item('test?')])]:
        (tmp_path / name).write_text(json.dumps(payload), encoding='utf-8')

    import federatedscope.llm.dataloader.dataloader as dataloader
    from federatedscope.llm.dataloader.dataloader import load_llm_dataset
    monkeypatch.setattr(dataloader, 'get_tokenizer',
                        lambda *args, **kwargs: (DummyTokenizer(), 0))

    cfg = _cfg(tmp_path, 'kqa_pro@llm', {
        'train_file': 'train.json',
        'val_file': 'val.json',
        'test_file': 'test.json',
    })
    train_ds, val_ds, test_ds = load_llm_dataset(cfg)[0]

    assert len(train_ds) == 1
    assert len(val_ds) == 1
    assert len(test_ds) == 1
    assert 'sg' in train_ds[0]


def test_load_llm_dataset_dispatches_graphquestions_with_derived_val(
        tmp_path, monkeypatch):
    train = [_graphquestions_item(1), _graphquestions_item(2),
             _graphquestions_item(3)]
    test = [_graphquestions_item(4)]
    (tmp_path / 'train.json').write_text(json.dumps(train), encoding='utf-8')
    (tmp_path / 'test.json').write_text(json.dumps(test), encoding='utf-8')

    import federatedscope.llm.dataloader.dataloader as dataloader
    from federatedscope.llm.dataloader.dataloader import load_llm_dataset
    monkeypatch.setattr(dataloader, 'get_tokenizer',
                        lambda *args, **kwargs: (DummyTokenizer(), 0))

    cfg = _cfg(tmp_path, 'graphquestions@llm', {
        'train_file': 'train.json',
        'test_file': 'test.json',
    })
    train_ds, val_ds, test_ds = load_llm_dataset(cfg)[0]

    assert len(train_ds) + len(val_ds) == 3
    assert len(val_ds) == 1
    assert len(test_ds) == 1
    assert 'sg' in train_ds[0]
