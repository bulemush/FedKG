import json
from types import SimpleNamespace

import pytest

pytest.importorskip('torch')

from federatedscope.llm.dataloader.task_datasets import (
    _format_grailqa_item,
)


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
            torch = pytest.importorskip('torch')
            return DummyTokenized(torch.tensor([ids], dtype=torch.long))
        return DummyTokenized(ids)


def _cfg(root='data/'):
    return SimpleNamespace(
        data=SimpleNamespace(
            root=str(root),
            type='grailqa@llm',
            splits=[0.8, 0.1, 0.1],
            args=[{
                'train_file': 'train.json',
                'val_file': 'dev.json',
                'test_file': 'test.json',
            }],
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
                use_graph_query_fallback=True,
            ),
        ),
    )


def _grailqa_train_item():
    return {
        'qid': 1,
        'question': 'what is the role of opera designer gig?',
        'answer': [{
            'answer_type': 'Entity',
            'answer_argument': 'm.0b787yg',
            'entity_name': 'Set Designer',
        }],
        'level': 'i.i.d.',
        'domains': ['opera'],
        'graph_query': {
            'nodes': [{
                'nid': 0,
                'node_type': 'class',
                'id': 'opera.opera_designer_role',
                'friendly_name': 'Opera Designer Role',
                'question_node': 1,
            }, {
                'nid': 1,
                'node_type': 'entity',
                'id': 'm.0pm2fgf',
                'friendly_name': 'opera designer gig',
                'question_node': 0,
            }],
            'edges': [{
                'start': 1,
                'end': 0,
                'relation': 'opera.opera_designer_gig.design_role',
            }],
        },
    }


def test_grailqa_formatter_builds_labeled_kg_record():
    record = _format_grailqa_item(
        _grailqa_train_item(),
        tokenizer=DummyTokenizer(),
        config=_cfg(),
        split_name='train',
    )

    assert record['context'] == \
        'Question: what is the role of opera designer gig?\nAnswer:'
    assert record['target'] == 'Set Designer'
    assert 'm.0b787yg' in record['answer_aliases']
    assert record['category'] == 'i.i.d.'
    assert record['sg']['node_ids']
    assert record['sg']['edge_index'] == [[1], [0]]
    assert len(record['sg']['edge_type']) == 1


def test_grailqa_public_test_formatter_allows_unlabeled_records():
    record = _format_grailqa_item(
        {
            'qid': 'test-0',
            'question': 'flagstaff observatory houses what telescope?',
        },
        tokenizer=DummyTokenizer(),
        config=_cfg(),
        split_name='test',
    )

    assert record['target'] == ''
    assert record['category'] == 'grailqa'
    assert record['sg']['edge_index'] == [[], []]


def test_load_llm_dataset_dispatches_grailqa(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    train = [_grailqa_train_item()]
    dev = [_grailqa_train_item()]
    test = [{'qid': 'test-0', 'question': 'public test question?'}]
    for name, payload in [('train.json', train), ('dev.json', dev),
                          ('test.json', test)]:
        (tmp_path / name).write_text(json.dumps(payload), encoding='utf-8')

    import federatedscope.llm.dataloader.dataloader as dataloader
    from federatedscope.llm.dataloader.dataloader import load_llm_dataset
    monkeypatch.setattr(dataloader, 'get_tokenizer',
                        lambda *args, **kwargs: (DummyTokenizer(), 0))

    train_ds, val_ds, test_ds = load_llm_dataset(_cfg(tmp_path))[0]

    assert len(train_ds) == 1
    assert len(val_ds) == 1
    assert len(test_ds) == 1
    assert 'sg' in train_ds[0]
