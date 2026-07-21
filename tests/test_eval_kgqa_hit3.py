from types import SimpleNamespace
from unittest.mock import patch

import torch

from fedbiot_script.eval_kgqa_hit3 import _evaluate_predictions
from fedbiot_script.eval_kgqa_hit3 import _generate_top3
from fedbiot_script import eval_kgqa_hit1 as hit1_eval


def test_hit3_accepts_a_correct_second_ranked_prediction():
    result = _evaluate_predictions(
        ['wrong answer', 'Paris', 'another answer'], ['Paris'], 'exact')
    assert result['hit@1'] == 0
    assert result['hit@3'] == 1
    assert result['exact@1'] == 0
    assert result['exact@3'] == 1


def test_hit3_rejects_a_correct_fourth_ranked_prediction():
    result = _evaluate_predictions(
        ['one', 'two', 'three', 'Paris'], ['Paris'], 'exact')
    assert result['hit@1'] == 0
    assert result['hit@3'] == 0


def test_hit3_supports_multiple_gold_aliases():
    result = _evaluate_predictions(
        ['wrong', 'City of Paris'], ['Paris', 'City of Paris'], 'exact')
    assert result['hit@3'] == 1


def test_contains_is_applied_to_each_candidate_independently():
    result = _evaluate_predictions(
        ['unknown', 'The answer is Paris because it is in France.'],
        ['Paris'], 'contains')
    assert result['hit@1'] == 0
    assert result['hit@3'] == 1


def test_beam_generation_expands_kg_side_channel_to_three_samples():
    class FakeTokenizer:
        pad_token_id = 0

        @staticmethod
        def decode(tokens, skip_special_tokens=True):
            del skip_special_tokens
            return str(int(tokens[-1]))

    class FakeModel:
        @staticmethod
        def generate(**kwargs):
            assert kwargs['kg_inputs']['node_ids'].shape[0] == 3
            return torch.tensor([[1, 2, 10], [1, 2, 20], [1, 2, 30]])

    class FakeBot:
        tokenizer = FakeTokenizer()
        model = FakeModel()

        @staticmethod
        def _normalize_generate_kwargs(kwargs):
            return kwargs

        @staticmethod
        def _get_model_input_device():
            return torch.device('cpu')

    cfg = SimpleNamespace(
        llm=SimpleNamespace(kg_adapter=SimpleNamespace(use=True)))
    encoded = {'input_ids': torch.tensor([[1, 2]])}
    model_kwargs = {'kg_inputs': {'node_ids': torch.tensor([[1]])}}

    def fake_build_batch(instances, input_ids, pad_id, kg_cfg):
        del pad_id, kg_cfg
        assert len(instances) == 3
        assert input_ids.shape == (3, 2)
        return {'node_ids': torch.arange(3).view(3, 1)}

    with patch.object(hit1_eval,
                      '_build_generation_inputs',
                      return_value=(encoded, model_kwargs)), \
            patch.object(hit1_eval,
                         'build_kg_batch',
                         side_effect=fake_build_batch):
        predictions = _generate_top3(
            FakeBot(), cfg, {'sg': {}}, {'num_beams': 3})

    assert predictions == ['10', '20', '30']
