import argparse
import csv
import json
import os
import re
import string

import torch
import transformers
from tqdm import tqdm

from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.cmd_args import parse_client_cfg
from federatedscope.core.configs.config import global_cfg
from federatedscope.llm.kg_adapter.data_utils import build_kg_batch
from federatedscope.llm.misc.fschat import FSChatBot, get_tokenizer
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.llm.offsite_tuning.utils import wrap_offsite_tuning_for_eval
from federatedscope.llm.dataloader.task_datasets import (
    _cfg_data_args,
    _first_existing,
    _format_cwq_item,
    _format_webqsp_item,
    _read_json,
)

transformers.logging.set_verbosity(40)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate WebQSP/CWQ generated answers with hit@1.')
    parser.add_argument('--cfg', required=True, help='Path to YAML config.')
    parser.add_argument('--dataset',
                        choices=['webqsp', 'cwq'],
                        default=None,
                        help='Override dataset type inferred from cfg.')
    parser.add_argument('--split',
                        choices=['test', 'val'],
                        default='test',
                        help='Evaluation split.')
    parser.add_argument('--ckpt',
                        default=None,
                        help='Checkpoint path. Defaults to cfg.federate.save_to '
                        'and FSChatBot prefix search.')
    parser.add_argument('--output',
                        default=None,
                        help='Prediction jsonl path.')
    parser.add_argument('--summary',
                        default=None,
                        help='Summary csv path.')
    parser.add_argument('--limit',
                        type=int,
                        default=-1,
                        help='Evaluate only first N records for debugging.')
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('opts',
                        nargs=argparse.REMAINDER,
                        help='Optional cfg overrides after --, e.g. -- device 0')
    return parser.parse_args()


def _load_cfg(args):
    cfg = global_cfg.clone()
    cfg.merge_from_file(args.cfg)
    cfg_opt, _ = parse_client_cfg(args.opts)
    cfg.merge_from_list(cfg_opt)
    if args.ckpt:
        cfg.defrost()
        cfg.federate.save_to = args.ckpt
    update_logger(cfg, clear_before_add=True)
    setup_seed(cfg.seed)
    return cfg


def _infer_dataset_name(cfg, override):
    if override:
        return override
    data_type = str(cfg.data.type).lower()
    if 'cwq' in data_type or 'complexwebquestions' in data_type:
        return 'cwq'
    return 'webqsp'


def _load_records(cfg, dataset_name, split, tokenizer):
    args = _cfg_data_args(cfg)
    root = cfg.data.root
    if dataset_name == 'webqsp':
        path = _first_existing(root, [
            args.get(f'{split}_file', None),
            args.get('test_file' if split == 'test' else 'val_file', None),
            'WebQSP/data/WebQSP.test.json' if split == 'test' else None,
            'WebQSP/data/WebQSP.dev.json' if split == 'val' else None,
            'webqsp/WebQSP.test.json' if split == 'test' else None,
            'webqsp/WebQSP.dev.json' if split == 'val' else None,
        ])
        if path is None:
            raise FileNotFoundError(f'Cannot find WebQSP {split} file.')
        raw = _read_json(path)
        questions = raw.get('Questions', raw if isinstance(raw, list) else [])
        records = [
            _format_webqsp_item(item, tokenizer=tokenizer, config=cfg)
            for item in questions
        ]
    else:
        path = _first_existing(root, [
            args.get(f'{split}_file', None),
            args.get('test_file' if split == 'test' else 'val_file', None),
            args.get('dev_file', None) if split == 'val' else None,
            'CWQ/ComplexWebQuestions_test.json' if split == 'test' else None,
            'CWQ/ComplexWebQuestions_dev.json' if split == 'val' else None,
            'cwq/ComplexWebQuestions_test.json' if split == 'test' else None,
            'cwq/ComplexWebQuestions_dev.json' if split == 'val' else None,
        ])
        if path is None:
            raise FileNotFoundError(f'Cannot find CWQ {split} file.')
        raw = _read_json(path)
        if isinstance(raw, dict):
            raw = raw.get('questions', raw.get('Questions', raw.get('data', [])))
        split_name = 'test' if split == 'test' else 'validation'
        records = [
            _format_cwq_item(item, tokenizer=tokenizer, config=cfg,
                             split_name=split_name)
            for item in raw
        ] if isinstance(raw, list) else []

    records = [record for record in records if record is not None]
    return path, records


def _strip_prediction(text):
    text = text.strip()
    text = re.split(r'\n|###|Question:|Q:', text, maxsplit=1)[0]
    text = re.sub(r'^(the answer is|answer is|answer:)\s*',
                  '',
                  text,
                  flags=re.IGNORECASE)
    text = text.split(';')[0]
    text = text.split(',')[0] if len(text.split(',')) <= 3 else text
    return text.strip()


def _normalize_answer(text):
    text = str(text).lower()
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    return text


def _hit_at_1(prediction, answers):
    pred = _normalize_answer(_strip_prediction(prediction))
    gold = [_normalize_answer(answer) for answer in answers]
    return int(pred != '' and pred in gold), pred, gold


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters()
                    if param.requires_grad)
    return total, trainable


class ExactCheckpointBot(FSChatBot):
    def __init__(self, config, ckpt_path):
        self.config = config
        self.device = f'cuda:{config.device}'
        self.add_special_tokens = True

        model_name, _ = config.model.type.split('@')
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                          config.llm.tok_len)
        self.model = get_llm(config)

        if config.llm.offsite_tuning.use:
            self.model = wrap_offsite_tuning_for_eval(self.model, config,
                                                      ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state_dict = ckpt['model'] if isinstance(ckpt, dict) and \
                'model' in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)

        print(f'Model loads from the checkpoint {ckpt_path}')
        self._prepare_inference_model()
        self.max_history_len = config.llm.chat.max_history_len
        self.max_len = config.llm.chat.max_len
        self.history = []
        self.curpfx = ''


def _build_generation_inputs(bot, cfg, record):
    input_text = record['context']
    encoded = bot.tokenizer(input_text,
                            return_tensors='pt',
                            add_special_tokens=True,
                            truncation=True,
                            max_length=bot.tokenizer.model_max_length)
    device = bot._get_model_input_device()
    encoded = {key: value.to(device) for key, value in encoded.items()}

    model_kwargs = {}
    if cfg.llm.kg_adapter.use:
        instance = {
            'input_ids': encoded['input_ids'][0],
            'labels': encoded['input_ids'][0],
        }
        if 'sg' in record:
            instance['sg'] = record['sg']
        kg_batch = build_kg_batch([instance],
                                  encoded['input_ids'].cpu(),
                                  pad_id=bot.tokenizer.pad_token_id,
                                  kg_cfg=cfg.llm.kg_adapter)
        if kg_batch is not None:
            model_kwargs['kg_inputs'] = _move_to_device(kg_batch, device)
            model_kwargs['sg'] = model_kwargs['kg_inputs']
    return encoded, model_kwargs


@torch.no_grad()
def _generate_one(bot, cfg, record, generation_kwargs):
    encoded, model_kwargs = _build_generation_inputs(bot, cfg, record)
    kwargs = bot._normalize_generate_kwargs(dict(generation_kwargs))
    output_ids = bot.model.generate(**encoded, **model_kwargs, **kwargs)
    new_tokens = output_ids[0, encoded['input_ids'].shape[1]:]
    return bot.tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    args = parse_args()
    cfg = _load_cfg(args)
    dataset_name = _infer_dataset_name(cfg, args.dataset)
    ckpt_path = cfg.federate.save_to
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {ckpt_path}. Pass --ckpt explicitly.')
    bot = ExactCheckpointBot(cfg, ckpt_path)
    split_path, records = _load_records(cfg, dataset_name, args.split,
                                        bot.tokenizer)
    if args.limit > 0:
        records = records[:args.limit]

    output_path = args.output or os.path.join(
        cfg.outdir, f'{dataset_name}_{args.split}_hit1_predictions.jsonl')
    summary_path = args.summary or os.path.join(
        cfg.outdir, f'{dataset_name}_{args.split}_hit1_summary.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    generation_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'num_beams': args.num_beams,
        'do_sample': args.temperature > 0,
        'temperature': args.temperature if args.temperature > 0 else None,
        'top_p': args.top_p,
        'pad_token_id': bot.tokenizer.pad_token_id,
        'eos_token_id': bot.tokenizer.eos_token_id,
    }
    generation_kwargs = {
        key: value
        for key, value in generation_kwargs.items()
        if value is not None
    }

    correct = 0
    with open(output_path, 'w', encoding='utf-8') as fout:
        for idx, record in enumerate(tqdm(records, desc='Evaluating hit@1')):
            prediction = _generate_one(bot, cfg, record, generation_kwargs)
            answers = [answer.strip() for answer in record['target'].split(';')]
            hit, normalized_pred, normalized_gold = _hit_at_1(
                prediction, answers)
            correct += hit
            fout.write(
                json.dumps(
                    {
                        'idx': idx,
                        'question': record['context'],
                        'prediction': prediction,
                        'prediction_clean': _strip_prediction(prediction),
                        'prediction_norm': normalized_pred,
                        'answers': answers,
                        'answers_norm': normalized_gold,
                        'hit': hit,
                    },
                    ensure_ascii=False) + '\n')

    total = len(records)
    hit1 = 0.0 if total == 0 else 100.0 * correct / total
    total_params, trainable_params = _count_parameters(bot.model)
    with open(summary_path, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.DictWriter(fout,
                                fieldnames=[
                                    'dataset', 'split', 'checkpoint',
                                    'data_file', 'base_model', 'total_params',
                                    'trainable_params', 'total', 'correct',
                                    'hit@1'
                                ])
        writer.writeheader()
        writer.writerow({
            'dataset': dataset_name,
            'split': args.split,
            'checkpoint': cfg.federate.save_to,
            'data_file': split_path,
            'base_model': cfg.model.type,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total': total,
            'correct': correct,
            'hit@1': f'{hit1:.2f}',
        })

    print(f'{dataset_name.upper()} {args.split} hit@1: {hit1:.2f} '
          f'({correct}/{total})')
    print(f'Predictions: {output_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
