import argparse
import csv
import json
import os
import re
import statistics
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
    _format_grailqa_item,
    _format_graphquestions_item,
    _format_kqapro_item,
    _format_webqsp_item,
    _read_json,
    _train_val_split,
)

transformers.logging.set_verbosity(40)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate KGQA generated answers with hit@1.')
    parser.add_argument('--cfg', required=True, help='Path to YAML config.')
    parser.add_argument('--dataset',
                        choices=[
                            'webqsp', 'cwq', 'grailqa', 'kqa_pro', 'kqapro',
                            'graphquestions'
                        ],
                        default=None,
                        help='Override dataset type inferred from cfg.')
    parser.add_argument('--split',
                        choices=['test', 'val'],
                        default=None,
                        help='Evaluation split. Defaults to '
                        'cfg.eval.kgqa.split when set, otherwise test.')
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
    parser.add_argument('--partition-manifest',
                        default=None,
                        help='Optional IID/non-IID partition manifest. When '
                        'set, per-client and aggregate metrics are emitted.')
    parser.add_argument('--partition-summary-json',
                        default=None,
                        help='Per-client JSON summary path. Defaults beside '
                        '--summary.')
    parser.add_argument('--checkpoint-distribution',
                        choices=['iid', 'noniid', 'unknown'],
                        default=None,
                        help='Distribution used to train the checkpoint. By '
                        'default this is inferred from the config.')
    parser.add_argument('--allow-diagnostic-mismatch',
                        action='store_true',
                        help='Allow an IID checkpoint/non-IID manifest (or '
                        'reverse) diagnostic. Such output is marked '
                        'diagnostic_only and must not enter the paper table.')
    parser.add_argument('--limit',
                        type=int,
                        default=-1,
                        help='Evaluate only first N records for debugging.')
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--instruction',
                        default='',
                        help='Optional instruction prepended to each question '
                        'during evaluation.')
    parser.add_argument('--candidate-file',
                        default=None,
                        help='Optional json/jsonl file with candidate answers. '
                        'Each item should contain `idx`, `ID`, or `id`, and a '
                        '`candidates`/`candidate_answers` list.')
    parser.add_argument('--candidate-topk',
                        type=int,
                        default=32,
                        help='Maximum number of candidates to rerank per '
                        'question when --candidate-file or record candidates '
                        'are available.')
    parser.add_argument('--rerank-candidates',
                        action='store_true',
                        help='Use model log-likelihood to choose from KG '
                        'candidate answers instead of free-form generation.')
    parser.add_argument('--match-mode',
                        choices=['contains', 'exact'],
                        default=None,
                        help='Hit@1 matching mode. `contains` matches the '
                        'KGQA convention used by KG-Adapter; `exact` keeps '
                        'the previous strict normalized exact match. Defaults '
                        'to cfg.eval.kgqa.match_mode when set, otherwise '
                        'contains.')
    parser.add_argument('--allow-unlabeled',
                        action='store_true',
                        help='Allow evaluating/generating on splits without '
                        'gold answers. hit@1 will be invalid for them.')
    parser.add_argument('opts',
                        nargs=argparse.REMAINDER,
                        help='Optional cfg overrides after --, e.g. -- device 0')
    args, unknown_args = parser.parse_known_args()
    args.opts.extend(unknown_args)
    return args


def _normalize_cfg_opts(opts):
    if opts and opts[0] == '--':
        opts = opts[1:]
    normalized = []
    for item in opts:
        if item.startswith('--') and len(item) > 2:
            item = item[2:]
        normalized.append(item)
    return normalized


def _load_cfg(args):
    cfg = global_cfg.clone()
    cfg.merge_from_file(args.cfg)
    args.opts = _normalize_cfg_opts(args.opts)
    cfg_opt, _ = parse_client_cfg(args.opts)
    cfg.merge_from_list(cfg_opt)
    if args.ckpt:
        cfg.defrost()
        cfg.federate.save_to = args.ckpt
    update_logger(cfg, clear_before_add=True)
    setup_seed(cfg.seed)
    return cfg


def _cfg_get(cfg_node, key, default=None):
    if cfg_node is None:
        return default
    if isinstance(cfg_node, dict):
        return cfg_node.get(key, default)
    return getattr(cfg_node, key, default)


def _apply_eval_defaults(args, cfg):
    kgqa_eval = _cfg_get(_cfg_get(cfg, 'eval', None), 'kgqa', None)
    if args.split is None:
        args.split = str(_cfg_get(kgqa_eval, 'split', 'test')).lower()
    if args.match_mode is None:
        args.match_mode = str(
            _cfg_get(kgqa_eval, 'match_mode', 'contains')).lower()
    if args.limit < 0:
        args.limit = int(_cfg_get(kgqa_eval, 'limit', -1))
    return args


def _infer_dataset_name(cfg, override):
    if override:
        return 'kqa_pro' if override == 'kqapro' else override
    data_type = str(cfg.data.type).lower()
    if 'cwq' in data_type or 'complexwebquestions' in data_type:
        return 'cwq'
    if 'grailqa' in data_type:
        return 'grailqa'
    if 'kqa_pro' in data_type or 'kqapro' in data_type:
        return 'kqa_pro'
    if 'graphquestions' in data_type:
        return 'graphquestions'
    return 'webqsp'


def _canonical_dataset_name(name):
    return 'kqa_pro' if name in ['kqapro', 'kqa_pro'] else name


def _load_manifest(path):
    with open(path, encoding='utf-8') as fin:
        manifest = json.load(fin)
    if manifest.get('schema_version') != 1:
        raise ValueError('Only partition manifest schema_version=1 is '
                         'supported.')
    return manifest


def _configured_manifest_path(cfg):
    if str(cfg.data.splitter) != 'partition_manifest':
        return None
    args = cfg.data.splitter_args[0] if cfg.data.splitter_args else {}
    if isinstance(args, dict):
        return args.get('manifest')
    return getattr(args, 'manifest', None)


def _infer_checkpoint_distribution(args, cfg):
    if args.checkpoint_distribution is not None:
        return args.checkpoint_distribution, 'command_line'
    configured_manifest = _configured_manifest_path(cfg)
    if configured_manifest and os.path.isfile(configured_manifest):
        return _load_manifest(configured_manifest).get(
            'partition_type', 'unknown'), 'config_manifest'
    if str(cfg.data.splitter) == 'iid':
        return 'iid', 'config_splitter'
    return 'unknown', 'unresolved'


def _partition_context(args, cfg, dataset_name, split_name, record_count):
    if not args.partition_manifest:
        return None
    manifest = _load_manifest(args.partition_manifest)
    manifest_dataset = _canonical_dataset_name(manifest.get('dataset', ''))
    if manifest_dataset != _canonical_dataset_name(dataset_name):
        raise ValueError(
            f'Manifest dataset {manifest_dataset} does not match '
            f'{dataset_name}.')
    split = manifest.get('splits', {}).get(split_name)
    if split is None:
        raise ValueError(f'Manifest has no {split_name} split.')
    expected = int(split.get('num_samples', -1))
    if expected != record_count:
        raise ValueError(
            f'Manifest expects {expected} {split_name} records, loaded '
            f'{record_count}.')

    client_by_index = {}
    client_num = int(manifest.get('client_num', -1))
    for client_id in range(1, client_num + 1):
        for index in split.get('clients', {}).get(str(client_id), []):
            index = int(index)
            if index in client_by_index:
                raise ValueError(f'Manifest repeats sample index {index}.')
            client_by_index[index] = client_id
    if sorted(client_by_index) != list(range(record_count)):
        raise ValueError('Manifest must assign every evaluation record '
                         'exactly once.')

    checkpoint_distribution, distribution_source = \
        _infer_checkpoint_distribution(args, cfg)
    partition_type = manifest.get('partition_type', 'unknown')
    reasons = []
    if checkpoint_distribution not in ['unknown', partition_type]:
        reasons.append('checkpoint_partition_mismatch')
        if not args.allow_diagnostic_mismatch:
            raise ValueError(
                f'{checkpoint_distribution} checkpoint with {partition_type} '
                'evaluation manifest is diagnostic only. Pass '
                '--allow-diagnostic-mismatch to run it explicitly.')
    if args.limit > 0:
        reasons.append('limited_debug_evaluation')
    return {
        'manifest': manifest,
        'path': os.path.abspath(args.partition_manifest),
        'partition_type': partition_type,
        'checkpoint_distribution': checkpoint_distribution,
        'checkpoint_distribution_source': distribution_source,
        'diagnostic_only': bool(reasons),
        'diagnostic_reasons': reasons,
        'client_by_index': client_by_index,
        'client_num': client_num,
    }


def _json_records(path):
    raw = _read_json(path)
    if isinstance(raw, dict):
        raw = raw.get('questions', raw.get('Questions', raw.get('data', [])))
    return raw if isinstance(raw, list) else []


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
    elif dataset_name == 'cwq':
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
    elif dataset_name == 'grailqa':
        path = _first_existing(root, [
            args.get(f'{split}_file', None),
            args.get('test_file' if split == 'test' else 'val_file', None),
            args.get('dev_file', None) if split == 'val' else None,
            'GrailQA_v1.0/grailqa_v1.0_test_public.json'
            if split == 'test' else None,
            'GrailQA_v1.0/grailqa_v1.0_dev.json'
            if split == 'val' else None,
            'grailqa/grailqa_v1.0_test_public.json'
            if split == 'test' else None,
            'grailqa/grailqa_v1.0_dev.json' if split == 'val' else None,
        ])
        if path is None:
            raise FileNotFoundError(f'Cannot find GrailQA {split} file.')
        split_name = 'test' if split == 'test' else 'validation'
        records = [
            _format_grailqa_item(item, tokenizer=tokenizer, config=cfg,
                                 split_name=split_name)
            for item in _json_records(path)
        ]
    elif dataset_name == 'kqa_pro':
        path = _first_existing(root, [
            args.get(f'{split}_file', None),
            args.get('test_file' if split == 'test' else 'val_file', None),
            args.get('dev_file', None) if split == 'val' else None,
            'kqa_pro/test.json' if split == 'test' else None,
            'kqa_pro/val.json' if split == 'val' else None,
            'kqa_pro/dev.json' if split == 'val' else None,
            'KQA-Pro/test.json' if split == 'test' else None,
            'KQA-Pro/val.json' if split == 'val' else None,
            'KQA-Pro/dev.json' if split == 'val' else None,
        ])
        if path is None:
            raise FileNotFoundError(f'Cannot find KQA-Pro {split} file.')
        split_name = 'test' if split == 'test' else 'validation'
        records = [
            _format_kqapro_item(item, tokenizer=tokenizer, config=cfg,
                                split_name=split_name)
            for item in _json_records(path)
        ]
    elif dataset_name == 'graphquestions':
        path = _first_existing(root, [
            args.get(f'{split}_file', None),
            args.get('test_file' if split == 'test' else 'val_file', None),
            args.get('dev_file', None) if split == 'val' else None,
            'GraphQuestions/graphquestions.testing.json'
            if split == 'test' else None,
            'GraphQuestions/graphquestions.validation.json'
            if split == 'val' else None,
            'GraphQuestions/graphquestions.dev.json'
            if split == 'val' else None,
            'graphquestions/graphquestions.testing.json'
            if split == 'test' else None,
            'graphquestions/graphquestions.validation.json'
            if split == 'val' else None,
            'graphquestions/graphquestions.dev.json' if split == 'val' else None,
        ])
        if path is None and split == 'val':
            train_path = _first_existing(root, [
                args.get('train_file', None),
                'GraphQuestions/graphquestions.training.json',
                'graphquestions/graphquestions.training.json',
                'graphquestions.training.json',
            ])
            if train_path is None:
                raise FileNotFoundError(
                    'Cannot find GraphQuestions train file to derive val split.'
                )
            train_records = [
                _format_graphquestions_item(item,
                                            tokenizer=tokenizer,
                                            config=cfg,
                                            split_name='train')
                for item in _json_records(train_path)
            ]
            train_records = [
                record for record in train_records if record is not None
            ]
            total_ratio = float(cfg.data.splits[0] + cfg.data.splits[1])
            val_ratio = 0.0 if total_ratio <= 0 else \
                float(cfg.data.splits[1]) / total_ratio
            _, records = _train_val_split(train_records, val_ratio)
            path = f'{train_path}#derived-val'
        elif path is None:
            raise FileNotFoundError(
                f'Cannot find GraphQuestions {split} file.')
        else:
            split_name = 'test' if split == 'test' else 'validation'
            records = [
                _format_graphquestions_item(item,
                                            tokenizer=tokenizer,
                                            config=cfg,
                                            split_name=split_name)
                for item in _json_records(path)
            ]
    elif dataset_name == 'cwq':
        raise ValueError(f'Unsupported KGQA dataset {dataset_name}.')

    records = [record for record in records if record is not None]
    return path, records


def _strip_prediction(text):
    text = text.strip()
    text = re.split(r'\n|###|Question:|Q:', text, maxsplit=1)[0]
    text = re.sub(r'^(the answer is|answer is|answer:)\s*',
                  '',
                  text,
                  flags=re.IGNORECASE)
    text = re.split(r'\b(because|therefore|so the answer is)\b',
                    text,
                    maxsplit=1,
                    flags=re.IGNORECASE)[0]
    return text.strip()


def _normalize_answer(text):
    text = str(text).lower()
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    return text


def _hit_at_1(prediction, answers, match_mode='contains'):
    if len(answers) == 0:
        pred = _normalize_answer(_strip_prediction(prediction))
        return 0, pred, [], 0
    pred = _normalize_answer(_strip_prediction(prediction))
    gold = []
    for answer in answers:
        normalized = _normalize_answer(answer)
        if normalized and normalized not in gold:
            gold.append(normalized)

    exact_hit = int(pred != '' and pred in gold)
    if match_mode == 'exact':
        return exact_hit, pred, gold, exact_hit

    padded_pred = f' {pred} '
    contains_hit = int(pred != '' and any(
        f' {answer} ' in padded_pred for answer in gold))
    return contains_hit, pred, gold, exact_hit


def _answers_from_raw_record(record):
    answers = []
    raw_answers = record.get('answers', record.get('answer', []))
    if isinstance(raw_answers, list):
        for raw_answer in raw_answers:
            if isinstance(raw_answer, dict):
                for key in [
                        'answer', 'entity_name', 'answer_argument', 'name',
                        'label'
                ]:
                    value = raw_answer.get(key, None)
                    if value not in [None, '']:
                        answers.append(str(value).strip())
                aliases = raw_answer.get('aliases', [])
                if isinstance(aliases, list):
                    answers.extend(str(alias).strip() for alias in aliases)
                elif aliases not in [None, '']:
                    answers.append(str(aliases).strip())
            elif raw_answer not in [None, '']:
                answers.append(str(raw_answer).strip())
    elif raw_answers not in [None, '']:
        answers.append(str(raw_answers).strip())
    return answers


def _record_answers(record):
    answers = [
        answer.strip()
        for answer in record.get('target', '').split(';')
        if answer.strip()
    ]
    answers.extend(_as_candidate_list(record.get('answer_aliases', [])))
    answers.extend(_answers_from_raw_record(record))

    deduped = []
    for answer in answers:
        if answer and answer not in deduped:
            deduped.append(answer)
    return deduped


def _as_candidate_list(value):
    if value in [None, '']:
        return []
    if isinstance(value, str):
        pieces = re.split(r';|\n|\t', value)
    elif isinstance(value, list):
        pieces = value
    else:
        pieces = [value]

    candidates = []
    for item in pieces:
        if isinstance(item, dict):
            item = item.get('answer',
                            item.get('entity_name',
                                     item.get('label',
                                              item.get('name', ''))))
        item = str(item).strip()
        if item and item not in candidates:
            candidates.append(item)
    return candidates


def _load_candidate_map(path):
    if path in [None, '']:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f'Candidate file not found: {path}')

    if path.endswith('.jsonl'):
        rows = [
            json.loads(line)
            for line in open(path, encoding='utf-8')
            if line.strip()
        ]
    else:
        with open(path, encoding='utf-8') as fin:
            rows = json.load(fin)
        if isinstance(rows, dict):
            rows = rows.get('data', rows.get('items', rows))
            if isinstance(rows, dict):
                return {
                    str(key): _as_candidate_list(value)
                    for key, value in rows.items()
                }

    candidate_map = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidates = _as_candidate_list(
            row.get('candidates', row.get('candidate_answers', [])))
        for key in ['idx', 'ID', 'id', 'question_id', 'qid']:
            if key in row:
                candidate_map[str(row[key])] = candidates
    return candidate_map


def _record_candidates(record, idx, candidate_map, topk):
    candidates = []
    for key in ['candidates', 'candidate_answers', 'candidate_entities']:
        candidates.extend(_as_candidate_list(record.get(key, [])))
    for key in ['ID', 'id', 'question_id', 'qid']:
        if key in record:
            candidates.extend(candidate_map.get(str(record[key]), []))
    candidates.extend(candidate_map.get(str(idx), []))

    deduped = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    if topk > 0:
        deduped = deduped[:topk]
    return deduped


def _apply_instruction(records, instruction):
    instruction = str(instruction).strip()
    if instruction == '':
        return records
    for record in records:
        context = record.get('context', '')
        if context.startswith(instruction):
            continue
        record['context'] = f'{instruction}\n{context}'
    return records


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


@torch.no_grad()
def _score_candidate(bot, cfg, record, candidate):
    prompt = record['context']
    full_text = prompt + ' ' + candidate
    encoded = bot.tokenizer(full_text,
                            return_tensors='pt',
                            add_special_tokens=True,
                            truncation=True,
                            max_length=bot.tokenizer.model_max_length)
    prompt_ids = bot.tokenizer(prompt,
                               return_tensors='pt',
                               add_special_tokens=True,
                               truncation=True,
                               max_length=bot.tokenizer.model_max_length)
    device = bot._get_model_input_device()
    encoded = {key: value.to(device) for key, value in encoded.items()}

    labels = encoded['input_ids'].clone()
    prompt_len = min(prompt_ids['input_ids'].shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100

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

    output = bot.model(**encoded, **model_kwargs)
    logits = output.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction='sum')
    token_count = shift_labels.ne(-100).sum().clamp_min(1)
    return -float(loss / token_count)


def _rerank_candidates(bot, cfg, record, candidates):
    scored = [
        (_score_candidate(bot, cfg, record, candidate), candidate)
        for candidate in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1], scored


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _partition_metrics(partition, client_totals, client_correct,
                       client_exact):
    clients = []
    for client_id in range(1, partition['client_num'] + 1):
        total = client_totals[client_id]
        correct = client_correct[client_id]
        exact_correct = client_exact[client_id]
        clients.append({
            'client_id': client_id,
            'total': total,
            'correct': correct,
            'hit@1': 0.0 if total == 0 else 100.0 * correct / total,
            'exact_correct': exact_correct,
            'exact@1': 0.0 if total == 0 else
            100.0 * exact_correct / total,
        })
    scores = [client['hit@1'] for client in clients]
    total = sum(client_totals.values())
    correct = sum(client_correct.values())
    exact_correct = sum(client_exact.values())
    return {
        'clients': clients,
        'macro_average': statistics.fmean(scores),
        'weighted_global': 0.0 if total == 0 else 100.0 * correct / total,
        'worst_client': min(scores),
        'client_std': statistics.pstdev(scores),
        'total': total,
        'correct': correct,
        'exact_correct': exact_correct,
        'exact_global': 0.0 if total == 0 else
        100.0 * exact_correct / total,
    }


def _write_partition_summaries(csv_path, json_path, metadata, metrics):
    common = {
        key: metadata[key]
        for key in [
            'dataset', 'split', 'checkpoint', 'data_file', 'base_model',
            'total_params', 'trainable_params',
            'partition_manifest', 'partition_type',
            'checkpoint_distribution', 'checkpoint_distribution_source',
            'diagnostic_only', 'diagnostic_reasons', 'match_mode'
        ]
    }
    fieldnames = list(common.keys()) + [
        'row_type', 'client_id', 'total', 'correct', 'hit@1',
        'exact_correct', 'exact@1', 'macro_average', 'weighted_global',
        'worst_client', 'client_std'
    ]
    with open(csv_path, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for client in metrics['clients']:
            row = dict(common)
            row.update({
                'row_type': 'client',
                'client_id': client['client_id'],
                'total': client['total'],
                'correct': client['correct'],
                'hit@1': f'{client["hit@1"]:.6f}',
                'exact_correct': client['exact_correct'],
                'exact@1': f'{client["exact@1"]:.6f}',
            })
            writer.writerow(row)
        row = dict(common)
        row.update({
            'row_type': 'aggregate',
            'client_id': '',
            'total': metrics['total'],
            'correct': metrics['correct'],
            'hit@1': f'{metrics["weighted_global"]:.6f}',
            'exact_correct': metrics['exact_correct'],
            'exact@1': f'{metrics["exact_global"]:.6f}',
            'macro_average': f'{metrics["macro_average"]:.6f}',
            'weighted_global': f'{metrics["weighted_global"]:.6f}',
            'worst_client': f'{metrics["worst_client"]:.6f}',
            'client_std': f'{metrics["client_std"]:.6f}',
        })
        writer.writerow(row)

    payload = {
        'schema_version': 1,
        **metadata,
        **metrics,
    }
    with open(json_path, 'w', encoding='utf-8', newline='\n') as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2, sort_keys=True)
        fout.write('\n')


def main():
    args = parse_args()
    cfg = _load_cfg(args)
    args = _apply_eval_defaults(args, cfg)
    dataset_name = _infer_dataset_name(cfg, args.dataset)
    ckpt_path = cfg.federate.save_to
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {ckpt_path}. Pass --ckpt explicitly.')
    bot = ExactCheckpointBot(cfg, ckpt_path)
    split_path, records = _load_records(cfg, dataset_name, args.split,
                                        bot.tokenizer)
    records = _apply_instruction(records, args.instruction)
    partition = _partition_context(args, cfg, dataset_name, args.split,
                                   len(records))
    if args.limit > 0:
        records = records[:args.limit]
    candidate_map = _load_candidate_map(args.candidate_file)
    unlabeled_num = sum(1 for record in records
                        if not str(record.get('target', '')).strip())
    if unlabeled_num > 0 and not args.allow_unlabeled:
        raise ValueError(
            f'{dataset_name.upper()} {args.split} contains {unlabeled_num} '
            'records without gold answers, so hit@1 cannot be computed. '
            'For CWQ, the official test file in this repo has no `answers` '
            'field. Use `--split val` to evaluate on dev, provide a labeled '
            'test file in cfg.data.args, or add `--allow-unlabeled` only to '
            'generate predictions without valid hit@1.')

    output_path = args.output or os.path.join(
        cfg.outdir, f'{dataset_name}_{args.split}_hit1_predictions.jsonl')
    summary_path = args.summary or os.path.join(
        cfg.outdir, f'{dataset_name}_{args.split}_hit1_summary.csv')
    partition_json_path = args.partition_summary_json or \
        os.path.splitext(summary_path)[0] + '.json'
    _ensure_parent(output_path)
    _ensure_parent(summary_path)
    if partition is not None:
        _ensure_parent(partition_json_path)

    generation_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'num_beams': args.num_beams,
        'use_cache': False,
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
    exact_correct = 0
    client_totals = {
        client_id: 0 for client_id in range(1, partition['client_num'] + 1)
    } if partition is not None else None
    client_correct = dict(client_totals) if partition is not None else None
    client_exact = dict(client_totals) if partition is not None else None
    with open(output_path, 'w', encoding='utf-8') as fout:
        for idx, record in enumerate(tqdm(records, desc='Evaluating hit@1')):
            candidates = _record_candidates(record, idx, candidate_map,
                                            args.candidate_topk)
            candidate_scores = []
            if args.rerank_candidates and candidates:
                prediction, candidate_scores = _rerank_candidates(
                    bot, cfg, record, candidates)
            else:
                prediction = _generate_one(bot, cfg, record,
                                           generation_kwargs)
            answers = _record_answers(record)
            hit, normalized_pred, normalized_gold, exact_hit = _hit_at_1(
                prediction, answers, args.match_mode)
            correct += hit
            exact_correct += exact_hit
            client_id = partition['client_by_index'][idx] \
                if partition is not None else None
            if partition is not None:
                client_totals[client_id] += 1
                client_correct[client_id] += hit
                client_exact[client_id] += exact_hit
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
                        'exact_hit': exact_hit,
                        'client_id': client_id,
                        'match_mode': args.match_mode,
                        'candidates': candidates,
                        'candidate_scores': candidate_scores[:10],
                    },
                    ensure_ascii=False) + '\n')

    total = len(records)
    hit1 = 0.0 if total == 0 else 100.0 * correct / total
    exact_hit1 = 0.0 if total == 0 else 100.0 * exact_correct / total
    total_params, trainable_params = _count_parameters(bot.model)
    if partition is None:
        with open(summary_path, 'w', encoding='utf-8', newline='') as fout:
            writer = csv.DictWriter(fout,
                                    fieldnames=[
                                        'dataset', 'split', 'checkpoint',
                                        'data_file', 'base_model',
                                        'total_params', 'trainable_params',
                                        'total', 'correct', 'hit@1',
                                        'exact_correct', 'exact@1',
                                        'match_mode'
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
                'exact_correct': exact_correct,
                'exact@1': f'{exact_hit1:.2f}',
                'match_mode': args.match_mode,
            })
    else:
        metrics = _partition_metrics(partition, client_totals,
                                     client_correct, client_exact)
        metadata = {
            'dataset': dataset_name,
            'split': args.split,
            'checkpoint': cfg.federate.save_to,
            'data_file': split_path,
            'base_model': cfg.model.type,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'partition_manifest': partition['path'],
            'partition_type': partition['partition_type'],
            'checkpoint_distribution': partition[
                'checkpoint_distribution'],
            'checkpoint_distribution_source': partition[
                'checkpoint_distribution_source'],
            'diagnostic_only': partition['diagnostic_only'],
            'diagnostic_reasons': partition['diagnostic_reasons'],
            'match_mode': args.match_mode,
        }
        _write_partition_summaries(summary_path, partition_json_path,
                                   metadata, metrics)

    print(f'{dataset_name.upper()} {args.split} hit@1: {hit1:.2f} '
          f'({correct}/{total})')
    print(f'Exact hit@1: {exact_hit1:.2f} ({exact_correct}/{total})')
    print(f'Predictions: {output_path}')
    print(f'Summary: {summary_path}')
    if partition is not None:
        print(f'Partition summary: {partition_json_path}')
        print(f'Macro average: {metrics["macro_average"]:.2f}')
        print(f'Worst client: {metrics["worst_client"]:.2f}')
        print(f'Client std: {metrics["client_std"]:.2f}')
        if partition['diagnostic_only']:
            print('DIAGNOSTIC ONLY: ' +
                  ', '.join(partition['diagnostic_reasons']))


if __name__ == '__main__':
    main()
