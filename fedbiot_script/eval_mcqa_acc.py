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

transformers.logging.set_verbosity(40)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate OpenBookQA/MCQA generated answers with accuracy.')
    parser.add_argument('--cfg', required=True, help='Path to YAML config.')
    parser.add_argument('--split',
                        choices=['test', 'val', 'train'],
                        default='test')
    parser.add_argument('--ckpt',
                        default=None,
                        help='Checkpoint path. Defaults to cfg.federate.save_to.')
    parser.add_argument('--output', default=None)
    parser.add_argument('--summary', default=None)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--eval-mode',
                        choices=['generate', 'choice_score'],
                        default='generate',
                        help='generate: free-form generation then map to a '
                        'choice; choice_score: rank choices by conditional '
                        'log-likelihood.')
    parser.add_argument('--score-target',
                        choices=['choice_text', 'label'],
                        default='choice_text',
                        help='Candidate text used by choice_score. '
                        'choice_text matches the current preprocessed '
                        'OpenBookQA training target.')
    parser.add_argument('--length-norm',
                        choices=['mean', 'sum'],
                        default='mean',
                        help='Normalize candidate log-likelihood by length.')
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


def _cfg_data_args(cfg):
    if hasattr(cfg.data, 'args') and len(cfg.data.args) > 0:
        return cfg.data.args[0]
    return {}


def _join_path(root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _split_file(cfg, split):
    args = _cfg_data_args(cfg)
    version = args.get('version', 'obqa_conceptnet_3hop')
    key = {
        'train': 'train_file',
        'val': 'val_file',
        'test': 'test_file',
    }[split]
    default_name = {
        'train': f'train_{version}.pt',
        'val': f'dev_{version}.pt',
        'test': f'test_{version}.pt',
    }[split]
    return _join_path(cfg.data.root, args.get(key, default_name))


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if hasattr(value, 'to') and callable(value.to):
        try:
            return value.to(device)
        except TypeError:
            return value
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


def _normalize_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())


def _strip_prediction(text):
    text = str(text).strip()
    text = re.split(r'\n|###|Question:|Q:', text, maxsplit=1)[0]
    text = re.sub(r'^(the answer is|answer is|answer:)\s*',
                  '',
                  text,
                  flags=re.IGNORECASE)
    return text.strip()


def _choice_from_prediction(prediction, choices):
    clean = _strip_prediction(prediction)
    upper = clean.upper()
    for label in ['A', 'B', 'C', 'D', 'E']:
        patterns = [
            rf'^\(?{label}\)?(?:\.|:|\s|$)',
            rf'\bOPTION\s+{label}\b',
            rf'\bCHOICE\s+{label}\b',
        ]
        if any(re.search(pattern, upper) for pattern in patterns):
            return label, clean

    norm_pred = _normalize_text(clean)
    for idx, choice in enumerate(choices):
        label = chr(ord('A') + idx)
        norm_choice = _normalize_text(choice)
        if norm_choice and (norm_choice == norm_pred or norm_choice in norm_pred):
            return label, clean
    return '', clean


def _prompt_from_record(record, tokenizer):
    if 'input_ids_no_response' in record:
        ids = record['input_ids_no_response']
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return tokenizer.decode(ids, skip_special_tokens=True)
    return record.get('context', record.get('question', ''))


def _build_generation_inputs(bot, cfg, record):
    prompt = _prompt_from_record(record, bot.tokenizer)
    encoded = bot.tokenizer(prompt,
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
    return prompt, encoded, model_kwargs


def _build_kg_kwargs(bot, cfg, record, input_ids, device):
    model_kwargs = {}
    if cfg.llm.kg_adapter.use:
        instance = {
            'input_ids': input_ids[0].detach().cpu(),
            'labels': input_ids[0].detach().cpu(),
        }
        if 'sg' in record:
            instance['sg'] = record['sg']
        kg_batch = build_kg_batch([instance],
                                  input_ids.detach().cpu(),
                                  pad_id=bot.tokenizer.pad_token_id,
                                  kg_cfg=cfg.llm.kg_adapter)
        if kg_batch is not None:
            model_kwargs['kg_inputs'] = _move_to_device(kg_batch, device)
            model_kwargs['sg'] = model_kwargs['kg_inputs']
    return model_kwargs


@torch.no_grad()
def _generate_one(bot, cfg, record, generation_kwargs):
    _, encoded, model_kwargs = _build_generation_inputs(bot, cfg, record)
    kwargs = bot._normalize_generate_kwargs(dict(generation_kwargs))
    output_ids = bot.model.generate(**encoded, **model_kwargs, **kwargs)
    new_tokens = output_ids[0, encoded['input_ids'].shape[1]:]
    return bot.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _candidate_text(label, choice, score_target):
    if score_target == 'label':
        return label
    return str(choice).strip()


@torch.no_grad()
def _score_choice(bot, cfg, record, prompt_ids, candidate_ids, length_norm):
    device = bot._get_model_input_device()
    full_ids = torch.cat([prompt_ids, candidate_ids], dim=1).to(device)
    attention_mask = torch.ones_like(full_ids, device=device)
    model_kwargs = _build_kg_kwargs(bot, cfg, record, full_ids, device)
    outputs = bot.model(input_ids=full_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        **model_kwargs)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    start = prompt_ids.size(1)
    token_scores = []
    log_probs = torch.log_softmax(logits[0, start - 1:-1, :].float(), dim=-1)
    target_ids = full_ids[0, start:]
    for pos, token_id in enumerate(target_ids):
        token_scores.append(log_probs[pos, token_id].item())
    if not token_scores:
        return float('-inf')
    total = sum(token_scores)
    if length_norm == 'mean':
        return total / len(token_scores)
    return total


def _choice_by_score(bot, cfg, record, score_target, length_norm):
    prompt = _prompt_from_record(record, bot.tokenizer)
    device = bot._get_model_input_device()
    prompt_encoded = bot.tokenizer(prompt,
                                   return_tensors='pt',
                                   add_special_tokens=True,
                                   truncation=True,
                                   max_length=bot.tokenizer.model_max_length)
    prompt_ids = prompt_encoded['input_ids'].to(device)

    scores = {}
    choices = record.get('choices', [])
    for idx, choice in enumerate(choices):
        label = chr(ord('A') + idx)
        candidate = _candidate_text(label, choice, score_target)
        candidate_ids = bot.tokenizer(' ' + candidate,
                                      return_tensors='pt',
                                      add_special_tokens=False).input_ids
        candidate_ids = candidate_ids.to(device)
        scores[label] = _score_choice(bot, cfg, record, prompt_ids,
                                      candidate_ids, length_norm)
    if not scores:
        return '', '', {}
    pred_label = max(scores, key=scores.get)
    prediction_clean = choices[ord(pred_label) - ord('A')] \
        if score_target == 'choice_text' else pred_label
    return pred_label, prediction_clean, scores


def _gold_label(record):
    answer_key = str(record.get('answerKey', '')).strip().upper()
    if answer_key:
        return answer_key
    choices = record.get('choices', [])
    answer = _normalize_text(record.get('answer', ''))
    for idx, choice in enumerate(choices):
        if _normalize_text(choice) == answer:
            return chr(ord('A') + idx)
    return ''


def main():
    args = parse_args()
    cfg = _load_cfg(args)
    ckpt_path = cfg.federate.save_to
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {ckpt_path}. Pass --ckpt explicitly.')

    split_path = _split_file(cfg, args.split)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f'Split file not found: {split_path}')
    records = torch.load(split_path, map_location='cpu')
    if args.limit > 0:
        records = records[:args.limit]

    bot = ExactCheckpointBot(cfg, ckpt_path)
    output_path = args.output or os.path.join(
        cfg.outdir, f'openbookqa_{args.split}_mcqa_predictions.jsonl')
    summary_path = args.summary or os.path.join(
        cfg.outdir, f'openbookqa_{args.split}_mcqa_summary.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

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
    with open(output_path, 'w', encoding='utf-8') as fout:
        for idx, record in enumerate(tqdm(records, desc='Evaluating MCQA')):
            choice_scores = {}
            if args.eval_mode == 'choice_score':
                pred_label, prediction_clean, choice_scores = \
                    _choice_by_score(bot, cfg, record, args.score_target,
                                     args.length_norm)
                prediction = prediction_clean
            else:
                prediction = _generate_one(bot, cfg, record,
                                           generation_kwargs)
                pred_label, prediction_clean = _choice_from_prediction(
                    prediction, record.get('choices', []))
            gold = _gold_label(record)
            hit = int(pred_label != '' and pred_label == gold)
            correct += hit
            fout.write(
                json.dumps(
                    {
                        'idx': idx,
                        'id': record.get('id', ''),
                        'question': record.get('question', ''),
                        'choices': record.get('choices', []),
                        'prediction': prediction,
                        'prediction_clean': prediction_clean,
                        'pred_label': pred_label,
                        'choice_scores': choice_scores,
                        'gold': gold,
                        'answer': record.get('answer', ''),
                        'hit': hit,
                    },
                    ensure_ascii=False) + '\n')

    total = len(records)
    acc = 0.0 if total == 0 else 100.0 * correct / total
    total_params, trainable_params = _count_parameters(bot.model)
    with open(summary_path, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.DictWriter(fout,
                                fieldnames=[
                                    'dataset', 'split', 'checkpoint',
                                    'data_file', 'eval_mode',
                                    'score_target', 'length_norm',
                                    'base_model', 'total_params',
                                    'trainable_params', 'total',
                                    'correct', 'accuracy'
                                ])
        writer.writeheader()
        writer.writerow({
            'dataset': 'openbookqa',
            'split': args.split,
            'checkpoint': ckpt_path,
            'data_file': split_path,
            'eval_mode': args.eval_mode,
            'score_target': args.score_target,
            'length_norm': args.length_norm,
            'base_model': cfg.model.type,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total': total,
            'correct': correct,
            'accuracy': f'{acc:.2f}',
        })

    print(f'OpenBookQA {args.split} accuracy: {acc:.2f} '
          f'({correct}/{total})')
    print(f'Predictions: {output_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
