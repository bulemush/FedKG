"""Evaluate IID KGQA checkpoints with ranked Hit@1 and Hit@3.

Free-form evaluation uses the three highest-ranked beam-search sequences.
Candidate-based evaluation scores every supplied candidate by normalized model
log-likelihood and evaluates the first three candidates after sorting.
"""

import argparse
import csv
import json
import os

import torch
from tqdm import tqdm

from fedbiot_script import eval_kgqa_hit1 as hit1_eval


TOP_K = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate IID KGQA generated answers with Hit@1/Hit@3.')
    parser.add_argument('--cfg', required=True, help='Path to YAML config.')
    parser.add_argument(
        '--dataset',
        choices=[
            'webqsp', 'cwq', 'grailqa', 'kqa_pro', 'kqapro',
            'graphquestions'
        ],
        default=None,
        help='Override dataset type inferred from cfg.')
    parser.add_argument(
        '--split',
        choices=['test', 'val'],
        default=None,
        help='Evaluation split. Defaults to cfg.eval.kgqa.split.')
    parser.add_argument(
        '--ckpt',
        default=None,
        help='Checkpoint path. Defaults to cfg.federate.save_to.')
    parser.add_argument('--output', default=None, help='Prediction JSONL path.')
    parser.add_argument('--summary', default=None, help='Summary CSV path.')
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--max-new-tokens', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--num-beams', type=int, default=None)
    parser.add_argument(
        '--instruction',
        default='',
        help='Optional instruction prepended to each question.')
    parser.add_argument(
        '--candidate-file',
        default=None,
        help='Optional JSON/JSONL candidate-answer file.')
    parser.add_argument(
        '--candidate-topk',
        type=int,
        default=32,
        help='Maximum candidates scored per question.')
    parser.add_argument(
        '--rerank-candidates',
        action='store_true',
        help='Rank supplied candidates by model log-likelihood.')
    parser.add_argument(
        '--match-mode',
        choices=['contains', 'exact'],
        default=None,
        help='Answer matching mode.')
    parser.add_argument(
        '--allow-unlabeled',
        action='store_true',
        help='Generate on an unlabeled split; reported Hits are invalid.')
    parser.add_argument(
        'opts',
        nargs=argparse.REMAINDER,
        help='Optional config overrides after --.')
    args, unknown_args = parser.parse_known_args()
    args.opts.extend(unknown_args)
    return args


def _apply_defaults(args, cfg):
    args = hit1_eval._apply_eval_defaults(args, cfg)
    kgqa_eval = hit1_eval._cfg_get(
        hit1_eval._cfg_get(cfg, 'eval', None), 'kgqa', None)
    if args.max_new_tokens is None:
        args.max_new_tokens = int(
            hit1_eval._cfg_get(kgqa_eval, 'max_new_tokens', 32))
    if args.temperature is None:
        args.temperature = float(
            hit1_eval._cfg_get(kgqa_eval, 'temperature', 0.0))
    if args.top_p is None:
        args.top_p = float(hit1_eval._cfg_get(kgqa_eval, 'top_p', 1.0))
    if args.num_beams is None:
        args.num_beams = int(
            hit1_eval._cfg_get(kgqa_eval, 'num_beams', TOP_K))
    args.num_beams = max(TOP_K, args.num_beams)
    return args


def _evaluate_predictions(predictions, answers, match_mode='contains'):
    """Return rank-aware Hit@1/Hit@3 values for independent predictions."""
    predictions = list(predictions[:TOP_K])
    rank_hits = []
    rank_exact_hits = []
    normalized_predictions = []
    normalized_gold = []
    for prediction in predictions:
        hit, normalized, gold, exact_hit = hit1_eval._hit_at_1(
            prediction, answers, match_mode)
        rank_hits.append(hit)
        rank_exact_hits.append(exact_hit)
        normalized_predictions.append(normalized)
        normalized_gold = gold
    return {
        'hit@1': int(bool(rank_hits) and bool(rank_hits[0])),
        'hit@3': int(any(rank_hits)),
        'exact@1': int(bool(rank_exact_hits) and bool(rank_exact_hits[0])),
        'exact@3': int(any(rank_exact_hits)),
        'rank_hits': rank_hits,
        'rank_exact_hits': rank_exact_hits,
        'predictions_norm': normalized_predictions,
        'answers_norm': normalized_gold,
    }


@torch.no_grad()
def _generate_top3(bot, cfg, record, generation_kwargs):
    encoded, model_kwargs = hit1_eval._build_generation_inputs(
        bot, cfg, record)
    kwargs = bot._normalize_generate_kwargs(dict(generation_kwargs))
    if kwargs.get('num_beams', 1) < TOP_K:
        raise RuntimeError(
            'Hit@3 beam evaluation requires num_beams >= 3, but model '
            'parallel inference reduced num_beams to 1. Evaluate on one '
            'visible GPU with `CUDA_VISIBLE_DEVICES=<id>` and append '
            '`-- llm.model_parallel.use False`, or provide a candidate file '
            'and use --rerank-candidates.')
    # Hugging Face expands token inputs from batch 1 to batch=num_beams before
    # the first beam-search forward pass. KG inputs live in a side-channel and
    # are not expanded by Transformers, so explicitly rebuild one identical
    # graph sample per beam to keep token and graph batch dimensions aligned.
    if cfg.llm.kg_adapter.use and 'kg_inputs' in model_kwargs:
        instance = {
            'input_ids': encoded['input_ids'][0],
            'labels': encoded['input_ids'][0],
        }
        if 'sg' in record:
            instance['sg'] = record['sg']
        beam_count = int(kwargs['num_beams'])
        expanded_input_ids = encoded['input_ids'].expand(beam_count, -1).cpu()
        expanded_instances = [dict(instance) for _ in range(beam_count)]
        kg_batch = hit1_eval.build_kg_batch(
            expanded_instances,
            expanded_input_ids,
            pad_id=bot.tokenizer.pad_token_id,
            kg_cfg=cfg.llm.kg_adapter)
        if kg_batch is not None:
            device = bot._get_model_input_device()
            model_kwargs['kg_inputs'] = hit1_eval._move_to_device(
                kg_batch, device)
            model_kwargs['sg'] = model_kwargs['kg_inputs']
    kwargs['num_return_sequences'] = TOP_K
    output_ids = bot.model.generate(**encoded, **model_kwargs, **kwargs)
    prompt_length = encoded['input_ids'].shape[1]
    return [
        bot.tokenizer.decode(sequence[prompt_length:],
                             skip_special_tokens=True)
        for sequence in output_ids[:TOP_K]
    ]


def _rerank_top3(bot, cfg, record, candidates):
    scored = [
        (hit1_eval._score_candidate(bot, cfg, record, candidate), candidate)
        for candidate in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored[:TOP_K]], scored


def _write_summary(path, row):
    fieldnames = [
        'dataset', 'split', 'checkpoint', 'data_file', 'base_model',
        'total_params', 'trainable_params', 'total', 'hit@1_correct',
        'hit@1', 'hit@3_correct', 'hit@3', 'exact@1_correct', 'exact@1',
        'exact@3_correct', 'exact@3', 'match_mode', 'ranking_method',
        'num_beams'
    ]
    with open(path, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    cfg = hit1_eval._load_cfg(args)
    args = _apply_defaults(args, cfg)
    dataset_name = hit1_eval._infer_dataset_name(cfg, args.dataset)
    ckpt_path = cfg.federate.save_to
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {ckpt_path}. Pass --ckpt explicitly.')

    bot = hit1_eval.ExactCheckpointBot(cfg, ckpt_path)
    split_path, records = hit1_eval._load_records(
        cfg, dataset_name, args.split, bot.tokenizer)
    records = hit1_eval._apply_instruction(records, args.instruction)
    if args.limit > 0:
        records = records[:args.limit]

    candidate_map = hit1_eval._load_candidate_map(args.candidate_file)
    unlabeled_num = sum(
        1 for record in records if not str(record.get('target', '')).strip())
    if unlabeled_num > 0 and not args.allow_unlabeled:
        raise ValueError(
            f'{dataset_name.upper()} {args.split} contains {unlabeled_num} '
            'records without gold answers. Use a labeled validation split; '
            'Hit@1 and Hit@3 cannot be reported on an unlabeled test split.')

    output_path = args.output or os.path.join(
        cfg.outdir, f'{dataset_name}_{args.split}_hit3_predictions.jsonl')
    summary_path = args.summary or os.path.join(
        cfg.outdir, f'{dataset_name}_{args.split}_hit3_summary.csv')
    hit1_eval._ensure_parent(output_path)
    hit1_eval._ensure_parent(summary_path)

    generation_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'num_beams': args.num_beams,
        'num_return_sequences': TOP_K,
        'early_stopping': True,
        'use_cache': False,
        'do_sample': args.temperature > 0,
        'temperature': args.temperature if args.temperature > 0 else None,
        'top_p': args.top_p,
        'pad_token_id': bot.tokenizer.pad_token_id,
        'eos_token_id': bot.tokenizer.eos_token_id,
    }
    generation_kwargs = {
        key: value for key, value in generation_kwargs.items()
        if value is not None
    }

    counters = {
        'hit@1': 0,
        'hit@3': 0,
        'exact@1': 0,
        'exact@3': 0,
    }
    ranking_methods = set()
    with open(output_path, 'w', encoding='utf-8') as fout:
        for idx, record in enumerate(
                tqdm(records, desc='Evaluating Hit@1/Hit@3')):
            candidates = hit1_eval._record_candidates(
                record, idx, candidate_map, args.candidate_topk)
            candidate_scores = []
            if args.rerank_candidates and candidates:
                predictions, candidate_scores = _rerank_top3(
                    bot, cfg, record, candidates)
                ranking_method = 'candidate_log_likelihood'
            else:
                predictions = _generate_top3(
                    bot, cfg, record, generation_kwargs)
                ranking_method = 'beam_search'
            ranking_methods.add(ranking_method)

            answers = hit1_eval._record_answers(record)
            result = _evaluate_predictions(
                predictions, answers, args.match_mode)
            for metric in counters:
                counters[metric] += result[metric]

            fout.write(json.dumps({
                'idx': idx,
                'question': record['context'],
                'predictions': predictions,
                'predictions_clean': [
                    hit1_eval._strip_prediction(item) for item in predictions
                ],
                'predictions_norm': result['predictions_norm'],
                'answers': answers,
                'answers_norm': result['answers_norm'],
                'hit@1': result['hit@1'],
                'hit@3': result['hit@3'],
                'exact@1': result['exact@1'],
                'exact@3': result['exact@3'],
                'rank_hits': result['rank_hits'],
                'rank_exact_hits': result['rank_exact_hits'],
                'match_mode': args.match_mode,
                'ranking_method': ranking_method,
                'candidates': candidates,
                'candidate_scores': candidate_scores[:10],
            }, ensure_ascii=False) + '\n')

    total = len(records)
    percentages = {
        metric: 0.0 if total == 0 else 100.0 * value / total
        for metric, value in counters.items()
    }
    total_params, trainable_params = hit1_eval._count_parameters(bot.model)
    ranking_method = ','.join(sorted(ranking_methods)) or 'none'
    _write_summary(summary_path, {
        'dataset': dataset_name,
        'split': args.split,
        'checkpoint': cfg.federate.save_to,
        'data_file': split_path,
        'base_model': cfg.model.type,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total': total,
        'hit@1_correct': counters['hit@1'],
        'hit@1': f'{percentages["hit@1"]:.2f}',
        'hit@3_correct': counters['hit@3'],
        'hit@3': f'{percentages["hit@3"]:.2f}',
        'exact@1_correct': counters['exact@1'],
        'exact@1': f'{percentages["exact@1"]:.2f}',
        'exact@3_correct': counters['exact@3'],
        'exact@3': f'{percentages["exact@3"]:.2f}',
        'match_mode': args.match_mode,
        'ranking_method': ranking_method,
        'num_beams': args.num_beams,
    })

    print(f'{dataset_name.upper()} {args.split} Hit@1: '
          f'{percentages["hit@1"]:.2f} ({counters["hit@1"]}/{total})')
    print(f'{dataset_name.upper()} {args.split} Hit@3: '
          f'{percentages["hit@3"]:.2f} ({counters["hit@3"]}/{total})')
    print(f'Exact@1: {percentages["exact@1"]:.2f}')
    print(f'Exact@3: {percentages["exact@3"]:.2f}')
    print(f'Ranking method: {ranking_method}')
    print(f'Predictions: {output_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
