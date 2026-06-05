import json
import os
import random
import re
import zipfile
import hashlib

from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataset.llm_dataset import LLMDataset


def _merge_extra_fields(base_item, raw_item, excluded_keys):
    merged = dict(base_item)
    for key, value in raw_item.items():
        if key not in excluded_keys and key not in merged:
            merged[key] = value
    return merged


def _kg_enabled(config):
    try:
        return bool(config.llm.kg_adapter.use)
    except Exception:
        return False


def _kg_vocab_size(config, field, default_value):
    try:
        value = int(getattr(config.llm.kg_adapter, field, default_value))
    except Exception:
        value = default_value
    return max(value, 2)


def _stable_vocab_id(text, vocab_size):
    if text in [None, '']:
        return 0
    digest = hashlib.md5(str(text).encode('utf-8')).hexdigest()
    return int(digest[:12], 16) % (vocab_size - 1) + 1


def _stable_relation_id(text, config):
    if text in [None, '']:
        return 0
    digest = hashlib.md5(str(text).encode('utf-8')).hexdigest()
    try:
        gnn_backend = str(getattr(config.llm.kg_adapter,
                                  'gnn_backend',
                                  'lite')).lower()
    except Exception:
        gnn_backend = 'lite'
    if gnn_backend == 'paper':
        try:
            namespace = int(getattr(config.llm.kg_adapter,
                                    'num_relations',
                                    1))
        except Exception:
            namespace = 1
        namespace = max(namespace, 1)
        return int(digest[:12], 16) % namespace
    edge_vocab_size = _kg_vocab_size(config, 'edge_vocab_size', 50000)
    return int(digest[:12], 16) % (edge_vocab_size - 1) + 1


def _encode_text_ids(tokenizer, text, max_len=16):
    if text in [None, '']:
        return [0]
    tokenized = tokenizer(str(text),
                          add_special_tokens=False,
                          truncation=True,
                          max_length=max_len)
    input_ids = tokenized.get('input_ids', [])
    if len(input_ids) == 0:
        return [0]
    return [int(token_id) for token_id in input_ids]


def _find_subsequence(sequence, pattern):
    if len(pattern) == 0 or len(sequence) < len(pattern):
        return []
    for start in range(len(sequence) - len(pattern) + 1):
        if sequence[start:start + len(pattern)] == pattern:
            return list(range(start, start + len(pattern)))
    return []


def _build_align_mask(tokenizer, context_text, node_mentions):
    context_ids = _encode_text_ids(tokenizer, context_text, max_len=4096)
    align_mask = [[0 for _ in range(len(node_mentions))]
                  for _ in range(len(context_ids))]
    for node_idx, mention in enumerate(node_mentions):
        mention_ids = _encode_text_ids(tokenizer, mention, max_len=32)
        if mention_ids == [0]:
            continue
        matched_positions = _find_subsequence(context_ids, mention_ids)
        for pos in matched_positions:
            align_mask[pos][node_idx] = 1
    return align_mask


def _build_token_entity_ids_from_align(align_mask):
    token_entity_ids = []
    for row in align_mask:
        refs = [idx for idx, value in enumerate(row) if value]
        if len(refs) == 0:
            token_entity_ids.append([-1])
        else:
            token_entity_ids.append(refs)
    return token_entity_ids


def _build_webqsp_sg(item, context_text, tokenizer, config):
    parses = item.get('Parses', item.get('parses', []))
    parse = parses[0] if isinstance(parses, list) and len(parses) > 0 else {}
    if not isinstance(parse, dict):
        parse = {}
    entity_vocab_size = _kg_vocab_size(config, 'entity_vocab_size', 50000)

    topic_mid = parse.get('TopicEntityMid', item.get('topic_entity_mid', ''))
    topic_name = parse.get('TopicEntityName',
                           parse.get('PotentialTopicEntityMention', 'topic'))
    mention = parse.get('PotentialTopicEntityMention', topic_name)
    chain = parse.get('InferentialChain', [])
    constraints = parse.get('Constraints', [])
    if chain is None:
        chain = []
    elif isinstance(chain, str):
        chain = [chain] if chain else []
    elif not isinstance(chain, list):
        try:
            chain = list(chain)
        except TypeError:
            chain = []
    chain = [str(relation) for relation in chain if relation not in [None, '']]
    if constraints is None:
        constraints = []
    elif isinstance(constraints, dict):
        constraints = [constraints]
    elif not isinstance(constraints, list):
        try:
            constraints = list(constraints)
        except TypeError:
            constraints = []
    constraints = [constraint for constraint in constraints
                   if isinstance(constraint, dict)]

    node_specs = [{
        'key': topic_mid or topic_name or 'topic',
        'text': topic_name or mention or 'topic entity',
        'mention': mention or topic_name or '',
        'type': 1,
    }]
    chain_node_indices = []
    for hop_idx, relation in enumerate(chain):
        is_last = hop_idx == len(chain) - 1
        node_text = 'answer variable' if is_last else \
            f'intermediate variable {hop_idx + 1}'
        node_specs.append({
            'key': f'var:{hop_idx + 1}:{relation}',
            'text': node_text,
            'mention': '',
            'type': 2 if not is_last else 3,
        })
        chain_node_indices.append(len(node_specs) - 1)

    edges = []
    previous_idx = 0
    for hop_idx, relation in enumerate(chain):
        current_idx = chain_node_indices[hop_idx]
        edges.append((previous_idx, current_idx, relation))
        previous_idx = current_idx

    for constraint in constraints:
        constraint_key = constraint.get('Argument',
                                        constraint.get('EntityName',
                                                       'constraint'))
        constraint_text = constraint.get('EntityName',
                                         constraint.get('Argument',
                                                        'constraint'))
        node_specs.append({
            'key': constraint_key,
            'text': constraint_text,
            'mention': '',
            'type': 4,
        })
        target_idx = len(node_specs) - 1
        source_node_index = int(constraint.get('SourceNodeIndex', -1))
        if 0 <= source_node_index < len(chain_node_indices):
            source_idx = chain_node_indices[source_node_index]
        else:
            source_idx = previous_idx if len(chain_node_indices) > 0 else 0
        relation = constraint.get('NodePredicate', 'constraint')
        edges.append((source_idx, target_idx, relation))

    if len(edges) == 0:
        node_specs.append({
            'key': 'answer_var',
            'text': 'answer variable',
            'mention': '',
            'type': 3,
        })
        edges.append((0, len(node_specs) - 1, 'question.related_to'))

    node_ids = [_stable_vocab_id(node['key'], entity_vocab_size)
                for node in node_specs]
    node_type = [node['type'] for node in node_specs]
    edge_index = [
        [src for src, _, _ in edges],
        [dst for _, dst, _ in edges],
    ]
    edge_type = [_stable_relation_id(rel, config) for _, _, rel in edges]
    nid2swid = [_encode_text_ids(tokenizer, node['text']) for node in node_specs]
    eid2swid = [_encode_text_ids(tokenizer, rel) for _, _, rel in edges]
    align_mask = _build_align_mask(tokenizer,
                                   context_text,
                                   [node['mention'] for node in node_specs])

    return {
        'node_ids': node_ids,
        'node_type': node_type,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'nid2swid': nid2swid,
        'eid2swid': eid2swid,
        'align_mask': align_mask,
        'token_entity_ids': _build_token_entity_ids_from_align(align_mask),
    }


def _strip_ns_token(token):
    token = str(token).strip()
    if token.startswith('ns:'):
        return token[3:]
    return token


def _parse_sparql_triples(sparql):
    if sparql in [None, '']:
        return []
    triple_pattern = re.compile(
        r'(?P<src>(?:ns:[^\s]+|\?[A-Za-z_][\w]*))\s+'
        r'(?P<rel>ns:[^\s]+)\s+'
        r'(?P<dst>(?:ns:[^\s]+|\?[A-Za-z_][\w]*))\s+\.')
    triples = []
    for match in triple_pattern.finditer(str(sparql)):
        src = _strip_ns_token(match.group('src'))
        rel = _strip_ns_token(match.group('rel'))
        dst = _strip_ns_token(match.group('dst'))
        triples.append((src, rel, dst))
    return triples


def _extract_cwq_answers(item):
    raw_answers = item.get('answers', item.get('answer', []))
    answers = []
    if isinstance(raw_answers, list):
        for answer in raw_answers:
            if isinstance(answer, dict):
                value = answer.get('answer',
                                   answer.get('answer_id',
                                              answer.get('entity_name', None)))
            else:
                value = answer
            if value not in [None, '']:
                answers.append(str(value))
    elif raw_answers not in [None, '']:
        answers.append(str(raw_answers))
    dedup_answers = []
    for answer in answers:
        if answer not in dedup_answers:
            dedup_answers.append(answer)
    return dedup_answers


def _extract_cwq_answer_aliases(item):
    raw_answers = item.get('answers', item.get('answer', []))
    aliases = []
    if not isinstance(raw_answers, list):
        return aliases

    for answer in raw_answers:
        if not isinstance(answer, dict):
            continue
        values = []
        canonical = answer.get('answer', None)
        if canonical not in [None, '']:
            values.append(canonical)
        raw_aliases = answer.get('aliases', [])
        if isinstance(raw_aliases, list):
            values.extend(raw_aliases)
        elif raw_aliases not in [None, '']:
            values.append(raw_aliases)

        for value in values:
            value = str(value).strip()
            if value and value not in aliases:
                aliases.append(value)
    return aliases


def _build_cwq_sg(item, context_text, tokenizer, config):
    sparql = item.get('sparql', item.get('Sparql', ''))
    triples = _parse_sparql_triples(sparql)
    entity_vocab_size = _kg_vocab_size(config, 'entity_vocab_size', 50000)
    edge_vocab_size = _kg_vocab_size(config, 'edge_vocab_size', 50000)
    answer_nodes = {}
    for answer in item.get('answers', []):
        if isinstance(answer, dict) and answer.get('answer_id', None):
            answer_nodes[str(answer['answer_id'])] = answer.get(
                'answer', answer['answer_id'])

    node_specs = []
    node_index = {}

    def ensure_node(node_key):
        if node_key in node_index:
            return node_index[node_key]
        if node_key.startswith('?'):
            node_text = f'variable {node_key[1:]}'
            mention = ''
            node_type = 2
        else:
            node_text = answer_nodes.get(node_key, node_key)
            mention = node_text if node_text in context_text else ''
            node_type = 3 if node_key in answer_nodes else 1
        node_index[node_key] = len(node_specs)
        node_specs.append({
            'key': node_key,
            'text': node_text,
            'mention': mention,
            'type': node_type,
        })
        return node_index[node_key]

    edges = []
    for src, rel, dst in triples:
        src_idx = ensure_node(src)
        dst_idx = ensure_node(dst)
        edges.append((src_idx, dst_idx, rel))

    if len(node_specs) == 0:
        question = item.get('question',
                            item.get('machine_question',
                                     item.get('webqsp_question',
                                              'complex question')))
        node_specs.append({
            'key': f'question::{question}',
            'text': str(question),
            'mention': str(question),
            'type': 1,
        })

    node_ids = [_stable_vocab_id(node['key'], entity_vocab_size)
                for node in node_specs]
    node_type = [node['type'] for node in node_specs]
    edge_index = [
        [src for src, _, _ in edges],
        [dst for _, dst, _ in edges],
    ] if edges else [[], []]
    edge_type = [_stable_relation_id(rel, config) for _, _, rel in edges]
    nid2swid = [_encode_text_ids(tokenizer, node['text']) for node in node_specs]
    eid2swid = [_encode_text_ids(tokenizer, rel) for _, _, rel in edges]
    align_mask = _build_align_mask(tokenizer,
                                   context_text,
                                   [node['mention'] for node in node_specs])

    return {
        'node_ids': node_ids,
        'node_type': node_type,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'nid2swid': nid2swid,
        'eid2swid': eid2swid,
        'align_mask': align_mask,
        'token_entity_ids': _build_token_entity_ids_from_align(align_mask),
    }


def _cfg_data_args(config):
    if hasattr(config.data, 'args') and len(config.data.args) > 0:
        return config.data.args[0]
    return {}


def _join_path(root, path):
    if path is None or path == '':
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _first_existing(root, candidates):
    for candidate in candidates:
        path = _join_path(root, candidate)
        if path is not None and os.path.exists(path):
            return path
    return None


def _download_to_path(url, target_path):
    folder = os.path.dirname(target_path)
    os.makedirs(folder, exist_ok=True)
    downloaded = download_url(url, folder=folder)
    if downloaded != target_path:
        if os.path.exists(target_path):
            return target_path
        os.replace(downloaded, target_path)
    return target_path


def _maybe_download_files(root, args):
    download_files = args.get('download_files', None)
    if not isinstance(download_files, dict):
        return
    for rel_path, url in download_files.items():
        target_path = _join_path(root, rel_path)
        if target_path is None or os.path.exists(target_path):
            continue
        _download_to_path(url, target_path)


def _maybe_download_archive(root, args):
    archive_url = args.get('archive_url', None)
    if archive_url in [None, '']:
        return

    extract_to = _join_path(root, args.get('extract_to', ''))
    if extract_to is None:
        extract_to = root
    marker_rel = args.get('archive_marker', None)
    marker_path = _join_path(root, marker_rel) if marker_rel else None
    if marker_path is not None and os.path.exists(marker_path):
        return

    archive_path = download_url(archive_url, folder=root)
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(archive_path, 'r') as archive:
        archive.extractall(extract_to)


def _read_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _read_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _train_val_split(records, val_ratio, seed=12345):
    if not records:
        return [], []
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    val_size = max(1, int(len(records) * val_ratio)) if len(records) > 1 \
        and val_ratio > 0 else 0
    val_index = set(indices[:val_size])
    train_records, val_records = [], []
    for idx, record in enumerate(records):
        if idx in val_index:
            val_records.append(record)
        else:
            train_records.append(record)
    return train_records, val_records


def _extract_webqsp_answers(item):
    if isinstance(item.get('answers', None), list) and item['answers']:
        return [str(answer) for answer in item['answers'] if answer not in
                [None, '']]

    parses = item.get('Parses', item.get('parses', []))
    answers = []
    for parse in parses:
        for answer in parse.get('Answers', parse.get('answers', [])):
            if isinstance(answer, dict):
                value = answer.get('EntityName',
                                   answer.get('AnswerArgument',
                                              answer.get('answer', None)))
            else:
                value = answer
            if value not in [None, '']:
                answers.append(str(value))
    dedup_answers = []
    for answer in answers:
        if answer not in dedup_answers:
            dedup_answers.append(answer)
    return dedup_answers


def _format_webqsp_item(item, tokenizer=None, config=None):
    question = item.get('ProcessedQuestion',
                        item.get('RawQuestion',
                                 item.get('question',
                                          item.get('Question', None))))
    if question is None:
        return None
    answers = _extract_webqsp_answers(item)
    if not answers:
        return None
    record = _merge_extra_fields(
        dict(context=f'Question: {str(question).strip()}\nAnswer:',
             target='; '.join(answers),
             category=item.get('category', 'webquestionssp')),
        item,
        excluded_keys={
            'ProcessedQuestion', 'RawQuestion', 'question', 'Question',
            'answers', 'Parses', 'parses', 'category'
        })
    if tokenizer is not None and config is not None and _kg_enabled(config):
        record['sg'] = _build_webqsp_sg(item, record['context'], tokenizer,
                                        config)
    return record


def _format_cwq_item(item, tokenizer=None, config=None, split_name=None):
    question = item.get('question',
                        item.get('machine_question',
                                 item.get('webqsp_question', None)))
    if question is None:
        return None
    answers = _extract_cwq_answers(item)
    split_flag = str(item.get('split', split_name or '')).lower()
    if not answers and split_flag != 'test':
        return None
    target = '; '.join(answers) if answers else ''
    answer_aliases = _extract_cwq_answer_aliases(item)
    record = _merge_extra_fields(
        dict(context=f'Question: {str(question).strip()}\nAnswer:',
             target=target,
             answer_aliases=answer_aliases,
             category=item.get('compositionality_type',
                               item.get('category', 'complexwebquestions'))),
        item,
        excluded_keys={
            'question', 'machine_question', 'webqsp_question', 'answers',
            'answer', 'answer_aliases', 'category'
        })
    if tokenizer is not None and config is not None and _kg_enabled(config):
        record['sg'] = _build_cwq_sg(item, record['context'], tokenizer,
                                     config)
    return record


def _build_llm_dataset(records, tokenizer):
    return LLMDataset(records,
                      tokenizer,
                      prompt_no_input='{context}',
                      prompt_input='{context}',
                      output_tag='target')


def load_webquestionssp_llm_dataset(config, tokenizer):
    args = _cfg_data_args(config)
    root = config.data.root
    _maybe_download_files(root, args)
    _maybe_download_archive(root, args)

    train_path = _first_existing(root, [
        args.get('train_file', None),
        'WebQSP/data/WebQSP.train.json',
        'webqsp/WebQSP.train.json',
        'WebQSP.train.json',
        'webquestionssp/WebQSP.train.json',
        'train.json',
    ])
    val_path = _first_existing(root, [
        args.get('val_file', None),
        'WebQSP/data/WebQSP.dev.json',
        'webqsp/WebQSP.dev.json',
        'WebQSP.dev.json',
        'webqsp/WebQSP.valid.json',
        'WebQSP.valid.json',
        'webquestionssp/WebQSP.dev.json',
        'dev.json',
        'valid.json',
    ])
    test_path = _first_existing(root, [
        args.get('test_file', None),
        'WebQSP/data/WebQSP.test.json',
        'webqsp/WebQSP.test.json',
        'WebQSP.test.json',
        'webquestionssp/WebQSP.test.json',
        'test.json',
    ])

    def _load_webqsp_split(path):
        raw = _read_json(path)
        questions = raw.get('Questions', raw if isinstance(raw, list) else [])
        records = [_format_webqsp_item(item, tokenizer, config)
                   for item in questions]
        return [item for item in records if item is not None]

    if train_path is not None:
        train_records = _load_webqsp_split(train_path)
        if val_path is not None:
            val_records = _load_webqsp_split(val_path)
        else:
            total_ratio = float(config.data.splits[0] + config.data.splits[1])
            val_ratio = 0.0 if total_ratio <= 0 else \
                float(config.data.splits[1]) / total_ratio
            train_records, val_records = _train_val_split(train_records,
                                                          val_ratio)
        if test_path is not None:
            test_records = _load_webqsp_split(test_path)
        else:
            test_records = []

        return (_build_llm_dataset(train_records, tokenizer),
                _build_llm_dataset(val_records, tokenizer),
                _build_llm_dataset(test_records, tokenizer))

    hf_hub = args.get('hf_hub', args.get('path', 'ml1996/webqsp'))

    import datasets
    dataset = datasets.load_dataset(hf_hub)
    train_split = 'train' if 'train' in dataset else list(dataset.keys())[0]
    train_records = [_format_webqsp_item(item, tokenizer, config)
                     for item in dataset[train_split]]
    train_records = [item for item in train_records if item is not None]

    if 'validation' in dataset:
        val_records = [_format_webqsp_item(item, tokenizer, config)
                       for item in dataset['validation']]
        val_records = [item for item in val_records if item is not None]
    else:
        total_ratio = float(config.data.splits[0] + config.data.splits[1])
        val_ratio = 0.0 if total_ratio <= 0 else \
            float(config.data.splits[1]) / total_ratio
        train_records, val_records = _train_val_split(train_records, val_ratio)

    if 'test' in dataset:
        test_records = [_format_webqsp_item(item, tokenizer, config)
                        for item in dataset['test']]
        test_records = [item for item in test_records if item is not None]
    else:
        test_records = []

    return (_build_llm_dataset(train_records, tokenizer),
            _build_llm_dataset(val_records, tokenizer),
            _build_llm_dataset(test_records, tokenizer))


def load_complexwebquestions_llm_dataset(config, tokenizer):
    args = _cfg_data_args(config)
    root = config.data.root
    _maybe_download_files(root, args)
    _maybe_download_archive(root, args)

    train_path = _first_existing(root, [
        args.get('train_file', None),
        'CWQ/ComplexWebQuestions_train.json',
        'CWQ/train.json',
        'cwq/ComplexWebQuestions_train.json',
        'cwq/train.json',
        'ComplexWebQuestions_train.json',
        'train.json',
    ])
    val_path = _first_existing(root, [
        args.get('val_file', None),
        args.get('dev_file', None),
        'CWQ/ComplexWebQuestions_dev.json',
        'CWQ/dev.json',
        'cwq/ComplexWebQuestions_dev.json',
        'cwq/dev.json',
        'ComplexWebQuestions_dev.json',
        'dev.json',
        'valid.json',
    ])
    test_path = _first_existing(root, [
        args.get('test_file', None),
        'CWQ/ComplexWebQuestions_test.json',
        'CWQ/test.json',
        'cwq/ComplexWebQuestions_test.json',
        'cwq/test.json',
        'ComplexWebQuestions_test.json',
        'test.json',
    ])

    def _load_cwq_split(path):
        raw = _read_json(path)
        if isinstance(raw, dict):
            raw = raw.get('questions', raw.get('Questions', raw.get('data', [])))
        split_name = 'train'
        lower_name = os.path.basename(path).lower()
        if 'test' in lower_name:
            split_name = 'test'
        elif 'dev' in lower_name or 'valid' in lower_name:
            split_name = 'validation'
        records = [_format_cwq_item(item, tokenizer, config, split_name)
                   for item in raw] if isinstance(raw, list) else []
        return [item for item in records if item is not None]

    if train_path is not None:
        train_records = _load_cwq_split(train_path)
        if val_path is not None:
            val_records = _load_cwq_split(val_path)
        else:
            total_ratio = float(config.data.splits[0] + config.data.splits[1])
            val_ratio = 0.0 if total_ratio <= 0 else \
                float(config.data.splits[1]) / total_ratio
            train_records, val_records = _train_val_split(train_records,
                                                          val_ratio)
        if test_path is not None:
            test_records = _load_cwq_split(test_path)
        else:
            test_records = []

        return (_build_llm_dataset(train_records, tokenizer),
                _build_llm_dataset(val_records, tokenizer),
                _build_llm_dataset(test_records, tokenizer))

    hf_hub = args.get('hf_hub', args.get('path', 'drt/complex_web_questions'))

    import datasets
    dataset = datasets.load_dataset(hf_hub, 'complex_web_questions')
    train_records = [_format_cwq_item(item, tokenizer, config, 'train')
                     for item in dataset['train']]
    train_records = [item for item in train_records if item is not None]

    if 'validation' in dataset:
        val_records = [_format_cwq_item(item, tokenizer, config, 'validation')
                       for item in dataset['validation']]
        val_records = [item for item in val_records if item is not None]
    elif 'dev' in dataset:
        val_records = [_format_cwq_item(item, tokenizer, config, 'validation')
                       for item in dataset['dev']]
        val_records = [item for item in val_records if item is not None]
    else:
        total_ratio = float(config.data.splits[0] + config.data.splits[1])
        val_ratio = 0.0 if total_ratio <= 0 else \
            float(config.data.splits[1]) / total_ratio
        train_records, val_records = _train_val_split(train_records, val_ratio)

    if 'test' in dataset:
        test_records = [_format_cwq_item(item, tokenizer, config, 'test')
                        for item in dataset['test']]
        test_records = [item for item in test_records if item is not None]
    else:
        test_records = []

    return (_build_llm_dataset(train_records, tokenizer),
            _build_llm_dataset(val_records, tokenizer),
            _build_llm_dataset(test_records, tokenizer))
