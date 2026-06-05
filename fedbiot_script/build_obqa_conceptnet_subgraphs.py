import argparse
import ast
import csv
import gzip
import hashlib
import json
import os
import re
import string
from collections import defaultdict, deque

import pandas as pd
import torch
from tqdm import tqdm

try:
    from torch_geometric.data import Data
except Exception:
    Data = None

from transformers import AutoTokenizer


OBQA_INSTRUCTION = (
    "You are an honest and helpful AI assistant. Now you're going to do a "
    "multiple choice task, you will be given a question and options, and you "
    "need to select the correct option. First output the correct answer."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build OpenBookQA ConceptNet subgraph .pt files.')
    parser.add_argument('--obqa-root',
                        default='data/openbookQA/main',
                        help='Directory containing train/validation/test '
                        'OpenBookQA parquet files.')
    parser.add_argument('--conceptnet',
                        required=True,
                        help='ConceptNet assertions CSV/TSV path.')
    parser.add_argument('--output-dir',
                        default='data',
                        help='Directory for generated .pt files and vocab.')
    parser.add_argument('--version',
                        default='obqa_conceptnet_3hop',
                        help='Output version name, e.g. train_<version>.pt.')
    parser.add_argument('--model-path',
                        required=True,
                        help='Tokenizer model path/name used by training.')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--max-ngram', type=int, default=5)
    parser.add_argument('--hop', type=int, default=3)
    parser.add_argument('--max-seeds', type=int, default=32)
    parser.add_argument('--max-nodes', type=int, default=250)
    parser.add_argument('--max-edges', type=int, default=800)
    parser.add_argument('--max-neighbors-per-node', type=int, default=40)
    parser.add_argument('--min-concept-len', type=int, default=2)
    parser.add_argument('--entity-vocab-size', type=int, default=50000)
    parser.add_argument('--relation-vocab-size', type=int, default=64)
    parser.add_argument('--undirected', action='store_true',
                        help='Add reverse adjacency during retrieval.')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Debug limit per split.')
    return parser.parse_args()


def normalize_text(text):
    text = str(text).lower()
    text = text.replace('_', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())


def concept_from_uri(uri):
    uri = str(uri)
    if not uri.startswith('/c/en/'):
        return None
    pieces = uri.split('/')
    if len(pieces) < 4:
        return None
    return normalize_text(pieces[3])


def relation_from_uri(uri):
    uri = str(uri).strip()
    if uri.startswith('/r/'):
        uri = uri[3:]
    return uri


def relation_text(relation):
    relation = re.sub(r'(?<!^)(?=[A-Z])', ' ', str(relation))
    return normalize_text(relation)


def stable_vocab_id(text, vocab_size):
    vocab_size = max(int(vocab_size), 2)
    digest = hashlib.md5(str(text).encode('utf-8')).hexdigest()
    return int(digest[:12], 16) % (vocab_size - 1) + 1


def parse_weight(data):
    if data in [None, '']:
        return 1.0
    if isinstance(data, dict):
        value = data.get('weight', 1.0)
    else:
        try:
            value = json.loads(data).get('weight', 1.0)
        except Exception:
            try:
                value = ast.literal_eval(str(data)).get('weight', 1.0)
            except Exception:
                value = 1.0
    try:
        return float(value)
    except Exception:
        return 1.0


def load_conceptnet(path, min_concept_len=2, undirected=False):
    node2id = {}
    rel2id = {}
    edges = []
    adjacency = defaultdict(list)

    def get_node_id(concept):
        if concept not in node2id:
            node2id[concept] = len(node2id)
        return node2id[concept]

    def get_rel_id(relation):
        if relation not in rel2id:
            rel2id[relation] = len(rel2id)
        return rel2id[relation]

    open_fn = gzip.open if str(path).endswith('.gz') else open
    with open_fn(path, 'rt', encoding='utf-8') as fin:
        sample = fin.read(4096)
        fin.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters='\t,')
        reader = csv.reader(fin, dialect)
        for row in tqdm(reader, desc='Loading ConceptNet'):
            if len(row) < 4:
                continue
            if row[0] == 'uri' and row[1] == 'rel':
                continue
            rel = relation_from_uri(row[1])
            src = concept_from_uri(row[2])
            dst = concept_from_uri(row[3])
            if not src or not dst:
                continue
            if len(src) < min_concept_len or len(dst) < min_concept_len:
                continue
            weight = parse_weight(row[4]) if len(row) > 4 else 1.0
            src_id = get_node_id(src)
            dst_id = get_node_id(dst)
            rel_id = get_rel_id(rel)
            edge_id = len(edges)
            edges.append((src_id, dst_id, rel_id, weight))
            adjacency[src_id].append((dst_id, rel_id, weight, edge_id))
            if undirected:
                adjacency[dst_id].append((src_id, rel_id, weight, edge_id))

    for node_id in list(adjacency.keys()):
        adjacency[node_id].sort(key=lambda item: item[2], reverse=True)
    id2node = [None] * len(node2id)
    for concept, node_id in node2id.items():
        id2node[node_id] = concept
    id2rel = [None] * len(rel2id)
    for relation, rel_id in rel2id.items():
        id2rel[rel_id] = relation

    return node2id, id2node, rel2id, id2rel, edges, adjacency


def tokenize_words(text):
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(text).lower())


def extract_ngram_seeds(text, node2id, max_ngram=5, max_seeds=32):
    words = tokenize_words(text)
    seeds = []
    seen = set()
    for ngram_len in range(max_ngram, 0, -1):
        for start in range(0, max(0, len(words) - ngram_len + 1)):
            concept = normalize_text(' '.join(words[start:start + ngram_len]))
            if concept in node2id and concept not in seen:
                seen.add(concept)
                seeds.append(node2id[concept])
                if len(seeds) >= max_seeds:
                    return seeds
    return seeds


def retrieve_subgraph(seeds, adjacency, edges, hop, max_nodes, max_edges,
                      max_neighbors_per_node):
    if not seeds:
        return [], [], []

    selected_nodes = []
    node_seen = set()
    selected_edge_ids = []
    edge_seen = set()
    queue = deque()
    for seed in seeds:
        if seed not in node_seen:
            node_seen.add(seed)
            selected_nodes.append(seed)
            queue.append((seed, 0))

    while queue and len(selected_nodes) < max_nodes and \
            len(selected_edge_ids) < max_edges:
        node_id, depth = queue.popleft()
        if depth >= hop:
            continue
        for dst_id, _, _, edge_id in adjacency.get(node_id, [])[
                :max_neighbors_per_node]:
            if edge_id not in edge_seen:
                edge_seen.add(edge_id)
                selected_edge_ids.append(edge_id)
            if dst_id not in node_seen and len(selected_nodes) < max_nodes:
                node_seen.add(dst_id)
                selected_nodes.append(dst_id)
                queue.append((dst_id, depth + 1))
            if len(selected_edge_ids) >= max_edges:
                break

    local_id = {global_id: idx for idx, global_id in enumerate(selected_nodes)}
    local_edges = []
    edge_types = []
    for edge_id in selected_edge_ids:
        src, dst, rel_id, _ = edges[edge_id]
        if src in local_id and dst in local_id:
            local_edges.append((local_id[src], local_id[dst]))
            edge_types.append(rel_id)

    return selected_nodes, local_edges, edge_types


def encode_text_ids(tokenizer, text, max_len=16):
    ids = tokenizer(str(text),
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_len).get('input_ids', [])
    return [int(item) for item in ids] if ids else [
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ]


def build_sg(tokenizer, selected_nodes, local_edges, edge_types, id2node, id2rel,
             args):
    valid_nodes = []
    old_to_new = {}
    for old_idx, node_id in enumerate(selected_nodes):
        if 0 <= int(node_id) < len(id2node):
            old_to_new[old_idx] = len(valid_nodes)
            valid_nodes.append(int(node_id))

    if not valid_nodes:
        valid_nodes = [0] if len(id2node) > 0 else []

    edge_index = [[], []]
    valid_edge_types = []
    for (src, dst), rel_id in zip(local_edges, edge_types):
        if src in old_to_new and dst in old_to_new and \
                0 <= int(rel_id) < len(id2rel):
            edge_index[0].append(old_to_new[src])
            edge_index[1].append(old_to_new[dst])
            valid_edge_types.append(int(rel_id))

    if valid_nodes:
        x = torch.tensor([
            stable_vocab_id(id2node[node_id], args.entity_vocab_size)
            for node_id in valid_nodes
        ],
                         dtype=torch.long)
        nid2swid = [
            encode_text_ids(tokenizer, id2node[node_id])
            for node_id in valid_nodes
        ]
    else:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
            else 0
        x = torch.zeros(1, dtype=torch.long)
        nid2swid = [[pad_id]]

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_type = torch.tensor(
        [
            rel_id % max(int(args.relation_vocab_size), 1)
            for rel_id in valid_edge_types
        ],
        dtype=torch.long)
    eid2swid = [
        encode_text_ids(tokenizer, relation_text(id2rel[rel_id]))
        for rel_id in valid_edge_types
    ]

    if Data is None:
        return {
            'x': x,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'nid2swid': nid2swid,
            'eid2swid': eid2swid,
        }

    sg = Data(x=x, edge_index=edge_index)
    sg.edge_type = edge_type
    sg.nid2swid = nid2swid
    sg.eid2swid = eid2swid
    return sg


def normalize_choices(value):
    if isinstance(value, dict):
        labels = value.get('label', [])
        texts = value.get('text', [])
    else:
        labels = getattr(value, 'label', None)
        texts = getattr(value, 'text', None)
        if labels is None or texts is None:
            try:
                labels = value['label']
                texts = value['text']
            except Exception:
                labels, texts = [], []
    return list(labels), list(texts)


def format_prompt(question, labels, texts):
    options = '\n'.join(
        f'({label}) {text}' for label, text in zip(labels, texts))
    return f'{OBQA_INSTRUCTION}\nQ: {question}\n{options}\nA:'


def answer_text(answer_key, labels, texts):
    for label, text in zip(labels, texts):
        if str(label).strip() == str(answer_key).strip():
            return str(text).strip()
    return str(answer_key).strip()


def build_item(row, tokenizer, node2id, id2node, id2rel, edges, adjacency, args):
    question = str(row['question_stem']).strip()
    labels, texts = normalize_choices(row['choices'])
    target = answer_text(row['answerKey'], labels, texts)
    prompt = format_prompt(question, labels, texts)

    retrieval_text = ' '.join([question] + [str(text) for text in texts])
    if 'fact1' in row and str(row['fact1']).strip():
        retrieval_text += ' ' + str(row['fact1']).strip()

    seeds = extract_ngram_seeds(retrieval_text,
                                node2id,
                                max_ngram=args.max_ngram,
                                max_seeds=args.max_seeds)
    selected_nodes, local_edges, edge_types = retrieve_subgraph(
        seeds,
        adjacency,
        edges,
        hop=args.hop,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_neighbors_per_node=args.max_neighbors_per_node)
    sg = build_sg(tokenizer, selected_nodes, local_edges, edge_types, id2node,
                  id2rel, args)

    source = prompt
    full_text = prompt + ' ' + target + tokenizer.eos_token
    source_ids = tokenizer(source,
                           add_special_tokens=True,
                           truncation=True,
                           max_length=args.max_seq_length)['input_ids']
    input_ids = tokenizer(full_text,
                          add_special_tokens=True,
                          truncation=True,
                          max_length=args.max_seq_length)['input_ids']
    labels_tensor = torch.tensor(input_ids, dtype=torch.long)
    labels_tensor[:min(len(source_ids), labels_tensor.numel())] = -100

    return {
        'id': str(row.get('id', '')),
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': labels_tensor,
        'input_ids_no_response': torch.tensor(source_ids, dtype=torch.long),
        'answer': target,
        'answerKey': str(row['answerKey']).strip(),
        'question': question,
        'choices': texts,
        'seed_count': len(seeds),
        'category': f"openbookqa_{str(row['answerKey']).strip()}",
        'sg': sg,
    }


def find_split_file(root, split):
    names = {
        'train': ['train-00000-of-00001.parquet', 'train.parquet'],
        'dev': [
            'validation-00000-of-00001.parquet', 'validation.parquet',
            'dev.parquet'
        ],
        'test': ['test-00000-of-00001.parquet', 'test.parquet'],
    }
    for name in names[split]:
        path = os.path.join(root, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'Cannot find {split} parquet under {root}')


def build_split(split, tokenizer, kg, args):
    node2id, id2node, _, id2rel, edges, adjacency = kg
    path = find_split_file(args.obqa_root, split)
    df = pd.read_parquet(path)
    if args.limit > 0:
        df = df.iloc[:args.limit]

    items = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Building {split}'):
        items.append(
            build_item(row, tokenizer, node2id, id2node, id2rel, edges,
                       adjacency, args))
    return items


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              use_fast=False,
                                              local_files_only=os.path.isdir(
                                                  args.model_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    kg = load_conceptnet(args.conceptnet,
                         min_concept_len=args.min_concept_len,
                         undirected=args.undirected)
    node2id, id2node, rel2id, id2rel, _, _ = kg

    vocab_path = os.path.join(args.output_dir, f'{args.version}_kg_vocab.pt')
    torch.save({
        'node2id': node2id,
        'id2node': id2node,
        'rel2id': rel2id,
        'id2rel': id2rel,
    }, vocab_path)
    print(f'KG vocab saved to {vocab_path}')

    for split in ['train', 'dev', 'test']:
        items = build_split(split, tokenizer, kg, args)
        output_path = os.path.join(args.output_dir,
                                   f'{split}_{args.version}.pt')
        torch.save(items, output_path)
        avg_seeds = sum(item['seed_count'] for item in items) / max(
            len(items), 1)
        print(f'{split}: {len(items)} items, avg seeds {avg_seeds:.2f}')
        print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
