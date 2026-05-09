from typing import Any, Dict, List, Optional

import torch


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _clone_value(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {k: _clone_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(v) for v in value)
    return value


def _iter_keys(data):
    if data is None:
        return []
    if isinstance(data, dict):
        return list(data.keys())
    keys = getattr(data, 'keys', None)
    if callable(keys):
        try:
            return list(keys())
        except TypeError:
            pass
    if isinstance(keys, (list, tuple, set)):
        return list(keys)
    if hasattr(data, '__dict__'):
        return [k for k in data.__dict__.keys() if not k.startswith('_')]
    return []


def _get_value(data, key, default=None):
    if data is None:
        return default
    if isinstance(data, dict):
        return data.get(key, default)
    try:
        return data[key]
    except Exception:
        return getattr(data, key, default)


def _to_plain_dict(data):
    if data is None:
        return {}
    if isinstance(data, dict):
        return {k: _clone_value(v) for k, v in data.items()}
    result = {}
    for key in _iter_keys(data):
        result[key] = _clone_value(_get_value(data, key))
    return result


def _to_tensor(value,
               dtype: Optional[torch.dtype] = None,
               default: Optional[torch.Tensor] = None):
    if value is None:
        return default
    if torch.is_tensor(value):
        tensor = value.clone()
    else:
        tensor = torch.tensor(value)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _merge_sample_graph(instance):
    merged = _to_plain_dict(instance.get('sg', None))
    merged.update(_to_plain_dict(instance.get('kg_inputs', None)))
    return merged


def _empty_graph(pad_id):
    return {
        'x': torch.zeros(1, dtype=torch.long),
        'edge_index': torch.zeros((2, 0), dtype=torch.long),
        'edge_type': torch.zeros(0, dtype=torch.long),
        'node_type': torch.zeros(1, dtype=torch.long),
        'nid2swid': [[pad_id]],
        'eid2swid': [],
    }


def _normalize_edge_index(edge_index):
    edge_index = _to_tensor(edge_index, dtype=torch.long)
    if edge_index is None:
        return None
    if edge_index.dim() == 2 and edge_index.size(0) == 2:
        return edge_index
    if edge_index.dim() == 2 and edge_index.size(1) == 2:
        return edge_index.transpose(0, 1).contiguous()
    if edge_index.dim() == 1 and edge_index.numel() % 2 == 0:
        return edge_index.view(2, -1)
    return None


def _normalize_row_index(value, expected_count=None):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.tolist()
    if not isinstance(value, list):
        value = [value]
    rows = []
    for row in value:
        if torch.is_tensor(row):
            row = row.view(-1).tolist()
        elif isinstance(row, tuple):
            row = list(row)
        elif not isinstance(row, list):
            row = [row]
        rows.append([int(item) for item in row])
    if expected_count is not None:
        rows = rows[:expected_count]
        if len(rows) < expected_count:
            rows.extend([[] for _ in range(expected_count - len(rows))])
    return rows


def _normalize_token_entity_ids(value, seq_len):
    if value is None:
        return None
    tensor = _to_tensor(value, dtype=torch.long)
    if tensor is None:
        return None
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.size(0) > seq_len:
        tensor = tensor[:seq_len]
    if tensor.size(0) < seq_len:
        pad = tensor.new_full((seq_len - tensor.size(0), tensor.size(1)), -1)
        tensor = torch.cat([tensor, pad], dim=0)
    return tensor


def _normalize_align_mask(value, seq_len, node_count):
    if value is None:
        return None
    tensor = _to_tensor(value)
    if tensor is None:
        return None
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.size(1) > node_count:
        tensor = tensor[:, :node_count]
    if tensor.size(1) < node_count:
        pad = torch.zeros(tensor.size(0),
                          node_count - tensor.size(1),
                          dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=1)
    if tensor.size(0) > seq_len:
        tensor = tensor[:seq_len]
    if tensor.size(0) < seq_len:
        pad = torch.zeros(seq_len - tensor.size(0),
                          tensor.size(1),
                          dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    return tensor.bool()


def _clip_graph(sample_graph, max_node_num):
    node_ids = sample_graph['x']
    edge_index = sample_graph['edge_index']
    edge_type = sample_graph['edge_type']
    node_type = sample_graph['node_type']
    if max_node_num <= 0 or node_ids.numel() <= max_node_num:
        return sample_graph

    keep_mask = (edge_index[0] < max_node_num) & (edge_index[1] < max_node_num)
    sample_graph['edge_index'] = edge_index[:, keep_mask]
    sample_graph['edge_type'] = edge_type[keep_mask]
    sample_graph['x'] = node_ids[:max_node_num]
    sample_graph['node_type'] = node_type[:max_node_num]

    if sample_graph.get('nid2swid', None) is not None:
        sample_graph['nid2swid'] = sample_graph['nid2swid'][:max_node_num]
    if sample_graph.get('eid2swid', None) is not None:
        keep_edge_idx = torch.nonzero(keep_mask, as_tuple=False).view(-1).tolist()
        sample_graph['eid2swid'] = [
            sample_graph['eid2swid'][idx]
            for idx in keep_edge_idx
            if idx < len(sample_graph['eid2swid'])
        ]
    align_mask = sample_graph.get('align_mask', None)
    if align_mask is not None:
        sample_graph['align_mask'] = align_mask[:, :max_node_num]
    token_entity_ids = sample_graph.get('token_entity_ids', None)
    if token_entity_ids is not None:
        sample_graph['token_entity_ids'] = token_entity_ids.masked_fill(
            token_entity_ids >= max_node_num, -1)
    return sample_graph


def _build_trip_tensor(node_ids_list, edge_type_list, pad_id):
    if len(node_ids_list) == 0:
        return None
    trip_rows = []
    trip_num = []
    for node_ids, edge_ids in zip(node_ids_list, edge_type_list):
        flat_trip = torch.cat([node_ids, edge_ids], dim=0)
        trip_rows.append(flat_trip.tolist())
        trip_num.append([int(node_ids.numel()), int(edge_ids.numel())])
    max_trip_num = max(len(row) for row in trip_rows)
    trip_ids = torch.full((len(trip_rows), max_trip_num),
                          pad_id,
                          dtype=torch.long)
    trip_mask = torch.zeros(len(trip_rows), max_trip_num, dtype=torch.bool)
    node_mask = torch.zeros(len(trip_rows), max_trip_num, dtype=torch.bool)
    edge_mask = torch.zeros(len(trip_rows), max_trip_num, dtype=torch.bool)
    for idx, row in enumerate(trip_rows):
        row_len = len(row)
        node_len, edge_len = trip_num[idx]
        if row_len > 0:
            trip_ids[idx, :row_len] = torch.tensor(row, dtype=torch.long)
            trip_mask[idx, :row_len] = True
        if node_len > 0:
            node_mask[idx, :node_len] = True
        if edge_len > 0:
            edge_mask[idx, node_len:node_len + edge_len] = True
    return {
        'trip_ids': trip_ids,
        'trip_num': trip_num,
        'trip_mask': trip_mask,
        'node_mask': node_mask,
        'edge_mask': edge_mask,
    }


def _derive_token_entity_ids(align_mask):
    if align_mask is None:
        return None
    batch_size, seq_len, _ = align_mask.size()
    max_refs = 0
    token_refs: List[List[List[int]]] = []
    for batch_idx in range(batch_size):
        batch_refs = []
        for token_idx in range(seq_len):
            refs = torch.nonzero(align_mask[batch_idx, token_idx],
                                 as_tuple=False).view(-1).tolist()
            batch_refs.append(refs)
            max_refs = max(max_refs, len(refs))
        token_refs.append(batch_refs)
    if max_refs == 0:
        return None
    token_entity_ids = torch.full((batch_size, seq_len, max_refs),
                                  -1,
                                  dtype=torch.long)
    for batch_idx in range(batch_size):
        for token_idx in range(seq_len):
            refs = token_refs[batch_idx][token_idx]
            if refs:
                token_entity_ids[batch_idx, token_idx, :len(refs)] = \
                    torch.tensor(refs, dtype=torch.long)
    return token_entity_ids


def _normalize_single_graph(instance,
                            seq_len,
                            pad_id,
                            kg_cfg=None):
    sample_graph = _merge_sample_graph(instance)
    if len(sample_graph) == 0:
        sample_graph = _empty_graph(pad_id)

    node_ids = _to_tensor(sample_graph.get('x', None), dtype=torch.long)
    if node_ids is None:
        node_ids = _to_tensor(sample_graph.get('node_ids', None),
                              dtype=torch.long)
    if node_ids is None:
        node_ids = _to_tensor(sample_graph.get('entity_ids', None),
                              dtype=torch.long)
    if node_ids is None:
        sample_graph = _empty_graph(pad_id)
        node_ids = sample_graph['x']
    node_ids = node_ids.view(-1)

    edge_index = _normalize_edge_index(sample_graph.get('edge_index', None))
    if edge_index is None:
        sample_graph = _empty_graph(pad_id)
        node_ids = sample_graph['x']
        edge_index = sample_graph['edge_index']

    edge_type = _to_tensor(sample_graph.get('edge_type', None),
                           dtype=torch.long)
    if edge_type is None:
        edge_type = _to_tensor(sample_graph.get('edge_ids', None),
                               dtype=torch.long)
    if edge_type is None:
        edge_type = _to_tensor(sample_graph.get('relation_ids', None),
                               dtype=torch.long)
    if edge_type is None:
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
    edge_type = edge_type.view(-1)
    if edge_type.numel() > edge_index.size(1):
        edge_type = edge_type[:edge_index.size(1)]
    if edge_type.numel() < edge_index.size(1):
        pad = torch.zeros(edge_index.size(1) - edge_type.numel(),
                          dtype=torch.long)
        edge_type = torch.cat([edge_type, pad], dim=0)

    node_type = _to_tensor(sample_graph.get('node_type', None),
                           dtype=torch.long)
    if node_type is None:
        node_type = torch.zeros(node_ids.numel(), dtype=torch.long)
    node_type = node_type.view(-1)
    if node_type.numel() > node_ids.numel():
        node_type = node_type[:node_ids.numel()]
    if node_type.numel() < node_ids.numel():
        pad = torch.zeros(node_ids.numel() - node_type.numel(),
                          dtype=torch.long)
        node_type = torch.cat([node_type, pad], dim=0)

    sample_graph = {
        'x': node_ids,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'node_type': node_type,
        'nid2swid': _normalize_row_index(sample_graph.get('nid2swid', None),
                                         expected_count=node_ids.numel()),
        'eid2swid': _normalize_row_index(sample_graph.get('eid2swid', None),
                                         expected_count=edge_index.size(1)),
        'token_entity_ids': _normalize_token_entity_ids(
            sample_graph.get('token_entity_ids', None), seq_len),
        'align_mask': _normalize_align_mask(
            sample_graph.get('align_mask', sample_graph.get('n2w', None)),
            seq_len,
            node_ids.numel()),
    }

    max_node_num = int(_cfg_get(kg_cfg, 'max_node_num_per_batch', 2500) or 0)
    sample_graph = _clip_graph(sample_graph, max_node_num)

    if int(_cfg_get(kg_cfg, 'num_relations', 1) or 1) == 1:
        sample_graph['edge_type'] = torch.zeros_like(sample_graph['edge_type'])

    sample_graph['num_nodes'] = int(sample_graph['x'].numel())
    sample_graph['num_edges'] = int(sample_graph['edge_index'].size(1))
    return sample_graph


def build_kg_batch(instances, input_ids, pad_id, kg_cfg=None):
    if not any(('sg' in instance or 'kg_inputs' in instance)
               for instance in instances):
        return None

    seq_len = int(input_ids.size(1))
    sample_graphs = [
        _normalize_single_graph(instance, seq_len, pad_id, kg_cfg)
        for instance in instances
    ]

    batch_size = len(sample_graphs)
    max_nodes = max(graph['num_nodes'] for graph in sample_graphs)
    max_edges = max(graph['num_edges'] for graph in sample_graphs)
    max_node_swid = max(
        max((len(row) for row in (graph.get('nid2swid') or [[]])), default=0)
        for graph in sample_graphs)
    max_edge_swid = max(
        max((len(row) for row in (graph.get('eid2swid') or [[]])), default=0)
        for graph in sample_graphs)

    node_ids = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    node_type = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    edge_index = torch.full((batch_size, 2, max_edges),
                            -1,
                            dtype=torch.long)
    edge_mask = torch.zeros(batch_size, max_edges, dtype=torch.bool)
    edge_type = torch.zeros(batch_size, max_edges, dtype=torch.long)

    nid2swid = None
    if max_node_swid > 0:
        nid2swid = torch.full((batch_size, max_nodes, max_node_swid),
                              pad_id,
                              dtype=torch.long)

    eid2swid = None
    if max_edge_swid > 0:
        eid2swid = torch.full((batch_size, max_edges, max_edge_swid),
                              pad_id,
                              dtype=torch.long)

    align_mask = None
    if any(graph.get('align_mask', None) is not None for graph in sample_graphs):
        align_mask = torch.zeros(batch_size,
                                 seq_len,
                                 max_nodes,
                                 dtype=torch.bool)

    token_entity_ids = None
    max_entity_refs = max(
        (graph['token_entity_ids'].size(1)
         if graph.get('token_entity_ids', None) is not None else 0)
        for graph in sample_graphs)
    if max_entity_refs > 0:
        token_entity_ids = torch.full((batch_size, seq_len, max_entity_refs),
                                      -1,
                                      dtype=torch.long)

    flat_nodes = []
    flat_node_types = []
    flat_edges = []
    flat_edge_types = []
    ptr = [0]
    num_edges = []

    for batch_idx, graph in enumerate(sample_graphs):
        current_nodes = graph['x']
        current_node_type = graph['node_type']
        current_edge_index = graph['edge_index']
        current_edge_type = graph['edge_type']
        node_count = graph['num_nodes']
        edge_count = graph['num_edges']

        if node_count > 0:
            node_ids[batch_idx, :node_count] = current_nodes
            node_mask[batch_idx, :node_count] = True
            node_type[batch_idx, :node_count] = current_node_type

        if edge_count > 0:
            edge_index[batch_idx, :, :edge_count] = current_edge_index
            edge_mask[batch_idx, :edge_count] = True
            edge_type[batch_idx, :edge_count] = current_edge_type

        if nid2swid is not None and graph.get('nid2swid', None) is not None:
            for node_idx, swids in enumerate(graph['nid2swid'][:node_count]):
                if not swids:
                    continue
                row = torch.tensor(swids[:max_node_swid], dtype=torch.long)
                nid2swid[batch_idx, node_idx, :row.numel()] = row

        if eid2swid is not None and graph.get('eid2swid', None) is not None:
            for edge_idx, swids in enumerate(graph['eid2swid'][:edge_count]):
                if not swids:
                    continue
                row = torch.tensor(swids[:max_edge_swid], dtype=torch.long)
                eid2swid[batch_idx, edge_idx, :row.numel()] = row

        if align_mask is not None and graph.get('align_mask', None) is not None:
            align_mask[batch_idx, :, :node_count] = graph['align_mask'][:, :node_count]

        if token_entity_ids is not None and \
                graph.get('token_entity_ids', None) is not None:
            current_token_entity_ids = graph['token_entity_ids']
            token_entity_ids[
                batch_idx, :, :current_token_entity_ids.size(1)
            ] = current_token_entity_ids

        flat_nodes.append(current_nodes)
        flat_node_types.append(current_node_type)
        if edge_count > 0:
            flat_edges.append(current_edge_index + ptr[-1])
            flat_edge_types.append(current_edge_type)
        ptr.append(ptr[-1] + node_count)
        num_edges.append(edge_count)

    if token_entity_ids is None and align_mask is not None:
        token_entity_ids = _derive_token_entity_ids(align_mask)

    batch = {
        'x': torch.cat(flat_nodes, dim=0) if flat_nodes else torch.zeros(
            0, dtype=torch.long),
        'ptr': torch.tensor(ptr, dtype=torch.long),
        'batch': torch.cat([
            torch.full((graph['num_nodes'],), idx, dtype=torch.long)
            for idx, graph in enumerate(sample_graphs)
        ],
                           dim=0) if sample_graphs else torch.zeros(0,
                                                                   dtype=torch.long),
        'num_edges': torch.tensor(num_edges, dtype=torch.long),
        'node_ids': node_ids,
        'entity_ids': node_ids,
        'node_mask': node_mask,
        'node_type': node_type,
        'edge_index': edge_index,
        'edge_mask': edge_mask,
        'edge_type': edge_type,
        'edge_ids': edge_type,
        'num_nodes': node_mask.sum(dim=-1),
        'max_node_num': int(max_nodes),
        'prune_mask': torch.ones(sum(graph['num_nodes']
                                     for graph in sample_graphs)),
    }

    if flat_edges:
        batch['flat_edge_index'] = torch.cat(flat_edges, dim=1)
        batch['flat_edge_type'] = torch.cat(flat_edge_types, dim=0)
    else:
        batch['flat_edge_index'] = torch.zeros((2, 0), dtype=torch.long)
        batch['flat_edge_type'] = torch.zeros(0, dtype=torch.long)

    if nid2swid is not None:
        batch['nid2swid'] = nid2swid
    if eid2swid is not None:
        batch['eid2swid'] = eid2swid
    if align_mask is not None:
        batch['align_mask'] = align_mask
    if token_entity_ids is not None:
        batch['token_entity_ids'] = token_entity_ids

    if _cfg_get(kg_cfg, 'use_trips', True):
        trip_inputs = _build_trip_tensor(
            [graph['x'] for graph in sample_graphs],
            [graph['edge_type'] for graph in sample_graphs],
            pad_id=pad_id)
        if trip_inputs is not None:
            batch['trips'] = trip_inputs

    return batch
