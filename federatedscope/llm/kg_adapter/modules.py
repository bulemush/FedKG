import copy
import importlib.util
import os
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn


def _to_attr_dict(data):
    if data is None:
        return {}
    if isinstance(data, dict):
        return dict(data)

    result = {}
    for key in dir(data):
        if key.startswith('_'):
            continue
        try:
            value = getattr(data, key)
        except AttributeError:
            continue
        if callable(value):
            continue
        result[key] = value
    return result


def _resolve_cfg_value(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_module_dtype_device(module):
    for param in module.parameters():
        return param.dtype, param.device
    for buffer in module.buffers():
        return buffer.dtype, buffer.device
    return None, None


def _cast_module_like(module, reference_module):
    dtype, device = _get_module_dtype_device(reference_module)
    if dtype is None and device is None:
        return module
    kwargs = {}
    if device is not None:
        kwargs['device'] = device
    if dtype is not None and (dtype.is_floating_point or dtype.is_complex):
        kwargs['dtype'] = dtype
    if kwargs:
        module.to(**kwargs)
    return module


def _get_module_compute_dtype(module, default=None):
    dtype, _ = _get_module_dtype_device(module)
    if dtype is None:
        return default
    return dtype


def _cast_tensor_for_module(tensor, module, default_dtype=None):
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    target_dtype = _get_module_compute_dtype(module, default=default_dtype)
    if target_dtype is None:
        return tensor
    if tensor.dtype == target_dtype:
        return tensor
    if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
        return tensor.to(dtype=target_dtype)
    return tensor


def _clone_cfg(cfg):
    if cfg is None:
        return {}
    try:
        return copy.deepcopy(cfg)
    except Exception:
        return cfg


def _as_tensor(value, dtype=None, device=None):
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.tensor(value, dtype=dtype)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _batchify_from_ptr(flat_tensor, ptr, max_count=None, pad_value=0):
    if flat_tensor is None or ptr is None:
        return None
    ptr = _as_tensor(ptr, dtype=torch.long, device=flat_tensor.device)
    batch_size = ptr.numel() - 1
    if batch_size <= 0:
        return None
    if max_count is None:
        max_count = int((ptr[1:] - ptr[:-1]).max().item())
    batch_shape = [batch_size, max_count] + list(flat_tensor.shape[1:])
    output = flat_tensor.new_full(batch_shape, pad_value)
    mask = torch.zeros(batch_size,
                       max_count,
                       dtype=torch.bool,
                       device=flat_tensor.device)
    for batch_idx in range(batch_size):
        left = int(ptr[batch_idx].item())
        right = int(ptr[batch_idx + 1].item())
        count = max(right - left, 0)
        if count == 0:
            continue
        output[batch_idx, :count] = flat_tensor[left:right]
        mask[batch_idx, :count] = True
    return output, mask


def _batchify_edges_from_counts(flat_tensor,
                                counts,
                                max_count=None,
                                pad_value=0):
    if flat_tensor is None or counts is None:
        return None
    counts = _as_tensor(counts, dtype=torch.long, device=flat_tensor.device)
    if counts.numel() == 0:
        return None
    batch_size = counts.numel()
    if max_count is None:
        max_count = int(counts.max().item()) if batch_size > 0 else 0
    batch_shape = [batch_size, max_count] + list(flat_tensor.shape[1:])
    output = flat_tensor.new_full(batch_shape, pad_value)
    mask = torch.zeros(batch_size,
                       max_count,
                       dtype=torch.bool,
                       device=flat_tensor.device)
    cursor = 0
    for batch_idx in range(batch_size):
        count = int(counts[batch_idx].item())
        if count == 0:
            continue
        output[batch_idx, :count] = flat_tensor[cursor:cursor + count]
        mask[batch_idx, :count] = True
        cursor += count
    return output, mask


def _normalize_edge_index(edge_index, batch_size, device, counts=None, ptr=None):
    if edge_index is None:
        return None, None
    edge_index = _as_tensor(edge_index, dtype=torch.long, device=device)
    if edge_index.dim() == 3 and edge_index.size(1) == 2:
        mask = edge_index.ge(0).all(dim=1)
        return edge_index, mask
    if edge_index.dim() == 3 and edge_index.size(-1) == 2:
        edge_index = edge_index.transpose(1, 2)
        mask = edge_index.ge(0).all(dim=1)
        return edge_index, mask
    if edge_index.dim() == 2 and edge_index.size(0) == 2 and batch_size == 1:
        edge_index = edge_index.unsqueeze(0)
        mask = edge_index.ge(0).all(dim=1)
        return edge_index, mask
    if edge_index.dim() == 2 and edge_index.size(0) == 2 and \
            counts is not None and ptr is not None:
        counts = _as_tensor(counts, dtype=torch.long, device=device)
        ptr = _as_tensor(ptr, dtype=torch.long, device=device)
        max_edges = int(counts.max().item()) if counts.numel() > 0 else 0
        output = edge_index.new_full((batch_size, 2, max_edges), -1)
        mask = torch.zeros(batch_size,
                           max_edges,
                           dtype=torch.bool,
                           device=device)
        cursor = 0
        for batch_idx in range(batch_size):
            edge_count = int(counts[batch_idx].item())
            if edge_count == 0:
                continue
            node_offset = int(ptr[batch_idx].item())
            local_edges = edge_index[:, cursor:cursor + edge_count] - node_offset
            output[batch_idx, :, :edge_count] = local_edges
            mask[batch_idx, :edge_count] = True
            cursor += edge_count
        return output, mask
    return None, None


class FSKGAdapterRuntime:
    def __init__(self):
        self.state = None

    def activate(self,
                 kg_inputs: Optional[Any] = None,
                 sg: Optional[Any] = None,
                 input_ids: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None):
        merged_inputs = _to_attr_dict(kg_inputs)
        sg_inputs = _to_attr_dict(sg)
        for key, value in sg_inputs.items():
            merged_inputs.setdefault(key, value)
        self.state = {
            'kg_inputs': merged_inputs if len(merged_inputs) > 0 else None,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'node_states': None,
            'node_mask': None,
            'edge_states': None,
            'edge_mask': None,
            'edge_index': None,
            'trip_states': None,
            'trip_mask': None,
        }

    def clear(self):
        self.state = None

    def get(self):
        return self.state


def maybe_activate_runtime(model,
                           kg_inputs: Optional[Any] = None,
                           sg: Optional[Any] = None,
                           input_ids: Optional[torch.Tensor] = None,
                           attention_mask: Optional[torch.Tensor] = None):
    runtime = getattr(model, 'kg_adapter_runtime', None)
    if runtime is not None:
        runtime.activate(kg_inputs=kg_inputs,
                         sg=sg,
                         input_ids=input_ids,
                         attention_mask=attention_mask)


def maybe_clear_runtime(model):
    runtime = getattr(model, 'kg_adapter_runtime', None)
    if runtime is not None:
        runtime.clear()


_PAPER_GNN_MODULE_CACHE = {}


def _resolve_cfg_path(cfg, key, default=''):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _iter_paper_gnn_candidates(kg_cfg=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    configured_path = _resolve_cfg_path(kg_cfg, 'paper_gnn_path', '')
    candidates = []
    if configured_path not in [None, '']:
        if os.path.isabs(configured_path):
            candidates.append(configured_path)
        else:
            candidates.append(os.path.join(repo_root, configured_path))
    candidates.extend([
        os.path.join(repo_root, 'federatedscope', 'llm', 'kg_adapter',
                     'paper_gnn', 'GNN.py'),
        os.path.join(repo_root, '..', 'KG-Adapter-main', 'model', 'GNN.py'),
    ])
    seen = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def _load_paper_gnn_module(kg_cfg=None):
    global _PAPER_GNN_MODULE_CACHE
    for candidate in _iter_paper_gnn_candidates(kg_cfg):
        if candidate in _PAPER_GNN_MODULE_CACHE:
            module = _PAPER_GNN_MODULE_CACHE[candidate]
            return None if module is False else module
        if not os.path.exists(candidate):
            _PAPER_GNN_MODULE_CACHE[candidate] = False
            continue

        spec = importlib.util.spec_from_file_location(
            'fedbiot_paper_gnn_module', candidate)
        if spec is None or spec.loader is None:
            _PAPER_GNN_MODULE_CACHE[candidate] = False
            continue

        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:
            _PAPER_GNN_MODULE_CACHE[candidate] = False
            continue
        _PAPER_GNN_MODULE_CACHE[candidate] = module
        return module
    return None


def _flatten_graph_batch(node_states, node_mask, edge_index, edge_states,
                         edge_mask, kg_inputs):
    if node_states is None or edge_index is None:
        return None

    batch_size, max_nodes, hidden_size = node_states.size()
    flat_nodes = []
    flat_node_types = []
    flat_edges = []
    flat_edge_states = []
    flat_edge_types = []
    batch = []
    node_offset = 0

    node_type = _as_tensor(kg_inputs.get('node_type', None),
                           dtype=torch.long,
                           device=node_states.device)
    edge_type = _as_tensor(kg_inputs.get('edge_type', None),
                           dtype=torch.long,
                           device=node_states.device)
    if edge_type is not None and edge_type.dim() == 2:
        edge_type_dense = edge_type
    elif edge_type is not None and edge_type.dim() == 1 and \
            kg_inputs.get('num_edges', None) is not None:
        batched = _batchify_edges_from_counts(edge_type,
                                              kg_inputs.get('num_edges'))
        edge_type_dense = batched[0] if batched is not None else None
    else:
        edge_type_dense = edge_type

    for batch_idx in range(batch_size):
        valid_nodes = node_mask[batch_idx] if node_mask is not None else \
            torch.ones(max_nodes,
                       dtype=torch.bool,
                       device=node_states.device)
        node_count = int(valid_nodes.sum().item())
        if node_count == 0:
            continue

        current_nodes = node_states[batch_idx][valid_nodes]
        flat_nodes.append(current_nodes)
        batch.append(torch.full((node_count,),
                                batch_idx,
                                dtype=torch.long,
                                device=node_states.device))

        if node_type is not None:
            if node_type.dim() == 2:
                flat_node_types.append(node_type[batch_idx][valid_nodes])
            elif node_type.dim() == 1 and kg_inputs.get('ptr', None) is not None:
                ptr = _as_tensor(kg_inputs.get('ptr'),
                                 dtype=torch.long,
                                 device=node_states.device)
                flat_node_types.append(node_type[ptr[batch_idx]:ptr[
                    batch_idx + 1]])
            else:
                flat_node_types.append(
                    torch.zeros(node_count,
                                dtype=torch.long,
                                device=node_states.device))

        valid_edges = edge_mask[batch_idx] if edge_mask is not None else \
            torch.ones(edge_index.size(-1),
                       dtype=torch.bool,
                       device=node_states.device)
        current_edge_index = edge_index[batch_idx][:, valid_edges]
        if current_edge_index.numel() > 0:
            valid = (current_edge_index[0] >= 0) & (current_edge_index[1] >= 0)
            if node_mask is not None:
                valid = valid & valid_nodes[current_edge_index[0]] & \
                    valid_nodes[current_edge_index[1]]
            current_edge_index = current_edge_index[:, valid]
            if current_edge_index.numel() > 0:
                local_to_global = torch.full((max_nodes,),
                                             -1,
                                             dtype=torch.long,
                                             device=node_states.device)
                local_to_global[valid_nodes] = torch.arange(
                    node_count, device=node_states.device) + node_offset
                current_edge_index = local_to_global[current_edge_index]
                flat_edges.append(current_edge_index)

                if edge_states is not None:
                    current_edge_states = edge_states[batch_idx][valid_edges]
                    current_edge_states = current_edge_states[valid]
                    flat_edge_states.append(current_edge_states)
                if edge_type_dense is not None:
                    current_edge_type = edge_type_dense[batch_idx][valid_edges]
                    current_edge_type = current_edge_type[valid]
                    flat_edge_types.append(current_edge_type)

        node_offset += node_count

    if len(flat_nodes) == 0:
        return None

    result = {
        'x': torch.cat(flat_nodes, dim=0),
        'batch': torch.cat(batch, dim=0),
    }
    if len(flat_edges) > 0:
        result['edge_index'] = torch.cat(flat_edges, dim=1)
    else:
        result['edge_index'] = torch.empty(2,
                                           0,
                                           dtype=torch.long,
                                           device=node_states.device)
    if len(flat_edge_states) > 0:
        result['edge_attr'] = torch.cat(flat_edge_states, dim=0)
    else:
        result['edge_attr'] = None
    if len(flat_edge_types) > 0:
        result['edge_type'] = torch.cat(flat_edge_types, dim=0)
    else:
        result['edge_type'] = torch.zeros(result['edge_index'].size(1),
                                          dtype=torch.long,
                                          device=node_states.device)
    if len(flat_node_types) > 0:
        result['node_type'] = torch.cat(flat_node_types, dim=0)
    else:
        result['node_type'] = torch.zeros(result['x'].size(0),
                                          dtype=torch.long,
                                          device=node_states.device)
    return result


def _dense_from_flat(flat_states, batch_vector, node_mask, hidden_size):
    if flat_states is None or batch_vector is None or node_mask is None:
        return flat_states
    batch_size, max_nodes = node_mask.size()
    dense = flat_states.new_zeros(batch_size, max_nodes, hidden_size)
    counters = torch.zeros(batch_size,
                           dtype=torch.long,
                           device=flat_states.device)
    for idx in range(flat_states.size(0)):
        batch_idx = int(batch_vector[idx].item())
        pos = int(counters[batch_idx].item())
        if pos < max_nodes:
            dense[batch_idx, pos] = flat_states[idx]
        counters[batch_idx] += 1
    return dense


class KGAdapterInputModule(nn.Module):
    def __init__(self, hidden_size, entity_hidden_size, num_heads, dropout):
        super().__init__()
        self.entity_proj = nn.Linear(entity_hidden_size, hidden_size)
        self.text_norm = nn.LayerNorm(hidden_size)
        self.entity_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ffn_gate = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        self._is_fs_kg_trainable_module = True

    def forward(self, hidden_states, node_states, node_mask=None):
        if node_states is None:
            return hidden_states
        output_dtype = hidden_states.dtype
        hidden_states = _cast_tensor_for_module(hidden_states, self.entity_proj,
                                                output_dtype)
        node_states = _cast_tensor_for_module(node_states, self.entity_proj,
                                              hidden_states.dtype)
        node_states = self.entity_proj(node_states)
        key_padding_mask = None
        if node_mask is not None:
            key_padding_mask = ~node_mask.bool()
        attn_output, _ = self.cross_attn(
            self.text_norm(hidden_states),
            self.entity_norm(node_states),
            self.entity_norm(node_states),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + torch.sigmoid(self.attn_gate) * \
            self.dropout(attn_output)
        hidden_states = hidden_states + torch.sigmoid(self.ffn_gate) * \
            self.dropout(self.ffn(self.ffn_norm(hidden_states)))
        return hidden_states.to(output_dtype)


class KGGraphMessagePassing(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.node_proj = nn.Linear(hidden_size, hidden_size)
        self.edge_proj = nn.Linear(hidden_size, hidden_size)
        self.update_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.update_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self._is_fs_kg_trainable_module = True

    def forward(self, node_states, edge_index, edge_states=None, node_mask=None,
                edge_mask=None):
        if node_states is None or edge_index is None:
            return node_states
        output_dtype = node_states.dtype
        node_states = _cast_tensor_for_module(node_states, self.node_proj,
                                              output_dtype)
        if edge_states is not None:
            edge_states = _cast_tensor_for_module(edge_states, self.edge_proj,
                                                  node_states.dtype)
        batch_size, node_count, hidden_size = node_states.size()
        output = node_states
        for batch_idx in range(batch_size):
            batch_nodes = output[batch_idx]
            batch_edges = edge_index[batch_idx]
            valid_edge_mask = edge_mask[batch_idx] if edge_mask is not None \
                else torch.ones(batch_edges.size(-1),
                                dtype=torch.bool,
                                device=batch_edges.device)
            batch_edges = batch_edges[:, valid_edge_mask]
            if batch_edges.numel() == 0:
                continue

            src = batch_edges[0]
            dst = batch_edges[1]
            valid_nodes = (src >= 0) & (dst >= 0) & \
                (src < node_count) & (dst < node_count)
            if node_mask is not None:
                batch_node_mask = node_mask[batch_idx]
                valid_nodes = valid_nodes & batch_node_mask[src] & \
                    batch_node_mask[dst]
            src = src[valid_nodes]
            dst = dst[valid_nodes]
            if src.numel() == 0:
                continue

            messages = self.node_proj(batch_nodes.index_select(0, src))
            if edge_states is not None:
                batch_edge_states = edge_states[batch_idx][valid_edge_mask]
                batch_edge_states = batch_edge_states[valid_nodes]
                messages = messages + self.edge_proj(batch_edge_states)

            aggregated = batch_nodes.new_zeros(node_count, hidden_size)
            counts = batch_nodes.new_zeros(node_count, 1)
            aggregated.index_add_(0, dst, messages)
            counts.index_add_(0, dst, torch.ones(dst.size(0),
                                                 1,
                                                 device=batch_nodes.device,
                                                 dtype=batch_nodes.dtype))
            aggregated = aggregated / counts.clamp_min(1.0)
            fused_input = torch.cat([batch_nodes, aggregated], dim=-1)
            update = torch.tanh(self.update_proj(fused_input))
            gate = torch.sigmoid(self.update_gate(fused_input))
            updated_nodes = self.norm(batch_nodes + gate * self.dropout(update))
            if node_mask is not None:
                updated_nodes = torch.where(node_mask[batch_idx].unsqueeze(-1),
                                            updated_nodes,
                                            batch_nodes)
            output[batch_idx] = updated_nodes
        return output.to(output_dtype)


class KGTripEncoderMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self._is_fs_kg_trainable_module = True

    def forward(self, node_states, edge_states, edge_index, node_mask=None,
                edge_mask=None):
        if node_states is None or edge_states is None or edge_index is None:
            return None, None
        output_dtype = node_states.dtype
        node_states = _cast_tensor_for_module(node_states, self.mlp,
                                              output_dtype)
        edge_states = _cast_tensor_for_module(edge_states, self.mlp,
                                              node_states.dtype)
        batch_size, _, hidden_size = node_states.size()
        trip_reps = []
        max_trip_num = 0
        for batch_idx in range(batch_size):
            batch_edges = edge_index[batch_idx]
            valid_edge_mask = edge_mask[batch_idx] if edge_mask is not None \
                else torch.ones(batch_edges.size(-1),
                                dtype=torch.bool,
                                device=batch_edges.device)
            batch_edges = batch_edges[:, valid_edge_mask]
            batch_edge_states = edge_states[batch_idx][valid_edge_mask]
            if batch_edges.numel() == 0:
                trip_reps.append(node_states.new_zeros((0, hidden_size)))
                continue
            src = batch_edges[0]
            dst = batch_edges[1]
            valid = (src >= 0) & (dst >= 0) & \
                (src < node_states.size(1)) & (dst < node_states.size(1))
            if node_mask is not None:
                valid = valid & node_mask[batch_idx][src] & \
                    node_mask[batch_idx][dst]
            src = src[valid]
            dst = dst[valid]
            batch_edge_states = batch_edge_states[valid]
            if src.numel() == 0:
                trip_reps.append(node_states.new_zeros((0, hidden_size)))
                continue
            head_states = node_states[batch_idx].index_select(0, src)
            tail_states = node_states[batch_idx].index_select(0, dst)
            trip_input = torch.cat([head_states, batch_edge_states, tail_states],
                                   dim=-1)
            trip_state = self.mlp(_cast_tensor_for_module(trip_input,
                                                          self.mlp,
                                                          node_states.dtype))
            trip_reps.append(trip_state)
            max_trip_num = max(max_trip_num, trip_state.size(0))

        if max_trip_num == 0:
            return None, None

        padded_trips = node_states.new_zeros(batch_size,
                                             max_trip_num,
                                             hidden_size)
        padded_masks = torch.zeros(batch_size,
                                   max_trip_num,
                                   dtype=torch.bool,
                                   device=node_states.device)
        for batch_idx, trip_state in enumerate(trip_reps):
            if trip_state.size(0) == 0:
                continue
            padded_trips[batch_idx, :trip_state.size(0)] = trip_state
            padded_masks[batch_idx, :trip_state.size(0)] = True
        return padded_trips.to(output_dtype), padded_masks


class KGCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self._is_fs_kg_trainable_module = True

    def forward(self, query_states, context_states, context_mask=None):
        if query_states is None or context_states is None:
            return query_states
        output_dtype = query_states.dtype
        query_states = _cast_tensor_for_module(query_states, self.cross_attn,
                                               output_dtype)
        context_states = _cast_tensor_for_module(context_states, self.cross_attn,
                                                 query_states.dtype)
        key_padding_mask = None
        if context_mask is not None:
            key_padding_mask = ~context_mask.bool()
        attn_output, _ = self.cross_attn(self.query_norm(query_states),
                                         self.context_norm(context_states),
                                         self.context_norm(context_states),
                                         key_padding_mask=key_padding_mask,
                                         need_weights=False)
        return self.dropout(attn_output).to(output_dtype)


class KGReasoningFFN(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)
        self._is_fs_kg_trainable_module = True

    def forward(self, hidden_states):
        output_dtype = hidden_states.dtype
        hidden_states = _cast_tensor_for_module(hidden_states, self.ffn,
                                                output_dtype)
        return (hidden_states +
                self.dropout(self.ffn(self.norm(hidden_states)))).to(output_dtype)


class KGJointReasoningModule(nn.Module):
    def __init__(self, model_hidden_size, kg_hidden_size, num_heads, dropout):
        super().__init__()
        self.text_down = nn.Linear(model_hidden_size, kg_hidden_size)
        self.text_up = nn.Linear(kg_hidden_size, model_hidden_size)
        self.node_from_text = KGCrossAttention(kg_hidden_size,
                                               num_heads,
                                               dropout)
        self.text_from_node = KGCrossAttention(kg_hidden_size,
                                               num_heads,
                                               dropout)
        self.node_ffn = KGReasoningFFN(kg_hidden_size, dropout)
        self.text_ffn = KGReasoningFFN(kg_hidden_size, dropout)
        self.text_gate = nn.Parameter(torch.zeros(1))
        self.node_gate = nn.Parameter(torch.zeros(1))
        self.output_gate = nn.Parameter(torch.zeros(1))
        self._is_fs_kg_trainable_module = True

    def forward(self, hidden_states, context_states, context_mask=None):
        if hidden_states is None or context_states is None:
            return hidden_states, context_states
        output_dtype = hidden_states.dtype
        context_output_dtype = context_states.dtype
        hidden_states = _cast_tensor_for_module(hidden_states, self.text_down,
                                                output_dtype)
        context_states = _cast_tensor_for_module(context_states, self.text_down,
                                                 hidden_states.dtype)
        text_states = self.text_down(hidden_states)
        updated_context = context_states + torch.sigmoid(self.node_gate) * \
            self.node_from_text(context_states, text_states)
        updated_context = self.node_ffn(updated_context)

        updated_text = text_states + torch.sigmoid(self.text_gate) * \
            self.text_from_node(text_states, context_states, context_mask)
        updated_text = self.text_ffn(updated_text)

        fused_hidden = hidden_states + torch.sigmoid(self.output_gate) * \
            self.text_up(updated_text)
        return fused_hidden.to(output_dtype), updated_context.to(
            context_output_dtype)


class PaperRGATWrapper(nn.Module):
    def __init__(self, hidden_size, num_relations, use_edge_emb, kg_cfg=None):
        super().__init__()
        paper_gnn = _load_paper_gnn_module(kg_cfg)
        if paper_gnn is None:
            raise ImportError('Paper GNN backend is unavailable.')
        self.conv = paper_gnn.RGATConv(
            hidden_size,
            hidden_size,
            num_relations,
            attention_mode="additive-self-attention",
            heads=1,
            dim=1,
            concat=False,
            edge_dim=hidden_size if use_edge_emb else None,
        )
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self._is_fs_kg_trainable_module = True

    def forward(self, node_states, edge_index, edge_states, node_mask, edge_mask,
                kg_inputs):
        flat_graph = _flatten_graph_batch(node_states, node_mask, edge_index,
                                          edge_states, edge_mask, kg_inputs)
        if flat_graph is None:
            return node_states
        output_dtype = node_states.dtype
        flat_graph['x'] = _cast_tensor_for_module(flat_graph['x'], self.conv,
                                                  output_dtype)
        if flat_graph['edge_attr'] is not None:
            flat_graph['edge_attr'] = _cast_tensor_for_module(
                flat_graph['edge_attr'], self.conv, flat_graph['x'].dtype)
        out, _ = self.conv(flat_graph['x'],
                           flat_graph['edge_index'],
                           flat_graph['edge_type'],
                           edge_attr=flat_graph['edge_attr'],
                           return_attention_weights=True)
        out = self.output_proj(out)
        return _dense_from_flat(out, flat_graph['batch'], node_mask,
                                out.size(-1)).to(output_dtype)


class PaperSRGATWrapper(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_relations,
                 use_edge_emb,
                 keep_ratio,
                 kg_cfg=None):
        super().__init__()
        paper_gnn = _load_paper_gnn_module(kg_cfg)
        if paper_gnn is None:
            raise ImportError('Paper GNN backend is unavailable.')
        args = SimpleNamespace(dev=False)
        self.conv = paper_gnn.SRGATConv(args=args,
                                        emb_dim=hidden_size,
                                        n_ntype=4,
                                        n_etype=num_relations,
                                        head_count=4)
        self.output_act = nn.GELU()
        self.pool = None
        if keep_ratio < 1:
            self.pool = paper_gnn.SAGPooling(hidden_size,
                                             ratio=keep_ratio,
                                             nonlinearity=torch.tanh)
        self.use_edge_emb = use_edge_emb
        self._is_fs_kg_trainable_module = True

    def forward(self, node_states, edge_index, edge_states, node_mask, edge_mask,
                kg_inputs):
        flat_graph = _flatten_graph_batch(node_states, node_mask, edge_index,
                                          edge_states, edge_mask, kg_inputs)
        if flat_graph is None:
            return node_states
        output_dtype = node_states.dtype
        flat_graph['x'] = _cast_tensor_for_module(flat_graph['x'], self.conv,
                                                  output_dtype)
        if flat_graph['edge_attr'] is not None:
            flat_graph['edge_attr'] = _cast_tensor_for_module(
                flat_graph['edge_attr'], self.conv, flat_graph['x'].dtype)
        out, attn_weights = self.conv(flat_graph['x'],
                                      flat_graph['edge_index'],
                                      flat_graph['edge_type'],
                                      node_type=flat_graph['node_type'],
                                      edge_attr=flat_graph['edge_attr'],
                                      return_attention_weights=True)
        out = self.output_act(out)
        if self.pool is not None:
            score = attn_weights[1].sum(dim=1)
            out, _, _, _, _, batch, perm, _ = self.pool(
                x=out,
                score=score,
                edge_index=flat_graph['edge_index'],
                edge_attr=flat_graph['edge_attr'],
                edge_type=flat_graph['edge_type'],
                node_type=flat_graph['node_type'],
                batch=flat_graph['batch'])
            flat_graph['batch'] = batch
        return _dense_from_flat(out, flat_graph['batch'], node_mask,
                                out.size(-1)).to(output_dtype)


class KGHybridEmbedding(nn.Module):
    def __init__(self, base_embedding, runtime, kg_cfg, hidden_size):
        super().__init__()
        self.base_embedding = base_embedding
        self.runtime = runtime
        self.hidden_size = hidden_size

        entity_hidden_size = _resolve_cfg_value(kg_cfg, 'entity_hidden_size',
                                                hidden_size)
        edge_vocab_size = _resolve_cfg_value(
            kg_cfg, 'edge_vocab_size',
            _resolve_cfg_value(kg_cfg, 'entity_vocab_size', 1))
        entity_vocab_size = _resolve_cfg_value(kg_cfg, 'entity_vocab_size', 1)
        num_heads = _resolve_cfg_value(kg_cfg, 'num_heads', 4)
        dropout = _resolve_cfg_value(kg_cfg, 'dropout', 0.0)

        self.entity_embedding = nn.Embedding(entity_vocab_size,
                                             entity_hidden_size)
        self.edge_embedding = nn.Embedding(edge_vocab_size, entity_hidden_size)
        self.subword_entity_proj = nn.Linear(hidden_size, entity_hidden_size)
        self.subword_edge_proj = nn.Linear(hidden_size, entity_hidden_size)
        self.entity_to_hidden = nn.Linear(entity_hidden_size, hidden_size)
        self.hybrid_gate = nn.Parameter(torch.zeros(1))
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_adapter = KGAdapterInputModule(
            hidden_size=hidden_size,
            entity_hidden_size=entity_hidden_size,
            num_heads=num_heads,
            dropout=dropout)
        _cast_module_like(self.entity_embedding, self.base_embedding)
        _cast_module_like(self.edge_embedding, self.base_embedding)
        _cast_module_like(self.subword_entity_proj, self.base_embedding)
        _cast_module_like(self.subword_edge_proj, self.base_embedding)
        _cast_module_like(self.entity_to_hidden, self.base_embedding)
        _cast_module_like(self.input_norm, self.base_embedding)
        _cast_module_like(self.input_adapter, self.base_embedding)
        self._is_fs_kg_trainable_module = True

    def set_fs_kg_trainable(self, is_trainable):
        for param in self.base_embedding.parameters():
            param.requires_grad = False
        for module in [
                self.entity_embedding,
                self.edge_embedding,
                self.subword_entity_proj,
                self.subword_edge_proj,
                self.entity_to_hidden,
                self.input_norm,
                self.input_adapter,
        ]:
            for param in module.parameters():
                param.requires_grad = is_trainable
        self.hybrid_gate.requires_grad = is_trainable

    @property
    def weight(self):
        return self.base_embedding.weight

    @property
    def num_embeddings(self):
        return self.base_embedding.num_embeddings

    @property
    def embedding_dim(self):
        return self.base_embedding.embedding_dim

    def forward(self, input_ids):
        word_embeds = self.base_embedding(input_ids)
        output_dtype = word_embeds.dtype
        runtime_state = self.runtime.get() if self.runtime is not None else None
        if runtime_state is None or runtime_state.get('kg_inputs') is None:
            return word_embeds

        kg_inputs = runtime_state['kg_inputs']
        node_states, node_mask = self._build_node_states(
            kg_inputs, word_embeds.device, word_embeds.dtype)
        edge_states, edge_mask = self._build_edge_states(
            kg_inputs,
            word_embeds.device,
            word_embeds.dtype,
            batch_size=node_states.size(0)
            if node_states is not None else word_embeds.size(0))
        edge_index, inferred_edge_mask = self._build_edge_index(
            kg_inputs,
            word_embeds.device,
            batch_size=node_states.size(0)
            if node_states is not None else word_embeds.size(0),
            node_mask=node_mask)
        if edge_mask is None:
            edge_mask = inferred_edge_mask
        runtime_state['node_states'] = node_states
        runtime_state['node_mask'] = node_mask
        runtime_state['edge_states'] = edge_states
        runtime_state['edge_mask'] = edge_mask
        runtime_state['edge_index'] = edge_index
        runtime_state['trip_states'] = None
        runtime_state['trip_mask'] = None

        token_entity_states = self._build_token_entity_states(
            kg_inputs,
            node_states,
            node_mask,
            word_embeds.device,
            word_embeds.dtype,
            word_embeds.size(1))
        if token_entity_states is not None:
            mixed = torch.sigmoid(self.hybrid_gate) * token_entity_states
            norm_input = word_embeds + mixed
            norm_input = _cast_tensor_for_module(norm_input, self.input_norm,
                                                 output_dtype)
            word_embeds = self.input_norm(norm_input).to(output_dtype)

        adapter_hidden = _cast_tensor_for_module(word_embeds,
                                                 self.input_adapter,
                                                 output_dtype)
        adapter_nodes = node_states
        if adapter_nodes is not None:
            adapter_nodes = _cast_tensor_for_module(adapter_nodes,
                                                    self.input_adapter,
                                                    adapter_hidden.dtype)
        return self.input_adapter(adapter_hidden, adapter_nodes,
                                  node_mask).to(output_dtype)

    def _lookup_node_ids(self, kg_inputs, device):
        node_ids = kg_inputs.get('entity_ids', None)
        if node_ids is None:
            node_ids = kg_inputs.get('node_ids', None)
        if node_ids is None:
            node_ids = kg_inputs.get('x', None)
        if node_ids is None:
            return None
        node_ids = _as_tensor(node_ids, dtype=torch.long, device=device)
        if node_ids.dim() == 1 and 'ptr' in kg_inputs:
            batched = _batchify_from_ptr(node_ids, kg_inputs.get('ptr'))
            if batched is not None:
                return batched[0]
        if node_ids.dim() == 1:
            node_ids = node_ids.unsqueeze(0)
        return node_ids

    def _build_node_states(self, kg_inputs, device, dtype):
        node_ids = self._lookup_node_ids(kg_inputs, device)
        node_states = None
        node_mask = _as_tensor(kg_inputs.get('node_mask', None),
                               dtype=torch.bool,
                               device=device)
        if node_ids is not None:
            if node_mask is None:
                node_mask = node_ids.ne(0)
            node_states = self.entity_embedding(node_ids)

        subword_index = kg_inputs.get('nid2swid', None)
        if subword_index is None:
            subword_index = kg_inputs.get('entity_subword_index', None)
        if subword_index is not None:
            subword_index = _as_tensor(subword_index,
                                       dtype=torch.long,
                                       device=device)
            if subword_index.dim() == 2:
                if node_states is not None and node_states.size(0) == 1:
                    subword_index = subword_index.unsqueeze(0)
                elif 'ptr' in kg_inputs:
                    batched = _batchify_from_ptr(subword_index,
                                                 kg_inputs.get('ptr'))
                    if batched is not None:
                        subword_index = batched[0]
                        if node_mask is None:
                            node_mask = batched[1]
                else:
                    subword_index = subword_index.unsqueeze(0)
            subword_embeds = self.base_embedding(subword_index)
            subword_mask = subword_index.ne(0).unsqueeze(-1).to(
                subword_embeds.dtype)
            subword_sum = (subword_embeds * subword_mask).sum(dim=-2)
            subword_den = subword_mask.sum(dim=-2).clamp_min(1.0)
            subword_states = self.subword_entity_proj(
                _cast_tensor_for_module(subword_sum / subword_den,
                                        self.subword_entity_proj,
                                        subword_embeds.dtype))
            if node_states is None:
                node_states = subword_states
                node_mask = subword_index.ne(0).any(dim=-1)
            else:
                gate = torch.sigmoid(self.hybrid_gate)
                node_states = gate * node_states + (1 - gate) * \
                    subword_states.to(node_states.dtype)
                node_mask = node_mask | subword_index.ne(0).any(dim=-1)

        if node_states is None:
            return None, None

        node_states = node_states.to(dtype)
        if node_mask is None:
            node_mask = torch.ones(node_states.size()[:2],
                                   dtype=torch.bool,
                                   device=device)
        return node_states, node_mask

    def _lookup_edge_ids(self, kg_inputs, device):
        edge_ids = kg_inputs.get('edge_ids', None)
        if edge_ids is None:
            edge_ids = kg_inputs.get('relation_ids', None)
        if edge_ids is None:
            edge_ids = kg_inputs.get('edge_type', None)
        if edge_ids is None:
            return None
        edge_ids = _as_tensor(edge_ids, dtype=torch.long, device=device)
        if edge_ids.dim() == 1 and kg_inputs.get('num_edges', None) is not None:
            batched = _batchify_edges_from_counts(edge_ids,
                                                  kg_inputs.get('num_edges'))
            if batched is not None:
                return batched[0]
        if edge_ids.dim() == 1:
            edge_ids = edge_ids.unsqueeze(0)
        return edge_ids

    def _build_edge_states(self, kg_inputs, device, dtype, batch_size):
        edge_ids = self._lookup_edge_ids(kg_inputs, device)
        edge_states = None
        edge_mask = _as_tensor(kg_inputs.get('edge_mask', None),
                               dtype=torch.bool,
                               device=device)
        if edge_ids is not None:
            if edge_mask is None:
                edge_mask = edge_ids.ge(0)
            edge_ids = edge_ids.clamp_min(0)
            edge_states = self.edge_embedding(edge_ids)

        subword_index = kg_inputs.get('eid2swid', None)
        if subword_index is None:
            subword_index = kg_inputs.get('edge_subword_index', None)
        if subword_index is not None:
            subword_index = _as_tensor(subword_index,
                                       dtype=torch.long,
                                       device=device)
            if subword_index.dim() == 2 and batch_size > 1 and \
                    kg_inputs.get('num_edges', None) is not None:
                batched = _batchify_edges_from_counts(subword_index,
                                                      kg_inputs.get('num_edges'))
                if batched is not None:
                    subword_index = batched[0]
                    if edge_mask is None:
                        edge_mask = batched[1]
            elif subword_index.dim() == 2:
                subword_index = subword_index.unsqueeze(0)
            subword_embeds = self.base_embedding(subword_index)
            subword_mask = subword_index.ne(0).unsqueeze(-1).to(
                subword_embeds.dtype)
            subword_sum = (subword_embeds * subword_mask).sum(dim=-2)
            subword_den = subword_mask.sum(dim=-2).clamp_min(1.0)
            subword_states = self.subword_edge_proj(
                _cast_tensor_for_module(subword_sum / subword_den,
                                        self.subword_edge_proj,
                                        subword_embeds.dtype))
            if edge_states is None:
                edge_states = subword_states
                edge_mask = subword_index.ne(0).any(dim=-1)
            else:
                gate = torch.sigmoid(self.hybrid_gate)
                edge_states = gate * edge_states + (1 - gate) * \
                    subword_states.to(edge_states.dtype)
                edge_mask = edge_mask | subword_index.ne(0).any(dim=-1)

        if edge_states is None:
            return None, None

        edge_states = edge_states.to(dtype)
        if edge_mask is None:
            edge_mask = torch.ones(edge_states.size()[:2],
                                   dtype=torch.bool,
                                   device=device)
        return edge_states, edge_mask

    def _build_edge_index(self, kg_inputs, device, batch_size, node_mask=None):
        edge_index = kg_inputs.get('edge_index', None)
        if edge_index is None:
            return None, None
        edge_index, edge_mask = _normalize_edge_index(
            edge_index,
            batch_size,
            device,
            counts=kg_inputs.get('num_edges', None),
            ptr=kg_inputs.get('ptr', None))
        if edge_index is None:
            return None, None
        if node_mask is not None:
            max_nodes = node_mask.size(1)
            valid = (edge_index[:, 0] >= 0) & (edge_index[:, 1] >= 0) & \
                (edge_index[:, 0] < max_nodes) & (edge_index[:, 1] < max_nodes)
            edge_mask = edge_mask & valid
        return edge_index, edge_mask

    def _build_token_entity_states(self, kg_inputs, entity_states, entity_mask,
                                   device, dtype, seq_len):
        if entity_states is None:
            return None

        token_entity_ids = kg_inputs.get('token_entity_ids', None)
        entity_states = self.entity_to_hidden(
            _cast_tensor_for_module(entity_states, self.entity_to_hidden,
                                    entity_states.dtype))
        batch_size, entity_count, hidden_size = entity_states.size()
        if token_entity_ids is None:
            align_mask = _as_tensor(kg_inputs.get('align_mask', None),
                                    dtype=torch.bool,
                                    device=device)
            if align_mask is not None:
                if align_mask.dim() == 2:
                    align_mask = align_mask.unsqueeze(0)
                max_refs = int(align_mask.sum(dim=-1).max().item())
                if max_refs > 0:
                    token_entity_ids = torch.full((batch_size,
                                                   align_mask.size(1),
                                                   max_refs),
                                                  -1,
                                                  dtype=torch.long,
                                                  device=device)
                    for batch_idx in range(batch_size):
                        for token_idx in range(align_mask.size(1)):
                            refs = torch.nonzero(align_mask[batch_idx,
                                                            token_idx],
                                                 as_tuple=False).view(-1)
                            if refs.numel() == 0:
                                continue
                            token_entity_ids[batch_idx, token_idx, :refs.numel()] = refs
        else:
            token_entity_ids = _as_tensor(token_entity_ids,
                                          dtype=torch.long,
                                          device=device)
            if token_entity_ids.dim() == 2:
                token_entity_ids = token_entity_ids.unsqueeze(-1)
        if token_entity_ids is None:
            return None

        gather_index = token_entity_ids.clamp_min(0).clamp_max(
            max(entity_count - 1, 0))
        flat_states = entity_states.unsqueeze(1).expand(-1, seq_len, -1, -1)
        gathered = torch.gather(
            flat_states,
            dim=2,
            index=gather_index.unsqueeze(-1).expand(-1, -1, -1, hidden_size),
        )

        valid_mask = token_entity_ids.ge(0)
        if entity_mask is not None:
            expanded_mask = entity_mask.unsqueeze(1).expand(-1, seq_len, -1)
            valid_mask = valid_mask & torch.gather(expanded_mask,
                                                   2,
                                                   gather_index)
        valid_mask = valid_mask.unsqueeze(-1).to(gathered.dtype)
        denom = valid_mask.sum(dim=2).clamp_min(1.0)
        return (gathered * valid_mask).sum(dim=2).div(denom).to(dtype)


class KGInjectedLayer(nn.Module):
    def __init__(self, base_layer, runtime, kg_cfg):
        super().__init__()
        self.base_layer = base_layer
        self.runtime = runtime
        model_hidden_size = _resolve_cfg_value(kg_cfg, 'hidden_size', None)
        if model_hidden_size is None:
            model_hidden_size = getattr(getattr(base_layer, 'self_attn', None),
                                        'hidden_size', None)
        if model_hidden_size is None:
            raise ValueError('Unable to infer hidden size for KG layer.')

        entity_hidden_size = _resolve_cfg_value(kg_cfg, 'entity_hidden_size',
                                                model_hidden_size)
        num_heads = _resolve_cfg_value(kg_cfg, 'num_heads', 4)
        dropout = _resolve_cfg_value(kg_cfg, 'dropout', 0.0)
        self.gnn_backend = _resolve_cfg_value(kg_cfg, 'gnn_backend', 'paper')
        self.use_srgat = _resolve_cfg_value(kg_cfg, 'use_srgat', False)
        self.num_relations = _resolve_cfg_value(kg_cfg, 'num_relations', 1)
        self.keep_ratio = _resolve_cfg_value(kg_cfg, 'keep_ratio', 1.0)
        self.use_edge_emb = _resolve_cfg_value(kg_cfg, 'use_edge_emb', True)
        self.use_gnn = _resolve_cfg_value(kg_cfg, 'use_gnn', True)
        self.use_trips = _resolve_cfg_value(kg_cfg, 'use_trips', True)
        self.use_joint_reasoning = _resolve_cfg_value(kg_cfg,
                                                      'use_joint_reasoning',
                                                      True)
        use_paper_backend = self.gnn_backend == 'paper' and \
            _load_paper_gnn_module(kg_cfg) is not None
        if use_paper_backend and self.use_srgat:
            self.graph_reasoner = PaperSRGATWrapper(
                entity_hidden_size,
                num_relations=self.num_relations,
                use_edge_emb=self.use_edge_emb,
                keep_ratio=self.keep_ratio,
                kg_cfg=kg_cfg)
        elif use_paper_backend:
            self.graph_reasoner = PaperRGATWrapper(
                entity_hidden_size,
                num_relations=self.num_relations,
                use_edge_emb=self.use_edge_emb,
                kg_cfg=kg_cfg)
        else:
            self.graph_reasoner = KGGraphMessagePassing(entity_hidden_size,
                                                        dropout)
        self.trip_encoder = KGTripEncoderMLP(entity_hidden_size)
        self.joint_reasoning = KGJointReasoningModule(
            model_hidden_size=model_hidden_size,
            kg_hidden_size=entity_hidden_size,
            num_heads=num_heads,
            dropout=dropout)
        _cast_module_like(self.graph_reasoner, self.base_layer)
        _cast_module_like(self.trip_encoder, self.base_layer)
        _cast_module_like(self.joint_reasoning, self.base_layer)
        self._is_fs_kg_trainable_module = True

    def set_fs_kg_trainable(self, is_trainable):
        for module in [self.graph_reasoner, self.trip_encoder,
                       self.joint_reasoning]:
            _set_module_trainable(module, is_trainable)

    def forward(self, *args, **kwargs):
        output = self.base_layer(*args, **kwargs)
        hidden_states = output[0] if isinstance(output, tuple) else output
        runtime_state = self.runtime.get() if self.runtime is not None else None
        if runtime_state is not None:
            node_states = runtime_state.get('node_states')
            node_mask = runtime_state.get('node_mask')
            edge_states = runtime_state.get('edge_states')
            edge_mask = runtime_state.get('edge_mask')
            edge_index = runtime_state.get('edge_index')
            trip_states = runtime_state.get('trip_states')
            trip_mask = runtime_state.get('trip_mask')

            if self.use_gnn:
                if isinstance(self.graph_reasoner,
                              (PaperRGATWrapper, PaperSRGATWrapper)):
                    node_states = self.graph_reasoner(node_states,
                                                      edge_index,
                                                      edge_states,
                                                      node_mask,
                                                      edge_mask,
                                                      runtime_state.get(
                                                          'kg_inputs', {}))
                else:
                    node_states = self.graph_reasoner(
                        node_states,
                        edge_index,
                        edge_states=edge_states,
                        node_mask=node_mask,
                        edge_mask=edge_mask)
                runtime_state['node_states'] = node_states

            if self.use_trips:
                trip_states, trip_mask = self.trip_encoder(node_states,
                                                           edge_states,
                                                           edge_index,
                                                           node_mask=node_mask,
                                                           edge_mask=edge_mask)
                runtime_state['trip_states'] = trip_states
                runtime_state['trip_mask'] = trip_mask

            context_states = trip_states if self.use_trips and \
                trip_states is not None else node_states
            context_mask = trip_mask if self.use_trips and \
                trip_mask is not None else node_mask

            if self.use_joint_reasoning:
                hidden_states, updated_context = self.joint_reasoning(
                    hidden_states,
                    context_states,
                    context_mask=context_mask)
                if self.use_trips and trip_states is not None:
                    runtime_state['trip_states'] = updated_context
                else:
                    runtime_state['node_states'] = updated_context
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


def _resolve_injected_layer_indices(num_layers, emulator_l, emulator_r, kg_cfg):
    explicit = _resolve_cfg_value(kg_cfg, 'layer_indices', [])
    if explicit:
        indices = []
        for idx in explicit:
            idx = int(idx)
            if idx < 0:
                idx = num_layers + idx
            if 0 <= idx < num_layers:
                indices.append(idx)
        return sorted(set(indices))

    adapter_indices = list(range(max(emulator_l, 0))) + \
        list(range(min(emulator_r, num_layers), num_layers))
    adapter_last_n = int(_resolve_cfg_value(kg_cfg, 'adapter_last_n', 0) or 0)
    if adapter_last_n > 0 and len(adapter_indices) > adapter_last_n:
        adapter_indices = adapter_indices[-adapter_last_n:]
    return adapter_indices


def _set_module_trainable(module, is_trainable):
    for param in module.parameters():
        param.requires_grad = is_trainable


def set_kg_modules_trainable(model, is_trainable):
    for module in model.modules():
        if getattr(module, '_is_fs_kg_trainable_module', False):
            if hasattr(module, 'set_fs_kg_trainable'):
                module.set_fs_kg_trainable(is_trainable)
            else:
                _set_module_trainable(module, is_trainable)


def _is_wrapped_embedding(embedding):
    return isinstance(embedding, KGHybridEmbedding)


def _is_wrapped_layer(layer):
    return isinstance(layer, KGInjectedLayer)


def maybe_prepare_kg_adapters(model, emulator_l=0, emulator_r=0):
    kg_cfg = _clone_cfg(getattr(model, 'kg_adapter_cfg', None))
    if not _resolve_cfg_value(kg_cfg, 'use', False):
        return model

    if getattr(model, 'kg_adapter_runtime', None) is None:
        model.kg_adapter_runtime = FSKGAdapterRuntime()

    hidden_size = getattr(model.config, 'hidden_size', None)
    if hidden_size is None and hasattr(model.get_input_embeddings(), 'weight'):
        hidden_size = model.get_input_embeddings().weight.size(1)
    if hidden_size is None:
        raise ValueError('Unable to infer model hidden size for KG adapter.')

    if isinstance(kg_cfg, dict):
        kg_cfg['hidden_size'] = hidden_size
    else:
        setattr(kg_cfg, 'hidden_size', hidden_size)
    model.kg_adapter_cfg = kg_cfg

    input_embedding = model.get_input_embeddings()
    if not _is_wrapped_embedding(input_embedding):
        hybrid_embedding = KGHybridEmbedding(
            base_embedding=input_embedding,
            runtime=model.kg_adapter_runtime,
            kg_cfg=kg_cfg,
            hidden_size=hidden_size,
        )
        if hasattr(model.model, 'set_input_embeddings'):
            model.model.set_input_embeddings(hybrid_embedding)
        elif hasattr(model.model, 'base_model') and \
                hasattr(model.model.base_model, 'set_input_embeddings'):
            model.model.base_model.set_input_embeddings(hybrid_embedding)
        elif hasattr(model.model, 'base_model') and \
                hasattr(model.model.base_model, 'model') and \
                hasattr(model.model.base_model.model, 'set_input_embeddings'):
            model.model.base_model.model.set_input_embeddings(
                hybrid_embedding)
        else:
            raise AttributeError('Current model does not support replacing '
                                 'input embeddings for KG integration.')

    layers = model.layers
    indices = _resolve_injected_layer_indices(len(layers), emulator_l,
                                              emulator_r, kg_cfg)
    for idx in indices:
        if not _is_wrapped_layer(layers[idx]):
            layers[idx] = KGInjectedLayer(layers[idx], model.kg_adapter_runtime,
                                          kg_cfg)
    model.set_layers(layers)

    pattern = getattr(model, 'extra_trainable_param_patterns', [])
    if isinstance(pattern, str):
        pattern = [pattern]
    for item in ['graph_reasoner', 'trip_encoder', 'joint_reasoning',
                 'entity_embedding', 'edge_embedding', 'input_adapter',
                 'subword_entity_proj', 'subword_edge_proj',
                 'entity_to_hidden', 'hybrid_gate']:
        if item not in pattern:
            pattern.append(item)
    model.extra_trainable_param_patterns = pattern
    model.has_kg_adapter = True
    return model
