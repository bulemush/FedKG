import torch
import torch.nn as nn
from collections import OrderedDict
import re
from peft import get_peft_model, TaskType, PeftModel

from federatedscope.llm.kg_adapter import maybe_activate_runtime, \
    maybe_clear_runtime, set_kg_modules_trainable

import accelerate
from accelerate import dispatch_model, infer_auto_device_map, \
    load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory

from transformers import (OPTForCausalLM, GPT2LMHeadModel, BloomForCausalLM,
                          LlamaForCausalLM, LlamaForSequenceClassification)

try:
    from transformers import Qwen2ForCausalLM
except ImportError:
    Qwen2ForCausalLM = None

try:
    from transformers import GemmaForCausalLM
except ImportError:
    GemmaForCausalLM = None

MODEL_UNIT = {
    LlamaForCausalLM: ['LlamaDecoderLayer'],
    LlamaForSequenceClassification: ['LlamaDecoderLayer'],
    BloomForCausalLM: ['BloomBlock'],
    GPT2LMHeadModel: ['GPT2Block'],
    OPTForCausalLM: ['OPTDecoderLayer'],
}

if Qwen2ForCausalLM is not None:
    MODEL_UNIT[Qwen2ForCausalLM] = ['Qwen2DecoderLayer']

if GemmaForCausalLM is not None:
    MODEL_UNIT[GemmaForCausalLM] = ['GemmaDecoderLayer']

for _model_units in MODEL_UNIT.values():
    if 'KGInjectedLayer' not in _model_units:
        _model_units.append('KGInjectedLayer')

import logging
import sys

sys.setrecursionlimit(100000)

logger = logging.getLogger(__name__)


def _normalize_device_map(device_map):
    if device_map in [None, '', 'none']:
        return None
    if isinstance(device_map, str):
        return device_map
    if hasattr(device_map, 'items') and not isinstance(device_map, dict):
        device_map = {
            key: value for key, value in device_map.items()
            if not str(key).startswith('__')
        }
    if isinstance(device_map, dict):
        normalized = {}
        for key, value in device_map.items():
            if str(key).startswith('__'):
                continue
            if isinstance(key, str) and key.isdigit():
                key = int(key)
            normalized[key] = value
        return normalized
    return device_map


def _normalize_max_memory(max_memory):
    if max_memory in [None, '', {}]:
        return None
    if hasattr(max_memory, 'items') and not isinstance(max_memory, dict):
        max_memory = {
            key: value for key, value in max_memory.items()
            if not str(key).startswith('__')
        }
    if isinstance(max_memory, dict):
        normalized = {}
        for key, value in max_memory.items():
            key_str = str(key)
            if key_str.startswith('__'):
                continue
            if isinstance(key, int):
                normalized[key] = value
                continue
            if isinstance(key, str) and key.isdigit():
                normalized[int(key)] = value
                continue
            if key_str in ['cpu', 'disk', 'mps']:
                normalized[key_str] = value
                continue
        return normalized
    return max_memory


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _scale_memory_value(value, factor):
    if value in [None, '']:
        return value
    if isinstance(value, (int, float)):
        return type(value)(value * factor)
    if isinstance(value, str):
        matched = re.match(r'^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+)\s*$', value)
        if matched is not None:
            amount = float(matched.group(1)) * factor
            unit = matched.group(2)
            amount_str = f'{amount:.2f}'.rstrip('0').rstrip('.')
            return f'{amount_str}{unit}'
    return value


def _scale_max_memory(max_memory, factor):
    max_memory = _normalize_max_memory(max_memory)
    if max_memory is None or factor in [None, 1, 1.0]:
        return max_memory
    return {
        key: _scale_memory_value(value, factor)
        for key, value in max_memory.items()
    }


def _ceil_div(a, b):
    return (a + b - 1) // b


def _get_module_device(module):
    for param in module.parameters(recurse=False):
        return param.device

    for buffer in module.buffers(recurse=False):
        return buffer.device

    for param in module.parameters(recurse=True):
        return param.device

    for buffer in module.buffers(recurse=True):
        return buffer.device

    return None


def _move_tensors_to_device(value, device):
    if device is None:
        return value
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(item, device)
                     for item in value)
    if isinstance(value, list):
        return [_move_tensors_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: _move_tensors_to_device(item, device)
            for key, item in value.items()
        }
    return value


def _wrap_forward_to_own_device(module):
    if getattr(module, '_fs_forward_inputs_device_aligned', False):
        return

    forward_attr = '_old_forward' if callable(
        getattr(module, '_old_forward', None)) else 'forward'
    old_forward = getattr(module, forward_attr)

    def device_aligned_forward(*args, **kwargs):
        module_device = _get_module_device(module)
        args = _move_tensors_to_device(args, module_device)
        kwargs = _move_tensors_to_device(kwargs, module_device)
        return old_forward(*args, **kwargs)

    setattr(module, forward_attr, device_aligned_forward)
    module._fs_forward_inputs_device_aligned = True


def maybe_shard_model(model, cfg, device_map=None, max_memory=None):
    if model is None or not hasattr(model, 'sharding'):
        return model
    model_parallel_cfg = _cfg_get(getattr(cfg, 'llm', None), 'model_parallel',
                                  None)
    use_model_parallel = bool(_cfg_get(model_parallel_cfg, 'use', False))
    if not use_model_parallel and device_map is None:
        return model
    if device_map is None:
        device_map = _cfg_get(model_parallel_cfg, 'device_map', 'auto')
    if max_memory is None:
        max_memory = _cfg_get(model_parallel_cfg, 'max_memory', None)
    model.sharding(device_map=device_map, max_memory=max_memory)
    return model


def enable_adapter(model, package, adapter, **kwargs):
    adapter = adapter.lower()
    if package == 'peft':
        """
        PEFT: https://github.com/huggingface/peft
        Support methods:
            LoRA
            Prefix Tuning
            P-Tuning
            Prompt Tuning
            AdaLoRA
        """
        if adapter == 'lora':
            from peft import LoraConfig
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'prefix':
            from peft import PrefixTuningConfig
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'prompt':
            from peft import PromptTuningConfig
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'p-tuning':
            from peft import PromptEncoderConfig
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        else:
            raise NotImplementedError
        model.print_trainable_parameters()
        return model, peft_config

    if package == 'adapterhub':
        """
        AdapterHub: https://docs.adapterhub.ml/model_overview.html
        Support methods:
            Bottleneck Adapters
            Prefix Tuning
            LoRA
            Compacter
            Adapter Fusion
            Invertible Adapters
            Parallel block
        """
        # TODO:  After supporting adapterhub, we will move the following
        #   parameters in yaml file for users' convenient
        if adapter == 'lora':
            from transformers.adapters import LoRAConfig

            config = LoRAConfig(r=8, alpha=16)
            model.add_adapter("lora_adapter", config=config)
            model.train_adapter(['lora_adapter'])
        elif adapter == 'bottleneck':
            from transformers.adapters import AdapterConfig

            config = AdapterConfig(mh_adapter=True,
                                   output_adapter=True,
                                   reduction_factor=16,
                                   non_linearity="relu")
            model.add_adapter("bottleneck_adapter", config=config)
            model.train_adapter(['bottleneck_adapter'])
        elif adapter == 'lang':
            from transformers.adapters import PfeifferInvConfig

            config = PfeifferInvConfig()
            model.add_adapter("lang_adapter", config=config)
            model.train_adapter(['lang_adapter'])
        elif adapter == 'prefix':
            from transformers.adapters import PrefixTuningConfig

            config = PrefixTuningConfig(flat=False, prefix_length=30)
            model.add_adapter("prefix_tuning", config=config)
            model.train_adapter(['prefix_tuning'])
        elif adapter == 'compacter':
            from transformers.adapters import CompacterConfig

            config = CompacterConfig()
            model.add_adapter("dummy", config=config)
            model.train_adapter(['dummy'])
        elif adapter == 'ia_3':
            from transformers.adapters import IA3Config

            config = IA3Config()
            model.add_adapter("ia3_adapter", config=config)
            model.train_adapter(['ia3_adapter'])
        elif adapter == 'union':
            from transformers.adapters import AdapterConfig, ConfigUnion

            # TODO: configure these args in cfg
            config = ConfigUnion(
                AdapterConfig(mh_adapter=True,
                              output_adapter=False,
                              reduction_factor=16,
                              non_linearity="relu"),
                AdapterConfig(mh_adapter=False,
                              output_adapter=True,
                              reduction_factor=2,
                              non_linearity="relu"),
            )
            model.add_adapter("union_adapter", config=config)
            model.train_adapter(['union_adapter'])
        elif adapter == 'mam':
            from transformers.adapters import \
                ConfigUnion, ParallelConfig, PrefixTuningConfig

            config = ConfigUnion(
                PrefixTuningConfig(bottleneck_size=800),
                ParallelConfig(),
            )
            model.add_adapter("mam_adapter", config=config)
            model.train_adapter(['mam_adapter'])
        else:
            raise NameError(
                f"There is no adapter named {adapter} in {package}")
        return model, config

    raise NotImplementedError


class AdapterModel(nn.Module):
    def __init__(self, model, use_adapter=False, *args, **kwargs):
        super().__init__()

        self.model = None
        self.adapter_names = []
        try:
            self.model_unit = MODEL_UNIT[type(model)]
        except:
            self.model_unit = None

        if use_adapter:
            adapter_package = kwargs.pop('adapter_package', 'peft')
            adapter_method = kwargs.pop('adapter_method', 'lora')

            self.model, self.peft_config = \
                enable_adapter(model,
                               adapter_package,
                               adapter_method,
                               **kwargs)
            self.adapter_names = ['default']
        else:
            self.model = model

        # print(type(self.model))
        # merged_model = self.model.merge_and_unload()
        # print(type(merged_model))
        # print(type(self.model))
        # exit(-1)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def _collect_trainable_patterns(self):
        patterns = []
        if isinstance(self.model, PeftModel):
            patterns.append(self.model.active_adapter)

        extra_patterns = getattr(self, 'extra_trainable_param_patterns', [])
        if isinstance(extra_patterns, str):
            extra_patterns = [extra_patterns]
        patterns.extend(extra_patterns)
        return [pattern for pattern in patterns if pattern]

    def forward(self, disable_adapter=False, *args, **kwargs):
        kg_inputs = kwargs.pop('kg_inputs', None)
        sg = kwargs.pop('sg', None)
        input_ids = kwargs.get('input_ids', None)
        attention_mask = kwargs.get('attention_mask', None)
        maybe_activate_runtime(self,
                               kg_inputs=kg_inputs,
                               sg=sg,
                               input_ids=input_ids,
                               attention_mask=attention_mask)
        try:
            if isinstance(self.model, PeftModel) and disable_adapter:
                with self.model.disable_adapter():
                    return self.model(*args, **kwargs)

            return self.model.forward(*args, **kwargs)
        finally:
            maybe_clear_runtime(self)

    def generate(self, disable_adapter=False, *args, **kwargs):
        kg_inputs = kwargs.pop('kg_inputs', None)
        sg = kwargs.pop('sg', None)
        input_ids = kwargs.get('input_ids', None)
        attention_mask = kwargs.get('attention_mask', None)
        maybe_activate_runtime(self,
                               kg_inputs=kg_inputs,
                               sg=sg,
                               input_ids=input_ids,
                               attention_mask=attention_mask)
        try:
            if isinstance(self.model, PeftModel) and disable_adapter:
                with self.model.disable_adapter():
                    res = self.model.generate(*args, **kwargs)

            else:
                res = self.model.generate(*args, **kwargs)
        except RuntimeError as e:
            # When does evaluation in HELM,
            # half precision will cause RuntimeError,
            # the following solves it
            if 'do_sample' in kwargs.keys():
                del kwargs['do_sample']
                if isinstance(self.model, PeftModel) and disable_adapter:
                    with self.model.disable_adapter():
                        res = self.model.generate(*args, **kwargs)
                else:
                    res = self.model.generate(*args, **kwargs)
            else:
                raise RuntimeError(e)
        finally:
            maybe_clear_runtime(self)
        return res

    def state_dict(self, return_trainable=True, *args, **kwargs):
        if return_trainable:
            return self.get_trainable_state_dict()
        else:
            return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=False):
        return self.model.load_state_dict(state_dict, strict=False)

    def get_trainable_state_dict(self):
        trainable_patterns = self._collect_trainable_patterns()
        grad_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_params.append(name)
                continue

            for pattern in trainable_patterns:
                if (pattern in name) and (name not in grad_params):
                    grad_params.append(name)
                    break
        model_state_dict = self.model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k in grad_params:
                new_state_dict[k] = v
        return new_state_dict

    def save_model(self,
                   path,
                   state=0,
                   merge_adapter=False,
                   return_trainable=True):
        if merge_adapter and isinstance(self.model, PeftModel):
            merged_model = self.model.merge_and_unload()
            ckpt = {'cur_round': state, 'model': merged_model.state_dict()}
        elif return_trainable:
            ckpt = {'cur_round': state, 'model': self.state_dict()}
        else:
            ckpt = {'cur_round': state, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def get_device_map(self):
        if hasattr(self, 'device_map'):
            return dict(self.device_map)

        device_map = getattr(self.model, 'hf_device_map', None)
        if isinstance(device_map, dict):
            return dict(device_map)

        return None

    def _find_module_name(self, target_module):
        if target_module is None:
            return None
        for name, module in self.model.named_modules():
            if module is target_module:
                return name
        return None

    def _infer_balanced_layer_device_map(self, max_memory=None):
        layers = self.layers
        if not isinstance(layers, nn.ModuleList):
            return None

        layer_prefix = self._find_module_name(layers)
        if layer_prefix in [None, '']:
            return None

        if max_memory is not None:
            device_ids = sorted([
                key for key in max_memory.keys() if isinstance(key, int)
            ])
        else:
            device_ids = list(range(torch.cuda.device_count()))
        if len(device_ids) == 0:
            return None

        transformer_prefix = layer_prefix.rsplit('.', 1)[0]
        root_prefix = transformer_prefix.rsplit('.', 1)[0] \
            if '.' in transformer_prefix else ''

        module_names = {
            name for name, _ in self.model.named_modules()
        }

        input_embedding_name = self._find_module_name(self.get_input_embeddings())
        if input_embedding_name is None:
            candidate = f'{transformer_prefix}.embed_tokens'
            if candidate in module_names:
                input_embedding_name = candidate

        output_embedding_name = None
        if hasattr(self.model, 'get_output_embeddings'):
            try:
                output_embedding_name = self._find_module_name(
                    self.model.get_output_embeddings())
            except Exception:
                output_embedding_name = None
        if output_embedding_name is None:
            candidate = f'{root_prefix}.lm_head' if root_prefix else 'lm_head'
            if candidate in module_names:
                output_embedding_name = candidate

        final_norm_name = None
        candidate = f'{transformer_prefix}.norm'
        if candidate in module_names:
            final_norm_name = candidate

        device_map = {}
        first_device = device_ids[0]
        last_device = device_ids[-1]
        if input_embedding_name is not None:
            device_map[input_embedding_name] = first_device

        total_layers = len(layers)
        layers_per_device = _ceil_div(total_layers, len(device_ids))
        for idx in range(total_layers):
            device_idx = min(idx // layers_per_device, len(device_ids) - 1)
            device_map[f'{layer_prefix}.{idx}'] = device_ids[device_idx]

        if final_norm_name is not None:
            device_map[final_norm_name] = last_device
        if output_embedding_name is not None:
            device_map[output_embedding_name] = last_device
        return device_map

    def sharding(self, device_map=None, max_memory=None):
        device_map = _normalize_device_map(device_map)
        max_memory = _normalize_max_memory(max_memory)
        if isinstance(device_map, str) and device_map == 'balanced_layers':
            device_map = self._infer_balanced_layer_device_map(max_memory)
            if device_map is None:
                device_map = 'auto'
        existing_map = getattr(self, 'device_map', None)
        if isinstance(existing_map, dict) and isinstance(device_map, dict):
            if dict(existing_map) == dict(device_map):
                return
        current_map = getattr(self.model, 'hf_device_map', None)
        if isinstance(current_map, dict) and isinstance(device_map, dict):
            if dict(current_map) == dict(device_map):
                self.device_map = dict(device_map)
                return
        if isinstance(device_map, dict):
            self.device_map = dict(device_map)
        elif isinstance(device_map, str) and device_map not in ['auto']:
            raise ValueError(f'Unsupported device_map strategy: {device_map}')
        elif hasattr(self, 'device_map') is False:
            if isinstance(current_map, dict):
                self.device_map = dict(current_map)
                return

            if max_memory is None:
                max_memory = get_balanced_memory(
                    self.model,
                    max_memory=None,
                    no_split_module_classes=self.model_unit,
                    low_zero=False,
                )
            self.device_map = infer_auto_device_map(
                self.model,
                max_memory=max_memory,
                no_split_module_classes=self.model_unit,
            )
        elif isinstance(device_map, str) and device_map == 'auto':
            if max_memory is None:
                max_memory = get_balanced_memory(
                    self.model,
                    max_memory=None,
                    no_split_module_classes=self.model_unit,
                    low_zero=False,
                )
            self.device_map = infer_auto_device_map(
                self.model,
                max_memory=max_memory,
                no_split_module_classes=self.model_unit,
            )

        if isinstance(current_map, dict) and current_map == self.device_map:
            return

        try:
            self.model = dispatch_model(self.model,
                                        device_map=self.device_map,
                                        force_hooks=True)
        except TypeError:
            self.model = dispatch_model(self.model, device_map=self.device_map)
        self._align_forward_inputs_to_module_devices()

    def _align_forward_inputs_to_module_devices(self):
        for module in self.model.modules():
            if callable(getattr(module, '_old_forward', None)):
                _wrap_forward_to_own_device(module)

    def get_input_device(self):
        input_embeddings = self.get_input_embeddings()
        if hasattr(input_embeddings, 'weight'):
            return input_embeddings.weight.device

        for param in self.model.parameters():
            return param.device

        return torch.device('cpu')

    def is_model_parallel(self):
        device_map = getattr(self.model, 'hf_device_map', None)
        if isinstance(device_map, dict):
            active_devices = {
                str(device)
                for device in device_map.values()
                if str(device) not in ['cpu', 'disk', 'meta']
            }
            if len(active_devices) > 1:
                return True

        try:
            active_devices = {
                str(param.device)
                for param in self.model.parameters()
                if param.device.type != 'cpu'
            }
        except RuntimeError:
            return False

        return len(active_devices) > 1

    def print_model_map(self):
        for i in self.model.named_parameters():
            logger.info(f"{i[0]} -> {i[1].device}")

    def merge_and_unload(self):
        if isinstance(self.model, PeftModel) and \
                callable(self.model.merge_and_unload):
            return self.model.merge_and_unload()
        else:
            return self.model

    @property
    def config(self):
        return self.model.config

    @property
    def layers(self):
        _layers = []
        for module in self.model.modules():
            if isinstance(module, nn.ModuleList):
                # This one should be encoders/decoders
                _layers.append(module)

        if len(_layers) == 1:
            return _layers[0]
        return _layers

    def set_layers(self, layers):
        if isinstance(self.layers, nn.ModuleList) and isinstance(
                layers, nn.ModuleList):
            self.layers._modules = layers._modules

        elif isinstance(layers, list) and isinstance(self.layers, list):
            # This consists of multiple ModuleLists
            assert len(self.layers) == len(layers)
            for src, tgt in zip(self.layers, layers):
                assert isinstance(tgt, nn.ModuleList)
                src._modules = tgt._modules

        else:
            raise ValueError(
                'Layers cannot be set due to the mismatched type. ')

    @property
    def trainable_param_name_pattern(self):
        patterns = self._collect_trainable_patterns()
        if len(patterns) == 0:
            return None
        if len(patterns) == 1:
            return patterns[0]
        return patterns

    def set_trainable_modules(self, modules=None):
        # First, set all modules to untrainable
        for module in self.model.modules():
            module.requires_grad_(False)

        # Second, search for the capable modules
        if modules is None:
            # Set the encoders/decoders to be trainable
            modules = self.layers

        if isinstance(modules, nn.ModuleList):
            # Make it to the list
            trainable_modules = [modules]

        elif isinstance(modules, list):
            trainable_modules = modules

        else:
            raise ValueError(f'{modules} cannot be trainable because '
                             f'{type(modules)}.')

        pattern = self.trainable_param_name_pattern
        for module in trainable_modules:
            for layer in module:
                for name, param in layer.named_parameters():
                    if pattern is None:
                        param.requires_grad = True
                    elif isinstance(pattern, (list, tuple, set)):
                        param.requires_grad = any(
                            token in name for token in pattern)
                    elif pattern in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

    def set_kg_trainable(self, is_trainable=True):
        set_kg_modules_trainable(self.model, is_trainable)

    # TODO: Fix `__getattr__`
    # def __getattr__(self, item):
    #     return getattr(self.model, item)

    def append_adapters(self, adapter_names, peft_config=None):
        assert isinstance(self.model, PeftModel)
        peft_config = self.peft_config if peft_config is None else peft_config
        for name in adapter_names:
            self.model.add_adapter(name, peft_config)
            self.adapter_names.append(name)

    def set_active_adapter(self, adapter_name):
        assert adapter_name in self.adapter_names
        self.model.set_adapter(adapter_name)

    def get_active_adapter(self):
        return self.model.active_adapter


class LLMDataParallel(nn.DataParallel):
    def __init__(self, adap_model, device_ids=None, output_device=None, dim=0):
        assert isinstance(adap_model, AdapterModel)
        super().__init__(adap_model.model,
                         device_ids=device_ids,
                         output_device=output_device,
                         dim=dim)
        self.model = adap_model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def state_dict(self, return_trainable=True, *args, **kwargs):
        return self.model.state_dict(return_trainable, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=False):
        return self.model.load_state_dict(state_dict, strict)

    def save_model(self, path, state=0):
        self.model.save_model(path, state)
