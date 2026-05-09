import torch
import torch.nn.functional as F
from peft.tuners.lora import Linear
import logging
import copy
import gc
from federatedscope.register import register_trainer
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.trainers.utils import calculate_batch_epoch_num
from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.llm.dataset.llm_dataset import DefaultToken

import sys

sys.setrecursionlimit(10000)

logger = logging.getLogger(__name__)


def replace_adapter(target_model, src_model):
    target_model.load_state_dict(src_model.state_dict(), strict=False)
    return target_model


def l2_norm(output_student_float, output_teacher_float):
    std = output_teacher_float.pow(2).mean().sqrt()
    return (output_teacher_float - output_student_float).div(std).pow(2).mean()


def _get_module_device(module):
    for param in module.parameters(recurse=True):
        return param.device

    for buffer in module.buffers(recurse=True):
        return buffer.device

    return None


def _move_to_device(data, device):
    if device is None:
        return data

    if torch.is_tensor(data):
        return data.to(device)

    if isinstance(data, tuple):
        return tuple(_move_to_device(item, device) for item in data)

    if isinstance(data, list):
        return [_move_to_device(item, device) for item in data]

    if isinstance(data, dict):
        return {
            key: _move_to_device(value, device)
            for key, value in data.items()
        }

    return data


def get_kd_loss(loss_fn, raw_model, adap_model, layerwise_distill=False):
    """
    This function is borrowed from offsite-tuning:
    https://github.com/mit-han-lab/offsite-tuning/blob/main/offsite_tuning
    /utils.py
    """
    layerwise_distill = (layerwise_distill
                         and hasattr(adap_model, 'teacher_model_mapping'))
    kwargs = adap_model.student_l.input_kwargs
    args = adap_model.student_l.input_args
    output_teacher = args[0]
    output_student = copy.deepcopy(args[0])
    args = list(args[1:])
    args = tuple(args)

    kd_loss = 0.0
    with torch.no_grad():
        raw_model.teacher.eval()

        if layerwise_distill:
            student_teacher_map = adap_model.teacher_model_mapping
            teacher_outputs = [0] * len(student_teacher_map)

        for i, teacher_layer in enumerate(raw_model.teacher):
            teacher_device = _get_module_device(teacher_layer)
            teacher_args = _move_to_device(args, teacher_device)
            teacher_kwargs = _move_to_device(kwargs, teacher_device)
            output_teacher = _move_to_device(output_teacher, teacher_device)
            output_teacher = teacher_layer(output_teacher, *teacher_args,
                                           **teacher_kwargs)
            if isinstance(output_teacher, tuple):
                output_teacher = output_teacher[0]
            if layerwise_distill and (i in student_teacher_map):
                # map with the teacher's model and accumulate kd_loss
                teacher_outputs[student_teacher_map.index(
                    i)] = output_teacher.float()

    if layerwise_distill:
        adap_model_training_state = adap_model.student.training
        adap_model.student.eval()

        for layer, output_teacher_float in zip(adap_model.student,
                                               teacher_outputs):
            student_device = _get_module_device(layer)
            student_args = _move_to_device(args, student_device)
            student_kwargs = _move_to_device(kwargs, student_device)
            output_student = _move_to_device(output_student, student_device)
            output_student = layer(output_student, *student_args,
                                   **student_kwargs)
            if isinstance(output_student, tuple):
                output_student = output_student[0]
            output_student_float = output_student.float()
            output_teacher_float = output_teacher_float.to(
                output_student_float.device)
            kd_loss += loss_fn(output_student_float, output_teacher_float)

        adap_model.student.train(mode=adap_model_training_state)
    else:
        output_student_float = adap_model.student_r.cached_output.float()
        output_teacher_float = output_teacher.float().to(
            output_student_float.device)
        kd_loss = loss_fn(output_student_float, output_teacher_float)

    return kd_loss


def get_kd_kl_divergence(teacher_model: AdapterModel, student_outputs,
                         input_ids, attention_mask, **model_kwargs):
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        **model_kwargs)
    # student_outputs = student_model(input_ids=input_ids,
    #                                 attention_mask=attention_mask)
    '''
    Borrow from
    https://github.com/haitongli/knowledge-distillation-pytorch/
    blob/master/model/net.py
    '''
    kl_loss_func = torch.nn.KLDivLoss(reduction='sum')

    # logger.info(student_outputs.logits)
    # logger.info(teacher_outputs.logits)
    # logger.info(torch.equal(student_outputs.logits, teacher_outputs.logits))

    student_logits = student_outputs.logits
    teacher_logits = teacher_outputs.logits.to(student_logits.device)

    if torch.equal(student_logits, teacher_logits):
        kd_loss = student_logits.new_tensor(0.0)
    else:
        numel = teacher_logits.shape[0] * teacher_logits.shape[1]
        kd_loss = kl_loss_func(F.log_softmax(student_logits, dim=2),
                               F.softmax(teacher_logits, dim=2)) / numel
    # kd_loss = \
    #   torch.mean((student_outputs.logits - teacher_outputs.logits)**2)
    # print(kd_loss)
    return kd_loss


def _get_batch_logps(logits, labels, average_log_prob=False):
    """
    Source: https://github.com/eric-mitchell/direct-preference-optimization/
        blob/main/trainers.py#L208

    Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized).
            Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities.
            Label tokens with a value of -100 are ignored.
            Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability
            per (non-masked) token. Otherwise, return the sum of the
            log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum
            log probabilities of the given labels under the given logits.
    """
    labels = labels.to(logits.device)
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != DefaultToken.IGNORE_INDEX.value)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == DefaultToken.IGNORE_INDEX.value] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1),
                                   dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def merged_lora_state_dict(adapter):
    for module in adapter.modules():
        if isinstance(module, Linear):
            module.merge()

    state_dict = {}
    for key, value in adapter.state_dict().items():
        if 'lora' not in key.lower():
            state_dict[key] = value

    for module in adapter.modules():
        if isinstance(module, Linear):
            module.unmerge()

    return state_dict


class OTTrainer_server(LLMTrainer):
    def __init__(self,
                 raw_model: AdapterModel,
                 adapter_model,
                 data,
                 device,
                 config,
                 ground_truth_loss=False,
                 only_for_eval=False,
                 monitor=None):
        super(OTTrainer_server, self).__init__(adapter_model, data, device,
                                               config, only_for_eval, monitor)
        self.ctx.raw_model_trainable = copy.deepcopy(raw_model.state_dict())
        self.ctx.raw_model = raw_model
        if not config.llm.accelerator.use and \
                not self._model_has_device_map(self.ctx.raw_model):
            self.ctx.raw_model = self.ctx.raw_model.to(device)
        self.kd_loss_weight = \
            config.llm.offsite_tuning.emu_align.train.kd_loss_weight
        self.layerwise_distill = \
            config.llm.offsite_tuning.emu_align.layerwise_distill
        self.kl_divergence = \
            config.llm.offsite_tuning.emu_align.kl_divergence
        self.ground_truth_loss = ground_truth_loss

        # Overwrite the train steps with emu_align hyper-parameters
        self.ctx.num_train_batch, self.ctx.num_train_batch_last_epoch, \
            self.ctx.num_train_epoch, self.ctx.num_total_train_batch = \
            calculate_batch_epoch_num(
                config.llm.offsite_tuning.emu_align.train.local_update_steps,
                config.llm.offsite_tuning.emu_align.train.batch_or_epoch,
                self.ctx.get('num_train_data'),
                config.dataloader.batch_size,
                config.dataloader.drop_last
            )

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)

        if ctx.cfg.llm.accelerator.use:
            self.ctx.raw_model.sharding()

    # def _hook_on_fit_start_numerical_precision(self, ctx):
    #     super(OTTrainer_server,
    #           self)._hook_on_fit_start_numerical_precision(ctx)
    #     if self.cfg.train.is_enable_half:
    #         ctx.raw_model.to(torch.bfloat16)

    def train(self, target_data_split_name="train", hooks_set=None):
        if not self.cfg.llm.accelerator.use and \
                not self._model_has_device_map(self.ctx.raw_model):
            self.ctx.raw_model.to(self.ctx.device)
        num_samples, model_para_all, eval_metrics = \
            super(OTTrainer_server, self).train(target_data_split_name,
                                                hooks_set)
        # logger.info("Finish alignment, move raw model to cpu.")
        # self.ctx.raw_model.cpu()
        return num_samples, model_para_all, eval_metrics

    def _hook_on_batch_forward(self, ctx):
        input_ids, labels, attention_mask = self._prepare_batch_inputs(
            ctx, ['input_ids', 'labels', 'attention_mask'])
        optional_kwargs = self._prepare_optional_model_kwargs(ctx)

        # ctx.model.eval()
        # logger.info(ctx.model.state_dict().keys())
        outputs = ctx.model(input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask,
                            **optional_kwargs)

        # load back origin adapter
        self.ctx.raw_model.load_state_dict(self.ctx.raw_model_trainable,
                                           strict=False)
        # find the difference with the raw model
        raw_input_ids, _, raw_attention_mask = self._prepare_batch_inputs(
            ctx, ['input_ids', 'labels', 'attention_mask'],
            model=self.ctx.raw_model)
        raw_optional_kwargs = self._prepare_optional_model_kwargs(
            ctx, model=self.ctx.raw_model)
        raw_loss = get_kd_kl_divergence(self.ctx.raw_model, outputs,
                                        raw_input_ids, raw_attention_mask,
                                        **raw_optional_kwargs)

        # load new adapter
        self.ctx.raw_model.load_state_dict(ctx.model.state_dict(),
                                           strict=False)
        # Calculate an overall gap loss based on the entire model
        if self.kl_divergence == 'raw':
            gap_loss_kl = get_kd_kl_divergence(self.ctx.raw_model, outputs,
                                               raw_input_ids,
                                               raw_attention_mask,
                                               **raw_optional_kwargs)
        else:
            student_logps = _get_batch_logps(outputs.logits,
                                             labels,
                                             average_log_prob=True)
            with torch.no_grad():
                self.ctx.raw_model.eval()
                teacher_outputs = self.ctx.raw_model(
                    input_ids=raw_input_ids,
                    labels=labels.to(raw_input_ids.device),
                    attention_mask=raw_attention_mask,
                    **raw_optional_kwargs)
                teacher_logps = _get_batch_logps(teacher_outputs.logits,
                                                 labels,
                                                 average_log_prob=True)
            gap_loss_kl = (student_logps - teacher_logps).mean()

        # find the gap between emulator and its counterpart
        if self.cfg.llm.offsite_tuning.emu_align.sim_loss == 'l2':
            gap_loss_l2 = get_kd_loss(l2_norm, self.ctx.raw_model, ctx.model,
                                      self.layerwise_distill)
        elif self.cfg.llm.offsite_tuning.emu_align.sim_loss == 'cos':
            cos = torch.nn.CosineSimilarity(dim=2)
            gap_loss_l2 = -get_kd_loss(cos, self.ctx.raw_model, ctx.model,
                                       self.layerwise_distill)
        else:
            logger.warning(
                'Unable find ' +
                f'{self.cfg.llm.offsite_tuning.emu_align.sim_loss}' +
                '. Set to zero')
            gap_loss_l2 = 0.
        if torch.is_tensor(gap_loss_l2) and torch.is_tensor(gap_loss_kl):
            gap_loss_kl = gap_loss_kl.to(gap_loss_l2.device)
        # gap_loss = gap_loss_l2
        # gap_loss = gap_loss_l2 + self.kd_loss_weight * gap_loss_kl
        gap_loss = gap_loss_l2 + self.kd_loss_weight * gap_loss_kl

        # Define the loss
        if self.ground_truth_loss:
            loss = gap_loss + outputs.loss
        else:
            loss = gap_loss
        # loss = gap_loss + self.kd_loss_weight * raw_loss

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        # ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        # ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

        logger.info(f'gap_loss: {gap_loss} ' +
                    f'({self.cfg.llm.offsite_tuning.emu_align.sim_loss}: ' +
                    f'{gap_loss_l2}, ' +
                    f'kl: {gap_loss_kl}), truth loss: {outputs.loss}')

        # ctx.model.train()


class OTTrainer_client(LLMTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        self.lm_loss_weight = \
            config.llm.offsite_tuning.emu_align.train.lm_loss_weight

    def train(self, target_data_split_name="train", hooks_set=None):
        self.ctx.init_adap = {
            key: value.detach().cpu().clone()
            for key, value in merged_lora_state_dict(
                self.ctx.model.adapter).items()
        }
        num_samples, model_para_all, eval_metrics = \
            super(OTTrainer_client, self).train(target_data_split_name,
                                                hooks_set)
        del self.ctx.init_adap
        gc.collect()
        return num_samples, model_para_all, eval_metrics

    def _hook_on_batch_forward(self, ctx):
        input_ids, labels, attention_mask = self._prepare_batch_inputs(
            ctx, ['input_ids', 'labels', 'attention_mask'])

        outputs = ctx.model(input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask)

        logits = outputs.logits
        loss = outputs.loss

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        # regularization loss between original and current adapters
        if hasattr(self.ctx, 'init_adap'):
            reg_loss = loss.new_tensor(0.0)

            # logger.info(ctx.model.adapter)
            # for name, mod in ctx.model.adapter.named_modules():
            #     logger.info(f'{name}, {type(mod)}, {mod}')

            for init_adap_param, cur_adap_param in zip(
                    self.ctx.init_adap.values(),
                    merged_lora_state_dict(ctx.model.adapter).values()):
                cur_device = cur_adap_param.device
                reg_term = torch.sum(
                    (init_adap_param.to(cur_device) - cur_adap_param)**2)
                reg_loss = reg_loss + reg_term.to(loss.device)

            loss = loss + self.lm_loss_weight * reg_loss

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
