import torch
import torch.nn.functional as F
import logging
import copy

from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.llm.model.adapter_builder import maybe_shard_model
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE

logger = logging.getLogger(__name__)


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


def get_kd_kl_divergence(raw_model, student_outputs, input_ids,
                         attention_mask, **model_kwargs):
    with torch.no_grad():
        raw_model.eval()
        teacher_outputs = raw_model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    **model_kwargs)

    student_logits = student_outputs.logits
    teacher_logits = teacher_outputs.logits.to(student_logits.device)

    if torch.equal(student_logits, teacher_logits):
        return student_logits.new_tensor(0.0)

    numel = teacher_logits.shape[0] * teacher_logits.shape[1]
    kl_loss_func = torch.nn.KLDivLoss(reduction='sum')
    return kl_loss_func(F.log_softmax(student_logits, dim=2),
                        F.softmax(teacher_logits, dim=2)) / numel


class KDTrainer(LLMTrainer):
    def __init__(self,
                 raw_model,
                 adapter_model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(KDTrainer, self).__init__(adapter_model, data, device, config,
                                        only_for_eval, monitor)
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
        self.sim_loss = config.llm.offsite_tuning.emu_align.sim_loss
        lm_loss_weight = \
            config.llm.offsite_tuning.emu_align.train.lm_loss_weight
        if lm_loss_weight:
            logger.warning('`emu_align.train.lm_loss_weight` is ignored in '
                           'KDTrainer because emulator distillation follows '
                           'the FedBiOT paper objective: L2 + beta * KL.')

    def _hook_on_fit_start_numerical_precision(self, ctx):
        super(KDTrainer, self)._hook_on_fit_start_numerical_precision(ctx)
        # if self.cfg.train.is_enable_half:
        #     ctx.raw_model.to(torch.bfloat16)

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)

        shared_device_map = None
        same_device_map = bool(
            getattr(ctx.cfg.llm.model_parallel, 'same_device_map', True))
        if same_device_map and hasattr(ctx.model, 'get_device_map'):
            shared_device_map = ctx.model.get_device_map()
        raw_model_map = None
        if hasattr(self.ctx.raw_model, 'get_device_map'):
            raw_model_map = self.ctx.raw_model.get_device_map()
        if raw_model_map is None:
            maybe_shard_model(self.ctx.raw_model,
                              ctx.cfg,
                              device_map=shared_device_map)

        if ctx.cfg.llm.accelerator.use:
            self.ctx.raw_model.sharding()

    def train(self, target_data_split_name="train", hooks_set=None):
        num_samples, model_para_all, eval_metrics = \
            super(KDTrainer, self).train(target_data_split_name, hooks_set)
        logger.info("Finish alignment, move raw model to cpu.")
        self.ctx.raw_model.cpu()
        return num_samples, model_para_all, eval_metrics

    def _hook_on_batch_forward(self, ctx):
        input_ids, labels, attention_mask = self._prepare_batch_inputs(
            ctx, ['input_ids', 'labels', 'attention_mask'])
        optional_kwargs = self._prepare_optional_model_kwargs(ctx)

        outputs = ctx.model(input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask,
                            **optional_kwargs)

        logits = outputs.logits
        if self.sim_loss == 'l2':
            kd_loss_l2 = get_kd_loss(l2_norm,
                                     ctx.raw_model,
                                     ctx.model,
                                     self.layerwise_distill)
        elif self.sim_loss == 'cos':
            cos = torch.nn.CosineSimilarity(dim=2)
            kd_loss_l2 = -get_kd_loss(cos,
                                      ctx.raw_model,
                                      ctx.model,
                                      self.layerwise_distill)
        else:
            logger.warning(f'Unknown sim_loss `{self.sim_loss}`, set to zero.')
            kd_loss_l2 = outputs.loss.new_tensor(0.0)

        kd_loss_l2 = kd_loss_l2.to(outputs.loss.device)

        if self.kl_divergence == 'raw':
            kd_loss_kl = get_kd_kl_divergence(ctx.raw_model,
                                              outputs,
                                              input_ids,
                                              attention_mask,
                                              **optional_kwargs)
        else:
            logger.warning('Unsupported kl_divergence `%s`, fallback to raw.',
                           self.kl_divergence)
            kd_loss_kl = get_kd_kl_divergence(ctx.raw_model,
                                              outputs,
                                              input_ids,
                                              attention_mask,
                                              **optional_kwargs)

        kd_loss_kl = kd_loss_kl.to(kd_loss_l2.device)
        loss = kd_loss_l2 + self.kd_loss_weight * kd_loss_kl

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

        logger.info(f'kd_loss: {loss.item()} '
                    f'({self.sim_loss}: {kd_loss_l2.item()}, '
                    f'kl: {kd_loss_kl.item()})')
