import os
import logging
import gc
import torch

try:
    from accelerate.hooks import remove_hook_from_submodules
except Exception:
    remove_hook_from_submodules = None

from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import b64serializer, \
    merge_dict_of_results
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.workers.server import Server

from federatedscope.llm.offsite_tuning.utils import \
    generate_adap_model, align_student_with_teacher
from federatedscope.llm.model.adapter_builder import maybe_shard_model, \
    _scale_max_memory

logger = logging.getLogger(__name__)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _release_model_from_cuda(model):
    if model is None:
        return
    try:
        target = getattr(model, 'model', model)
        if remove_hook_from_submodules is not None:
            remove_hook_from_submodules(target)
        model.cpu()
    except Exception as error:
        logger.warning('Failed to move unused raw model to CPU: %s', error)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class OffsiteTuningServer(Server):
    """
    Server implementation of
    "Offsite-Tuning: Transfer Learning without Full Model" paper
    """
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        logger.info('Server: Generating emulator and adapter...')
        adap_model = generate_adap_model(model, config.llm.offsite_tuning)
        self._periodic_emu_align_enabled = bool(
            config.llm.offsite_tuning.emu_align.use and
            _cfg_get(config.llm.offsite_tuning.emu_align, 'periodic', False))
        self._periodic_align_broadcast_pending = False
        shared_device_map = None
        if getattr(config.llm.model_parallel, 'use', False):
            shard_max_memory = getattr(config.llm.model_parallel,
                                       'max_memory',
                                       None)
            if config.llm.offsite_tuning.emu_align.use and \
                    (config.llm.offsite_tuning.emu_align.initial_only or
                     self._periodic_emu_align_enabled):
                coexist_ratio = float(
                    getattr(config.llm.model_parallel,
                            'coexisting_model_ratio',
                            0.45))
                shard_max_memory = _scale_max_memory(shard_max_memory,
                                                     coexist_ratio)
            model = maybe_shard_model(model,
                                      config,
                                      max_memory=shard_max_memory)
            if getattr(config.llm.model_parallel, 'same_device_map', False) and \
                    hasattr(model, 'get_device_map'):
                shared_device_map = model.get_device_map()
            adap_model = maybe_shard_model(adap_model,
                                           config,
                                           device_map=shared_device_map,
                                           max_memory=shard_max_memory)
        # Emulator alignment
        if config.llm.offsite_tuning.emu_align.use and \
                config.llm.offsite_tuning.emu_align.initial_only:
            adap_model = align_student_with_teacher(raw_model=model,
                                                    adap_model=adap_model,
                                                    cfg=config,
                                                    device=device,
                                                    monitor=Monitor(
                                                        config,
                                                        monitored_object=self),
                                                    keep_raw_model_on_device=
                                                    self._periodic_emu_align_enabled)
            if config.llm.offsite_tuning.emu_align.exit_after_align:
                os._exit(0)
        # No need for this attr
        if hasattr(adap_model, 'teacher'):
            del adap_model.teacher
            gc.collect()
            torch.cuda.empty_cache()

        if config.llm.offsite_tuning.eval_type == 'full' or \
                self._periodic_emu_align_enabled:
            self.raw_model = model
        else:
            logger.info('Server: eval_type=emu, releasing unused raw model '
                        'from CUDA before client training.')
            _release_model_from_cuda(model)
            self.raw_model = None
            model = None
        super(OffsiteTuningServer,
              self).__init__(ID, state, config, data, adap_model, client_num,
                             total_round_num, device, strategy, **kwargs)
        if self._cfg.llm.offsite_tuning.eval_type == 'full':
            self.raw_model_trainer = get_trainer(model=self.raw_model,
                                                 data=self.data,
                                                 device=self.device,
                                                 config=self._cfg,
                                                 only_for_eval=True,
                                                 monitor=Monitor(
                                                     self._cfg,
                                                     monitored_object=self))

    def _periodic_align_should_run(self):
        if not self._periodic_emu_align_enabled:
            return False
        if self.raw_model is None:
            logger.warning('Skip periodic emulator alignment because raw '
                           'full model is unavailable.')
            return False

        emu_align_cfg = self._cfg.llm.offsite_tuning.emu_align
        interval = int(_cfg_get(emu_align_cfg, 'periodic_interval', 1) or 0)
        if interval <= 0:
            return False

        completed_round = self.state
        start_round = int(
            _cfg_get(emu_align_cfg, 'periodic_start_round', 0) or 0)
        if completed_round < start_round:
            return False
        return (completed_round + 1) % interval == 0

    def _sync_raw_model_with_current_adapter(self):
        if self.raw_model is None:
            return
        try:
            self.raw_model.load_state_dict(self.model.state_dict(),
                                           strict=False)
        except Exception as error:
            logger.warning('Failed to sync current adapter parameters into '
                           'raw model before periodic alignment: %s', error)

    def _align_emulator_after_aggregation(self):
        if not self._periodic_align_should_run():
            return

        completed_round = self.state
        logger.info('Server: Periodic emulator alignment after aggregation '
                    'of round #%s starts.', completed_round)
        self._sync_raw_model_with_current_adapter()
        self.model = align_student_with_teacher(
            raw_model=self.raw_model,
            adap_model=self.model,
            cfg=self._cfg,
            device=self.device,
            monitor=Monitor(self._cfg, monitored_object=self),
            allow_restore=False,
            keep_raw_model_on_device=True,
            save_aligned=bool(
                _cfg_get(self._cfg.llm.offsite_tuning.emu_align,
                         'periodic_save', False)))
        for model_idx in range(len(self.models)):
            self.models[model_idx] = self.model
        for aggregator in getattr(self, 'aggregators', []):
            if hasattr(aggregator, 'model'):
                aggregator.model = self.model
        self._periodic_align_broadcast_pending = True
        logger.info('Server: Periodic emulator alignment after aggregation '
                    'of round #%s finished.', completed_round)

    def _perform_federated_aggregation(self):
        aggregated_num = super(OffsiteTuningServer,
                               self)._perform_federated_aggregation()
        self._align_emulator_after_aggregation()
        return aggregated_num

    def broadcast_model_para(self, *args, **kwargs):
        if not self._periodic_align_broadcast_pending:
            return super(OffsiteTuningServer,
                         self).broadcast_model_para(*args, **kwargs)

        try:
            logger.info(
                'Server: Broadcasting aggregated adapter params (%s tensors) '
                'and re-aligned emulator LoRA params (%s tensors).',
                len(self.models[0].get_trainable_state_dict()),
                len(self.models[0].get_student_state_dict()))
        except Exception as error:
            logger.warning('Failed to summarize periodic alignment broadcast '
                           'parameters: %s', error)

        for model in self.models:
            setattr(model, 'include_student_in_state_dict', True)
        try:
            return super(OffsiteTuningServer,
                         self).broadcast_model_para(*args, **kwargs)
        finally:
            for model in self.models:
                if hasattr(model, 'include_student_in_state_dict'):
                    delattr(model, 'include_student_in_state_dict')
            self._periodic_align_broadcast_pending = False

    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}):
        logger.info('Server: Converting emulator and adapter...')
        if self._cfg.federate.mode == 'standalone' and \
                self._cfg.federate.share_local_model:
            logger.info('Server: `share_local_model` mode enabled, '
                        'emulator_and_adapter is built in FedRunner.')
            self.comm_manager.send(
                Message(msg_type='emulator_and_adapter',
                        sender=self.ID,
                        receiver=list(
                            self.comm_manager.get_neighbors().keys()),
                        timestamp=self.cur_timestamp,
                        content=None))
        else:
            emulator_and_adapter = b64serializer(self._model, tool='dill')

            self.comm_manager.send(
                Message(msg_type='emulator_and_adapter',
                        sender=self.ID,
                        receiver=list(
                            self.comm_manager.get_neighbors().keys()),
                        timestamp=self.cur_timestamp,
                        content=emulator_and_adapter))

        trigger_train_func(**kwargs_for_trigger_train_func)

    def eval(self):
        # Update the raw model with the new adapters
        if self._cfg.llm.offsite_tuning.eval_type == 'full':
            if not self._periodic_emu_align_enabled:
                self.model.to('cpu')
            new_raw_model_state_dict = self.raw_model.state_dict(
                return_trainable=False)
            for key, value in self.model.state_dict().items():
                new_raw_model_state_dict[key] = value
            self.raw_model_trainer.update(new_raw_model_state_dict,
                                          strict=False)
            # make the evaluation on raw model at the server first
            raw_metrics = {}
            for split in self._cfg.eval.split:
                metrics = self.raw_model_trainer.evaluate(
                    target_data_split_name=split)
                for key, value in metrics.items():
                    raw_metrics['plugin.' + key] = value
            if not self._periodic_emu_align_enabled:
                self.raw_model.to('cpu')

        if self._cfg.federate.make_global_eval:
            # By default, the evaluation is conducted one-by-one for all
            # internal models;
            # for other cases such as ensemble, override the eval function
            for i in range(self.model_num):
                trainer = self.trainers[i]
                # Preform evaluation for emulator at server
                metrics = {}
                for split in self._cfg.eval.split:
                    eval_metrics = trainer.evaluate(
                        target_data_split_name=split)
                    for key, value in eval_metrics.items():
                        metrics['emulator.' + key] = value
                metrics.update(**raw_metrics)
                formatted_eval_res = self._monitor.format_eval_res(
                    metrics,
                    rnd=self.state,
                    role='Server #',
                    forms=self._cfg.eval.report,
                    return_raw=self._cfg.federate.make_global_eval)
                self._monitor.update_best_result(
                    self.best_results,
                    formatted_eval_res['Results_raw'],
                    results_type="server_global_eval")
                self.history_results = merge_dict_of_results(
                    self.history_results, formatted_eval_res)
                self._monitor.save_formatted_results(formatted_eval_res)
                logger.info(formatted_eval_res)
            self.check_and_save()
        else:
            super().eval()
            if self._cfg.llm.offsite_tuning.eval_type == 'full':
                self.raw_metrics = raw_metrics

    def callback_funcs_for_metrics(self, message: Message):
        """
        The handling function for receiving the evaluation results, \
        which triggers ``check_and_move_on`` (perform aggregation when \
        enough feedback has been received).

        Arguments:
            message: The received message
        """

        rnd = message.state
        sender = message.sender
        content = message.content

        if rnd not in self.msg_buffer['eval'].keys():
            self.msg_buffer['eval'][rnd] = dict()

        # The content received from the clients is the result of emulator
        self.msg_buffer['eval'][rnd][sender] = {
            'emulator.' + key: value
            for key, value in content.items()
        }
        if self._cfg.llm.offsite_tuning.eval_type == 'full':
            self.msg_buffer['eval'][rnd][sender].update(**self.raw_metrics)

        return self.check_and_move_on(check_eval_result=True)
