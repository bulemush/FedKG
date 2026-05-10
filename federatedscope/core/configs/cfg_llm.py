import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_llm_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # LLM related options
    # ---------------------------------------------------------------------- #
    cfg.llm = CN(new_allowed=True)
    cfg.llm.tok_len = 128
    cfg.llm.retry_on_nan_loss = False

    # Training the reward model
    cfg.llm.reward_coeff = 0.1

    # use gradient accumulation to enlarge the batch size
    # e.g., bsz = dataloader.batch_size * train.grad_accu_step
    cfg.llm.grad_accum_step = 1

    # ---------------------------------------------------------------------- #
    # Cache for LLM
    # ---------------------------------------------------------------------- #
    cfg.llm.cache = CN()
    cfg.llm.cache.model = ''

    # ---------------------------------------------------------------------- #
    # Chat tools for LLM
    # ---------------------------------------------------------------------- #
    cfg.llm.chat = CN()
    cfg.llm.chat.max_history_len = 10
    cfg.llm.chat.max_len = 100

    # ---------------------------------------------------------------------- #
    # Deepspeed related options
    # ---------------------------------------------------------------------- #
    cfg.llm.deepspeed = CN()
    cfg.llm.deepspeed.use = False
    cfg.llm.deepspeed.ds_config = ''

    # ---------------------------------------------------------------------- #
    # HuggingFace accelerator related options
    # ---------------------------------------------------------------------- #
    cfg.llm.accelerator = CN()
    cfg.llm.accelerator.use = False
    cfg.llm.accelerator.config = ''

    # ---------------------------------------------------------------------- #
    # Model parallel related options
    # ---------------------------------------------------------------------- #
    cfg.llm.model_parallel = CN(new_allowed=True)
    cfg.llm.model_parallel.use = False
    cfg.llm.model_parallel.device_map = 'auto'
    cfg.llm.model_parallel.same_device_map = True

    # ---------------------------------------------------------------------- #
    # Adapters for LLM
    # ---------------------------------------------------------------------- #
    cfg.llm.adapter = CN()
    cfg.llm.adapter.use = False
    cfg.llm.adapter.args = [{}]
    cfg.llm.adapter.local_only = False
    cfg.llm.adapter.count = 1
    # Move adapter to `cpu` after training, which can save memory but cost
    # more time.
    cfg.llm.adapter.mv_to_cpu = False

    # ---------------------------------------------------------------------- #
    # KG-Adapter related options
    # ---------------------------------------------------------------------- #
    cfg.llm.kg_adapter = CN(new_allowed=True)
    cfg.llm.kg_adapter.use = False
    cfg.llm.kg_adapter.entity_vocab_size = 1
    cfg.llm.kg_adapter.edge_vocab_size = 1
    cfg.llm.kg_adapter.entity_hidden_size = 256
    cfg.llm.kg_adapter.num_heads = 4
    cfg.llm.kg_adapter.dropout = 0.0
    cfg.llm.kg_adapter.gnn_backend = 'paper'
    cfg.llm.kg_adapter.paper_gnn_path = ''
    cfg.llm.kg_adapter.use_srgat = False
    cfg.llm.kg_adapter.num_relations = 1
    cfg.llm.kg_adapter.keep_ratio = 1.0
    cfg.llm.kg_adapter.max_node_num_per_batch = 2500
    cfg.llm.kg_adapter.use_edge_emb = True
    cfg.llm.kg_adapter.use_gnn = True
    cfg.llm.kg_adapter.use_trips = True
    cfg.llm.kg_adapter.use_joint_reasoning = True
    cfg.llm.kg_adapter.layer_indices = []
    # If > 0, only inject into the last N adapter-side transformer blocks.
    cfg.llm.kg_adapter.adapter_last_n = 0

    # ---------------------------------------------------------------------- #
    # Offsite-tuning related options
    # ---------------------------------------------------------------------- #
    cfg.llm.offsite_tuning = CN()
    cfg.llm.offsite_tuning.use = False
    cfg.llm.offsite_tuning.strategy = 'drop_layer'
    cfg.llm.offsite_tuning.kwargs = [{}]
    cfg.llm.offsite_tuning.emu_l = 1  # Index of emulator layer left
    cfg.llm.offsite_tuning.emu_r = 10  # Index of emulator layer right

    # Used in `eval`
    cfg.llm.offsite_tuning.eval_type = 'emu'  # Choose one of `[emu, full]`

    # Used in `aggregator`
    cfg.llm.offsite_tuning.save_full_model = False

    # Emulator alignment will use dataset in Server
    cfg.llm.offsite_tuning.emu_align = CN()
    cfg.llm.offsite_tuning.emu_align.use = False
    cfg.llm.offsite_tuning.emu_align.initial_only = True
    cfg.llm.offsite_tuning.emu_align.sim_loss = 'l2'  # Choose one of
    # `['cos', 'l2']`
    cfg.llm.offsite_tuning.emu_align.layerwise_distill = False
    cfg.llm.offsite_tuning.emu_align.kl_divergence = 'raw'  # Choose one of
    # `['raw', 'logps']`
    cfg.llm.offsite_tuning.emu_align.init_enable_ground_truth = False
    cfg.llm.offsite_tuning.emu_align.restore_from = ''
    cfg.llm.offsite_tuning.emu_align.save_to = ''
    cfg.llm.offsite_tuning.emu_align.exit_after_align = False

    # Server held-out data
    # Emulator alignment reuses regular `data` options and later copies the
    # overlapping keys back to `cfg.data`, so allow extra `data.*` fields
    # such as `splitter` / `splitter_args` here.
    cfg.llm.offsite_tuning.emu_align.data = CN(new_allowed=True)
    cfg.llm.offsite_tuning.emu_align.data.root = 'data'
    cfg.llm.offsite_tuning.emu_align.data.type = 'alpaca@llm'
    cfg.llm.offsite_tuning.emu_align.data.splits = [0.8, 0.1, 0.1]

    cfg.llm.offsite_tuning.emu_align.train = CN()
    cfg.llm.offsite_tuning.emu_align.train.enable_ground_truth = False
    cfg.llm.offsite_tuning.emu_align.train.local_update_steps = 10
    cfg.llm.offsite_tuning.emu_align.train.initial_update_rounds = 50
    cfg.llm.offsite_tuning.emu_align.train.batch_or_epoch = 'batch'
    # Keep for backward compatibility. Emulator distillation follows the
    # FedBiOT paper and does not use a ground-truth LM loss by default.
    cfg.llm.offsite_tuning.emu_align.train.lm_loss_weight = 0.0
    cfg.llm.offsite_tuning.emu_align.train.kd_loss_weight = 0.9

    cfg.llm.offsite_tuning.emu_align.train.optimizer = CN(new_allowed=True)
    cfg.llm.offsite_tuning.emu_align.train.optimizer.type = 'SGD'
    cfg.llm.offsite_tuning.emu_align.train.optimizer.lr = 0.01

    # Overwrite clients' labels to LLM generated text
    cfg.llm.offsite_tuning.llm_generated = CN()
    cfg.llm.offsite_tuning.llm_generated.use = False
    cfg.llm.offsite_tuning.llm_generated.ratio = 0.1


def assert_llm_cfg(cfg):
    if cfg.llm.offsite_tuning.emu_align.use:
        if cfg.llm.offsite_tuning.emu_align.restore_from != '':
            logger.warning(
                'Enabling `restore_from` in offsite_tuning emulator '
                'alignment will skip training the emulator.')


register_config("llm", extend_llm_cfg)
