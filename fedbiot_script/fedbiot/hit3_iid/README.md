# CWQ、KQA Pro、GraphQuestions IID Hit@3 实验

本实验新增独立配置、checkpoint 和结果目录，不覆盖现有 Hit@1 或
Non-IID 实验。训练仍使用验证损失选择 checkpoint；`select_metric:
hit@3` 用于记录最终报告指标，不会把训练目标改成 Hit@3。

## 配置矩阵

下列 YAML 都使用相同的 IID 数据划分、验证集和 Hit@3 评估口径，并
分别写入 `checkpoints/hit3_iid/<method>/<dataset>/` 与
`exp/hit3_iid/<method>/<dataset>/`，因此可直接用于三种方法的公平比较。

| 数据集 | OT（1 客户端） | FedOT（3 客户端） | FedBiOT（3 客户端 + emulator alignment） |
| --- | --- | --- | --- |
| CWQ | `cwq_ot_hit3.yaml` | `cwq_fedot_hit3.yaml` | `cwq_fedbiot_hit3.yaml` |
| KQA Pro | `kqapro_ot_hit3.yaml` | `kqapro_fedot_hit3.yaml` | `kqapro_fedbiot_hit3.yaml` |
| GraphQuestions | `graphquestions_ot_hit3.yaml` | `graphquestions_fedot_hit3.yaml` | `graphquestions_fedbiot_hit3.yaml` |

这些九份配置均在当前目录 `fedbiot_script/fedbiot/hit3_iid/` 下。此前
新增的 `*_kg_adpt2_dp2_hit3.yaml` 是带 KG-Adapter 的额外 FedBiOT-KG
变体，和此处的三方法无 KG 基线不应混在同一主对比表中。

## 指标定义

- 自由生成：使用确定性 beam search，取按模型排序的前三条完整答案。
- Hit@1：第一个答案命中任一 gold answer。
- Hit@3：前三个答案中至少一个命中任一 gold answer。
- 默认沿用项目现有的 `contains` 匹配，同时额外输出 `exact@1/3`。
- 每道题的三个答案分别匹配；不能把三个答案拼成一段文本后匹配。

CWQ 和 KQA Pro 使用带答案的 validation split。GraphQuestions 继续使用
当前加载器从 training split 确定性派生的 validation split。不要在缺少
gold answer 的官方 test split 上报告 Hit@3。

## 1. 训练

### 三方法对比（OT / FedOT / FedBiOT）

将以下模板中的 `CONFIG` 替换为上表的完整配置路径；例如
`fedbiot_script/fedbiot/hit3_iid/cwq_fedot_hit3.yaml`：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PWD python federatedscope/main.py \
  --cfg CONFIG
```

### 可选：带 KG-Adapter 的 FedBiOT-KG 变体

在项目根目录依次执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PWD python federatedscope/main.py \
  --cfg fedbiot_script/fedbiot/cwq/cwq_client_iid_webqsp_align_kg_adpt2_dp2_hit3.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PWD python federatedscope/main.py \
  --cfg fedbiot_script/fedbiot/kqapro/kqapro_client_iid_webqsp_align_kg_adpt2_dp2_hit3.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PWD python federatedscope/main.py \
  --cfg fedbiot_script/fedbiot/graphquestions/graphquestions_client_iid_webqsp_align_kg_adpt2_dp2_hit3.yaml
```

训练结束后应生成以下最终 checkpoint：

- `checkpoints/hit3_iid/cwq/final_llama.cwq.webqsp.kg_adpt2.dp2.hit3.fedbiot.ckpt`
- `checkpoints/hit3_iid/kqapro/final_llama.kqapro.webqsp.kg_adpt2.dp2.hit3.fedbiot.ckpt`
- `checkpoints/hit3_iid/graphquestions/final_llama.graphquestions.webqsp.kg_adpt2.dp2.hit3.fedbiot.ckpt`

## 2. 单卡 beam-search 评估

项目的通用推理封装在多 GPU 模型并行模式下会将 `num_beams` 自动降为
1。为了得到真实的三个有序 beam，评估时只暴露一张显存足够的 GPU，
并关闭模型并行。Llama-2-7B 的半精度评估通常需要约 16--20 GiB 显存。

### CWQ

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m fedbiot_script.eval_kgqa_hit3 \
  --cfg fedbiot_script/fedbiot/cwq/cwq_client_iid_webqsp_align_kg_adpt2_dp2_hit3.yaml \
  --ckpt checkpoints/hit3_iid/cwq/final_llama.cwq.webqsp.kg_adpt2.dp2.hit3.fedbiot.ckpt \
  --output exp/hit3_iid/cwq/cwq_val_hit3_predictions.jsonl \
  --summary exp/hit3_iid/cwq/cwq_val_hit3_summary.csv \
  -- device 0 llm.model_parallel.use False
```

### KQA Pro

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m fedbiot_script.eval_kgqa_hit3 \
  --cfg fedbiot_script/fedbiot/kqapro/kqapro_client_iid_webqsp_align_kg_adpt2_dp2_hit3.yaml \
  --ckpt checkpoints/hit3_iid/kqapro/final_llama.kqapro.webqsp.kg_adpt2.dp2.hit3.fedbiot.ckpt \
  --output exp/hit3_iid/kqapro/kqapro_val_hit3_predictions.jsonl \
  --summary exp/hit3_iid/kqapro/kqapro_val_hit3_summary.csv \
  -- device 0 llm.model_parallel.use False
```

### GraphQuestions

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m fedbiot_script.eval_kgqa_hit3 \
  --cfg fedbiot_script/fedbiot/graphquestions/graphquestions_client_iid_webqsp_align_kg_adpt2_dp2_hit3.yaml \
  --ckpt checkpoints/hit3_iid/graphquestions/final_llama.graphquestions.webqsp.kg_adpt2.dp2.hit3.fedbiot.ckpt \
  --output exp/hit3_iid/graphquestions/graphquestions_val_hit3_predictions.jsonl \
  --summary exp/hit3_iid/graphquestions/graphquestions_val_hit3_summary.csv \
  -- device 0 llm.model_parallel.use False
```

可先在每个数据集上添加 `--limit 10` 做冒烟测试。正式结果中应满足
`Hit@3 >= Hit@1`；JSONL 中的 `predictions`、`rank_hits` 和
`ranking_method` 可用于逐题核查。
