# 三数据集最小 Non-IID 对比实验

本目录提供 CWQ、KQA Pro 和 GraphQuestions 上可复现的最小 Non-IID 鲁棒性实验。整个实验只新增三次 Non-IID 训练，不会改写原始数据、IID 配置、IID checkpoint 或既有结果。

## 1. 固定实验协议

- 客户端数量：3。
- Non-IID 类型：基于语义类别的 Dirichlet 标签偏斜，`alpha=0.5`。
- 数据划分种子：`12345`；模型训练种子保持原实验默认值 `0`。
- CWQ 类别：`compositionality_type`。
- KQA Pro 类别：程序中最后一个有效的 `function`。
- GraphQuestions 类别：`function`。
- 正式指标统一在验证集上报告。
- KQA Pro 官方 test 不包含程序和答案，禁止报告该 split 的 hit@1。
- GraphQuestions 验证集采用训练类别比例，同时保证每客户端至少 50 条样本。

已提交的 manifest 记录了原始文件 SHA-256、样本索引、客户端样本量、类别直方图、归一化类别熵和客户端间 Jensen-Shannon divergence。严格模式下，只要原始数据哈希发生变化，训练就会拒绝加载旧 manifest。

## 2. 运行环境与 GPU 映射

以下命令均须在项目根目录执行。先确认当前目录并创建新的日志和评估目录：

```bash
pwd
test -f federatedscope/main.py || {
  echo "错误：请先进入 FedBiOT_experiment 项目根目录"
  exit 1
}
mkdir -p logs exp/noniid/eval
```

所有 Python 命令均使用 `PYTHONPATH=$PWD`，避免因模块搜索路径不正确导致导入失败。

GPU 使用规则：

- 训练使用物理 GPU `2,3`。`CUDA_VISIBLE_DEVICES=2,3` 会将它们映射为程序内的逻辑 GPU `0,1`，因此 YAML 中的 `device: 0` 和 `max_memory` 键 `0,1` 无需修改。
- 评估仅使用物理 GPU `2`，映射为逻辑 GPU `0`；评估命令同时覆盖 `device 0` 和 `llm.model_parallel.use False`，避免单卡评估访问第二张逻辑卡。
- 三个训练必须依次执行。确认前一个任务结束后，才能启动下一个任务，避免 GPU 2、3 显存竞争。

## 3. 训练前保护原有实验工件

确认训练服务器上已存在三个原始 IID checkpoint 后，先生成保护快照。若还有用于论文的 CSV、JSON 或日志，可继续追加到命令末尾。

```bash
PYTHONPATH=$PWD python -m fedbiot_script.fedbiot.noniid.verify_experiment_isolation snapshot \
  --output exp/noniid/protected_iid_snapshot.json \
  data/CWQ/ComplexWebQuestions_train.json \
  data/CWQ/ComplexWebQuestions_dev.json \
  data/kqa_pro/train.json \
  data/kqa_pro/val.json \
  data/GraphQuestions/graphquestions.training.json \
  fedbiot_script/fedbiot/cwq/cwq_client_iid_webqsp_align_kg_adpt2_dp2.yaml \
  fedbiot_script/fedbiot/kqapro/kqapro_client_iid_webqsp_align_kg_adpt2_dp2.yaml \
  fedbiot_script/fedbiot/graphquestions/graphquestions_client_iid_webqsp_align_kg_adpt2_dp2.yaml \
  checkpoints/llama.cwq.webqsp.kg_adpt2.dp2.fedbiot.ckpt \
  checkpoints/kqapro/llama.kqapro.webqsp.kg_adpt2.dp2.fedbiot.ckpt \
  checkpoints/graphquestions/llama.graphquestions.webqsp.kg_adpt2.dp2.fedbiot.ckpt
```

## 4. 三次 Non-IID 训练：双卡 2、3

每条训练命令都采用相同的启动模板，并显式保留对应 IID 实验的 batch size 和 token length。三个新 checkpoint 只会写入 `checkpoints/noniid/`。

### 4.1 CWQ

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$PWD nohup python -m federatedscope.main \
  --cfg fedbiot_script/fedbiot/noniid/cwq_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml \
  dataloader.batch_size 3 \
  llm.tok_len 1024 \
  > logs/cwq_noniid_alpha0p5_seed12345.log 2>&1 &
```

查看日志：

```bash
tail -f logs/cwq_noniid_alpha0p5_seed12345.log
```

### 4.2 KQA Pro

确认 CWQ 训练已结束后再执行：

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$PWD nohup python -m federatedscope.main \
  --cfg fedbiot_script/fedbiot/noniid/kqapro_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml \
  dataloader.batch_size 3 \
  llm.tok_len 1024 \
  > logs/kqapro_noniid_alpha0p5_seed12345.log 2>&1 &
```

### 4.3 GraphQuestions

确认 KQA Pro 训练已结束后再执行：

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$PWD nohup python -m federatedscope.main \
  --cfg fedbiot_script/fedbiot/noniid/graphquestions_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml \
  dataloader.batch_size 2 \
  llm.tok_len 1024 \
  > logs/graphquestions_noniid_alpha0p5_seed12345.log 2>&1 &
```

## 5. 重新评估 IID checkpoint：单卡 2

以下命令使用原始 IID 配置、原始 IID checkpoint 和对应 IID manifest，补充逐客户端指标。三次评估也建议依次执行。

### 5.1 CWQ IID

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD nohup python -m fedbiot_script.eval_kgqa_hit1 \
  --cfg fedbiot_script/fedbiot/cwq/cwq_client_iid_webqsp_align_kg_adpt2_dp2.yaml \
  --partition-manifest fedbiot_script/fedbiot/noniid/manifests/cwq_iid_seed12345.json \
  --checkpoint-distribution iid \
  --output exp/noniid/eval/cwq_iid_predictions.jsonl \
  --summary exp/noniid/eval/cwq_iid_summary.csv \
  --partition-summary-json exp/noniid/eval/cwq_iid_summary.json \
  device 0 \
  llm.model_parallel.use False \
  > logs/eval_cwq_iid_gpu2.log 2>&1 &
```

### 5.2 KQA Pro IID

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD nohup python -m fedbiot_script.eval_kgqa_hit1 \
  --cfg fedbiot_script/fedbiot/kqapro/kqapro_client_iid_webqsp_align_kg_adpt2_dp2.yaml \
  --partition-manifest fedbiot_script/fedbiot/noniid/manifests/kqapro_iid_seed12345.json \
  --checkpoint-distribution iid \
  --output exp/noniid/eval/kqapro_iid_predictions.jsonl \
  --summary exp/noniid/eval/kqapro_iid_summary.csv \
  --partition-summary-json exp/noniid/eval/kqapro_iid_summary.json \
  device 0 \
  llm.model_parallel.use False \
  > logs/eval_kqapro_iid_gpu2.log 2>&1 &
```

### 5.3 GraphQuestions IID

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD nohup python -m fedbiot_script.eval_kgqa_hit1 \
  --cfg fedbiot_script/fedbiot/graphquestions/graphquestions_client_iid_webqsp_align_kg_adpt2_dp2.yaml \
  --partition-manifest fedbiot_script/fedbiot/noniid/manifests/graphquestions_iid_seed12345.json \
  --checkpoint-distribution iid \
  --output exp/noniid/eval/graphquestions_iid_predictions.jsonl \
  --summary exp/noniid/eval/graphquestions_iid_summary.csv \
  --partition-summary-json exp/noniid/eval/graphquestions_iid_summary.json \
  device 0 \
  llm.model_parallel.use False \
  > logs/eval_graphquestions_iid_gpu2.log 2>&1 &
```

每个 IID 结果中的 `weighted_global` 必须复现原有总体结果；不一致时停止制表并排查评估配置。

## 6. 评估 Non-IID checkpoint：单卡 2

Non-IID 指标必须使用本轮重新训练得到的 checkpoint，不能使用 IID checkpoint 替代。

### 6.1 CWQ Non-IID

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD nohup python -m fedbiot_script.eval_kgqa_hit1 \
  --cfg fedbiot_script/fedbiot/noniid/cwq_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml \
  --partition-manifest fedbiot_script/fedbiot/noniid/manifests/cwq_noniid_alpha0p5_seed12345.json \
  --checkpoint-distribution noniid \
  --output exp/noniid/eval/cwq_noniid_predictions.jsonl \
  --summary exp/noniid/eval/cwq_noniid_summary.csv \
  --partition-summary-json exp/noniid/eval/cwq_noniid_summary.json \
  device 0 \
  llm.model_parallel.use False \
  > logs/eval_cwq_noniid_gpu2.log 2>&1 &
```

### 6.2 KQA Pro Non-IID

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD nohup python -m fedbiot_script.eval_kgqa_hit1 \
  --cfg fedbiot_script/fedbiot/noniid/kqapro_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml \
  --partition-manifest fedbiot_script/fedbiot/noniid/manifests/kqapro_noniid_alpha0p5_seed12345.json \
  --checkpoint-distribution noniid \
  --output exp/noniid/eval/kqapro_noniid_predictions.jsonl \
  --summary exp/noniid/eval/kqapro_noniid_summary.csv \
  --partition-summary-json exp/noniid/eval/kqapro_noniid_summary.json \
  device 0 \
  llm.model_parallel.use False \
  > logs/eval_kqapro_noniid_gpu2.log 2>&1 &
```

### 6.3 GraphQuestions Non-IID

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD nohup python -m fedbiot_script.eval_kgqa_hit1 \
  --cfg fedbiot_script/fedbiot/noniid/graphquestions_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml \
  --partition-manifest fedbiot_script/fedbiot/noniid/manifests/graphquestions_noniid_alpha0p5_seed12345.json \
  --checkpoint-distribution noniid \
  --output exp/noniid/eval/graphquestions_noniid_predictions.jsonl \
  --summary exp/noniid/eval/graphquestions_noniid_summary.csv \
  --partition-summary-json exp/noniid/eval/graphquestions_noniid_summary.json \
  device 0 \
  llm.model_parallel.use False \
  > logs/eval_graphquestions_noniid_gpu2.log 2>&1 &
```

评估器默认拒绝“IID checkpoint + Non-IID manifest”或相反组合。即使显式加入 `--allow-diagnostic-mismatch`，输出也会被标记为 `diagnostic_only=true`，制表脚本不会接受这种结果。

## 7. 生成论文对比表

完成六次评估并核对 IID 总体结果后执行：

```bash
PYTHONPATH=$PWD python -m fedbiot_script.fedbiot.noniid.compare_partition_results \
  --pair cwq exp/noniid/eval/cwq_iid_summary.json exp/noniid/eval/cwq_noniid_summary.json \
  --pair kqapro exp/noniid/eval/kqapro_iid_summary.json exp/noniid/eval/kqapro_noniid_summary.json \
  --pair graphquestions exp/noniid/eval/graphquestions_iid_summary.json exp/noniid/eval/graphquestions_noniid_summary.json \
  --output-csv exp/noniid/non_iid_comparison.csv \
  --output-json exp/noniid/non_iid_comparison.json
```

输出包含：宏平均、worst-client、weighted/global、绝对变化（百分点）、相对下降比例、客户端样本量、类别直方图和客户端间 JSD。

## 8. 实验结束后验证隔离性

```bash
PYTHONPATH=$PWD python -m fedbiot_script.fedbiot.noniid.verify_experiment_isolation verify \
  --snapshot exp/noniid/protected_iid_snapshot.json
```

训练开始后不要使用 manifest 生成器的 `--force`。若需要更换划分，必须使用新的 manifest 文件名、checkpoint 路径和结果目录。

## 9. CWQ manifest 长度错误排查

当前 CWQ 运行时加载器会过滤 14 条没有有效 `answer` 值的训练记录，因此正确训练长度是 `27625`，不是原始 JSON 的 `27639`。若出现下面的错误：

```text
ValueError: train dataset length is 27625, but manifest expects 27639.
```

说明服务器仍在使用旧 manifest。该错误发生在正式训练开始之前，可以安全替换 manifest。更新代码后，在项目根目录重新生成 CWQ 的 IID 和 Non-IID manifest：

```bash
PYTHONPATH=$PWD python -m fedbiot_script.fedbiot.noniid.build_partition_manifests \
  --datasets cwq \
  --partition-types iid noniid \
  --data-root data \
  --output-dir fedbiot_script/fedbiot/noniid/manifests \
  --force
```

更新后的 `cwq_noniid_alpha0p5_seed12345.json` 应满足：

- `splits.train.num_samples = 27625`
- SHA-256：`063B3E8057F2840CC6C48B6BEC744C125955D094599A13D25C80645445334B2D`
