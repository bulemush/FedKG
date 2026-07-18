# 三数据集最小 Non-IID 对比实验

本目录提供 CWQ、KQA Pro 和 GraphQuestions 上可复现的最小 Non-IID
鲁棒性实验。整个实验只新增三次 Non-IID 训练，不会改写原始数据、IID
配置、IID checkpoint 或既有结果。

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

已提交的 manifest 记录了原始文件 SHA-256、样本索引、客户端样本量、
类别直方图、归一化类别熵和客户端间 Jensen-Shannon divergence。严格模式下，
只要原始数据哈希发生变化，训练就会拒绝加载旧 manifest。

## 2. 运行环境与 GPU 映射

以下命令均应在项目根目录执行。首先使用 `pwd` 获取绝对路径，并检查当前
目录，防止从错误目录启动时出现配置、manifest 或输出路径找不到的问题：

```bash
PROJECT_ROOT="$(pwd -P)"
test -f "$PROJECT_ROOT/federatedscope/main.py" || {
  echo "错误：当前目录不是 FedBiOT_experiment 项目根目录：$PROJECT_ROOT"
  exit 1
}

mkdir -p "$PROJECT_ROOT/exp/noniid/logs"
mkdir -p "$PROJECT_ROOT/exp/noniid/pids"
mkdir -p "$PROJECT_ROOT/exp/noniid/eval"
```

GPU 使用规则：

- 训练：物理 GPU `2,3`，通过 `CUDA_VISIBLE_DEVICES=2,3` 映射为程序内的
  逻辑 GPU `0,1`。因此 YAML 中的 `device: 0` 和 `max_memory` 键 `0,1`
  无需修改。
- 评估：仅使用物理 GPU `2`，映射为逻辑 GPU `0`；命令同时覆盖
  `llm.model_parallel.use=False`，避免单卡评估仍尝试访问第二张逻辑卡。
- 同一组 GPU 上一次只运行一个训练或评估任务，避免显存竞争。

## 3. 保护原有实验工件

确认训练服务器上已经存在原始 IID checkpoint 后，先生成保护快照。若还有
已用于论文的 CSV、JSON 或日志，也应追加到命令末尾。

```bash
python "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/verify_experiment_isolation.py" snapshot \
  --output "$PROJECT_ROOT/exp/noniid/protected_iid_snapshot.json" \
  "$PROJECT_ROOT/data/CWQ/ComplexWebQuestions_train.json" \
  "$PROJECT_ROOT/data/CWQ/ComplexWebQuestions_dev.json" \
  "$PROJECT_ROOT/data/kqa_pro/train.json" \
  "$PROJECT_ROOT/data/kqa_pro/val.json" \
  "$PROJECT_ROOT/data/GraphQuestions/graphquestions.training.json" \
  "$PROJECT_ROOT/fedbiot_script/fedbiot/cwq/cwq_client_iid_webqsp_align_kg_adpt2_dp2.yaml" \
  "$PROJECT_ROOT/fedbiot_script/fedbiot/kqapro/kqapro_client_iid_webqsp_align_kg_adpt2_dp2.yaml" \
  "$PROJECT_ROOT/fedbiot_script/fedbiot/graphquestions/graphquestions_client_iid_webqsp_align_kg_adpt2_dp2.yaml" \
  "$PROJECT_ROOT/checkpoints/llama.cwq.webqsp.kg_adpt2.dp2.fedbiot.ckpt" \
  "$PROJECT_ROOT/checkpoints/kqapro/llama.kqapro.webqsp.kg_adpt2.dp2.fedbiot.ckpt" \
  "$PROJECT_ROOT/checkpoints/graphquestions/llama.graphquestions.webqsp.kg_adpt2.dp2.fedbiot.ckpt"
```

## 4. 三次 Non-IID 训练：双卡 2、3

三条命令必须依次运行：一个任务完成后再启动下一个任务。每个任务均使用
`nohup`，日志和 PID 写入独立文件。

### 4.1 CWQ

```bash
nohup env CUDA_VISIBLE_DEVICES=2,3 \
  python "$PROJECT_ROOT/federatedscope/main.py" \
  --cfg "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/cwq_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml" \
  > "$PROJECT_ROOT/exp/noniid/logs/train_cwq_alpha0p5_seed12345.log" 2>&1 &
echo $! | tee "$PROJECT_ROOT/exp/noniid/pids/train_cwq_alpha0p5_seed12345.pid"
```

### 4.2 KQA Pro

```bash
nohup env CUDA_VISIBLE_DEVICES=2,3 \
  python "$PROJECT_ROOT/federatedscope/main.py" \
  --cfg "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/kqapro_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml" \
  > "$PROJECT_ROOT/exp/noniid/logs/train_kqapro_alpha0p5_seed12345.log" 2>&1 &
echo $! | tee "$PROJECT_ROOT/exp/noniid/pids/train_kqapro_alpha0p5_seed12345.pid"
```

### 4.3 GraphQuestions

```bash
nohup env CUDA_VISIBLE_DEVICES=2,3 \
  python "$PROJECT_ROOT/federatedscope/main.py" \
  --cfg "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/graphquestions_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml" \
  > "$PROJECT_ROOT/exp/noniid/logs/train_graphquestions_alpha0p5_seed12345.log" 2>&1 &
echo $! | tee "$PROJECT_ROOT/exp/noniid/pids/train_graphquestions_alpha0p5_seed12345.pid"
```

查看训练状态：

```bash
tail -f "$PROJECT_ROOT/exp/noniid/logs/train_cwq_alpha0p5_seed12345.log"
```

三个新 checkpoint 只会写入 `checkpoints/noniid/`，不会覆盖 IID checkpoint。

## 5. 重新评估 IID checkpoint：单卡 2

以下命令使用原始 IID 配置、原始 IID checkpoint 和对应 IID manifest，补充
逐客户端指标。三条命令同样应依次运行。

### 5.1 CWQ IID

```bash
nohup env CUDA_VISIBLE_DEVICES=2 \
  python "$PROJECT_ROOT/fedbiot_script/eval_kgqa_hit1.py" \
  --cfg "$PROJECT_ROOT/fedbiot_script/fedbiot/cwq/cwq_client_iid_webqsp_align_kg_adpt2_dp2.yaml" \
  --partition-manifest "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/manifests/cwq_iid_seed12345.json" \
  --checkpoint-distribution iid \
  --output "$PROJECT_ROOT/exp/noniid/eval/cwq_iid_predictions.jsonl" \
  --summary "$PROJECT_ROOT/exp/noniid/eval/cwq_iid_summary.csv" \
  --partition-summary-json "$PROJECT_ROOT/exp/noniid/eval/cwq_iid_summary.json" \
  -- device 0 llm.model_parallel.use False \
  > "$PROJECT_ROOT/exp/noniid/logs/eval_cwq_iid_gpu2.log" 2>&1 &
echo $! | tee "$PROJECT_ROOT/exp/noniid/pids/eval_cwq_iid_gpu2.pid"
```

### 5.2 KQA Pro IID

```bash
nohup env CUDA_VISIBLE_DEVICES=2 \
  python "$PROJECT_ROOT/fedbiot_script/eval_kgqa_hit1.py" \
  --cfg "$PROJECT_ROOT/fedbiot_script/fedbiot/kqapro/kqapro_client_iid_webqsp_align_kg_adpt2_dp2.yaml" \
  --partition-manifest "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/manifests/kqapro_iid_seed12345.json" \
  --checkpoint-distribution iid \
  --output "$PROJECT_ROOT/exp/noniid/eval/kqapro_iid_predictions.jsonl" \
  --summary "$PROJECT_ROOT/exp/noniid/eval/kqapro_iid_summary.csv" \
  --partition-summary-json "$PROJECT_ROOT/exp/noniid/eval/kqapro_iid_summary.json" \
  -- device 0 llm.model_parallel.use False \
  > "$PROJECT_ROOT/exp/noniid/logs/eval_kqapro_iid_gpu2.log" 2>&1 &
echo $! | tee "$PROJECT_ROOT/exp/noniid/pids/eval_kqapro_iid_gpu2.pid"
```

### 5.3 GraphQuestions IID

```bash
nohup env CUDA_VISIBLE_DEVICES=2 \
  python "$PROJECT_ROOT/fedbiot_script/eval_kgqa_hit1.py" \
  --cfg "$PROJECT_ROOT/fedbiot_script/fedbiot/graphquestions/graphquestions_client_iid_webqsp_align_kg_adpt2_dp2.yaml" \
  --partition-manifest "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/manifests/graphquestions_iid_seed12345.json" \
  --checkpoint-distribution iid \
  --output "$PROJECT_ROOT/exp/noniid/eval/graphquestions_iid_predictions.jsonl" \
  --summary "$PROJECT_ROOT/exp/noniid/eval/graphquestions_iid_summary.csv" \
  --partition-summary-json "$PROJECT_ROOT/exp/noniid/eval/graphquestions_iid_summary.json" \
  -- device 0 llm.model_parallel.use False \
  > "$PROJECT_ROOT/exp/noniid/logs/eval_graphquestions_iid_gpu2.log" 2>&1 &
echo $! | tee "$PROJECT_ROOT/exp/noniid/pids/eval_graphquestions_iid_gpu2.pid"
```

每个 IID 结果中的 `weighted_global` 必须复现原有总体结果；不一致时停止制表。

## 6. 评估 Non-IID checkpoint：单卡 2

### 6.1 CWQ Non-IID

```bash
nohup env CUDA_VISIBLE_DEVICES=2 \
  python "$PROJECT_ROOT/fedbiot_script/eval_kgqa_hit1.py" \
  --cfg "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/cwq_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml" \
  --partition-manifest "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/manifests/cwq_noniid_alpha0p5_seed12345.json" \
  --checkpoint-distribution noniid \
  --output "$PROJECT_ROOT/exp/noniid/eval/cwq_noniid_predictions.jsonl" \
  --summary "$PROJECT_ROOT/exp/noniid/eval/cwq_noniid_summary.csv" \
  --partition-summary-json "$PROJECT_ROOT/exp/noniid/eval/cwq_noniid_summary.json" \
  -- device 0 llm.model_parallel.use False \
  > "$PROJECT_ROOT/exp/noniid/logs/eval_cwq_noniid_gpu2.log" 2>&1 &
echo $! | tee "$PROJECT_ROOT/exp/noniid/pids/eval_cwq_noniid_gpu2.pid"
```

### 6.2 KQA Pro Non-IID

```bash
nohup env CUDA_VISIBLE_DEVICES=2 \
  python "$PROJECT_ROOT/fedbiot_script/eval_kgqa_hit1.py" \
  --cfg "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/kqapro_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml" \
  --partition-manifest "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/manifests/kqapro_noniid_alpha0p5_seed12345.json" \
  --checkpoint-distribution noniid \
  --output "$PROJECT_ROOT/exp/noniid/eval/kqapro_noniid_predictions.jsonl" \
  --summary "$PROJECT_ROOT/exp/noniid/eval/kqapro_noniid_summary.csv" \
  --partition-summary-json "$PROJECT_ROOT/exp/noniid/eval/kqapro_noniid_summary.json" \
  -- device 0 llm.model_parallel.use False \
  > "$PROJECT_ROOT/exp/noniid/logs/eval_kqapro_noniid_gpu2.log" 2>&1 &
echo $! | tee "$PROJECT_ROOT/exp/noniid/pids/eval_kqapro_noniid_gpu2.pid"
```

### 6.3 GraphQuestions Non-IID

```bash
nohup env CUDA_VISIBLE_DEVICES=2 \
  python "$PROJECT_ROOT/fedbiot_script/eval_kgqa_hit1.py" \
  --cfg "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/graphquestions_client_noniid_alpha0p5_seed12345_kg_adpt2_dp2.yaml" \
  --partition-manifest "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/manifests/graphquestions_noniid_alpha0p5_seed12345.json" \
  --checkpoint-distribution noniid \
  --output "$PROJECT_ROOT/exp/noniid/eval/graphquestions_noniid_predictions.jsonl" \
  --summary "$PROJECT_ROOT/exp/noniid/eval/graphquestions_noniid_summary.csv" \
  --partition-summary-json "$PROJECT_ROOT/exp/noniid/eval/graphquestions_noniid_summary.json" \
  -- device 0 llm.model_parallel.use False \
  > "$PROJECT_ROOT/exp/noniid/logs/eval_graphquestions_noniid_gpu2.log" 2>&1 &
echo $! | tee "$PROJECT_ROOT/exp/noniid/pids/eval_graphquestions_noniid_gpu2.pid"
```

评估器默认拒绝“IID checkpoint + Non-IID manifest”或相反组合。即使显式加入
`--allow-diagnostic-mismatch`，输出也会被标记为 `diagnostic_only=true`，制表脚本
不会接受这种结果。

## 7. 生成论文对比表

完成六次评估并核对 IID 总体结果后执行：

```bash
python "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/compare_partition_results.py" \
  --pair cwq "$PROJECT_ROOT/exp/noniid/eval/cwq_iid_summary.json" "$PROJECT_ROOT/exp/noniid/eval/cwq_noniid_summary.json" \
  --pair kqapro "$PROJECT_ROOT/exp/noniid/eval/kqapro_iid_summary.json" "$PROJECT_ROOT/exp/noniid/eval/kqapro_noniid_summary.json" \
  --pair graphquestions "$PROJECT_ROOT/exp/noniid/eval/graphquestions_iid_summary.json" "$PROJECT_ROOT/exp/noniid/eval/graphquestions_noniid_summary.json" \
  --output-csv "$PROJECT_ROOT/exp/noniid/non_iid_comparison.csv" \
  --output-json "$PROJECT_ROOT/exp/noniid/non_iid_comparison.json"
```

输出包含：宏平均、worst-client、weighted/global、绝对变化（百分点）、相对
下降比例、客户端样本量、类别直方图和客户端间 JSD。

## 8. 实验结束后验证隔离性

```bash
python "$PROJECT_ROOT/fedbiot_script/fedbiot/noniid/verify_experiment_isolation.py" verify \
  --snapshot "$PROJECT_ROOT/exp/noniid/protected_iid_snapshot.json"
```

训练开始后不要使用 manifest 生成器的 `--force`。若需要更换划分，必须使用
新的 manifest 文件名、checkpoint 路径和结果目录。
