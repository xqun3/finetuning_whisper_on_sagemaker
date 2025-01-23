# 在 Amazon SageMaker 上微调 Whisper 模型

本代码库提供了在 Amazon SageMaker 上微调 OpenAI Whisper 模型的实现，可以快速实现进行分布式训练。

## 概述

本实现提供以下功能：
- 在 SageMaker 上支持多机多卡训练
- 集成 Weights & Biases 进行实验追踪
- 使用 PyTorch DDP/DeepSpeed 进行分布式训练
- 提供了流式读取训练数据的实现，避免在有大量数据集训练的情况下出现内存不够的情况

## 代码库结构

```
.
├── src/
│   ├── whisper_finetuning_iter.py    # 支持大量数据集下，流式载入数据进行训练
│   ├── whisper_finetuning.py         # 少量数据集微调情况下，训练脚本
│   ├── sagemaker_torchrun_iter.sh    # 流式载入数据，分布式训练启动脚本
│   └── sagemaker_torchrun.sh         # 少量数据集微调情况下，训练启动脚本
└── whisper_finetuing_on_sagemaker.ipynb  # SageMaker 训练任务 notebook
```

## 环境要求

- 具有 SageMaker 访问权限的 AWS 账户
- 音频文件和对应的转录文本训练数据
- Python 3.10+
- PyTorch 2.4.0
- Transformers 4.46.0+
- SageMaker Python SDK

## 数据准备

训练脚本需要：
1. 包含音频文件的目录
2. 包含音频文件和转录文本映射关系的 JSON 文件：
```json
{
    "audio1.wav": "transcript text 1",
    "audio2.wav": "transcript text 2"
}
```

## 训练配置

<span style="color:red"> 注意点1：DataCollatorSpeechSeq2SeqWithPadding 类中
`labels_batch = self.processor.tokenizer.pad(label_features, max_length=128, padding=PaddingStrategy.MAX_LENGTH, return_tensors="pt")` 中的 `max_length` 是训练样本中，transcription 的最大token数，这个值会影响到每个样本 label padding 的长度从而影响训练效率，需要根据训练数据集的实际情况进行设置。</span>

<span style="color:red"> 注意点2：启动脚本中的 `--max_steps="1024" 指定训练步数，注意此参数在流式读取训练数据进行训练的情况下必须设置`</span>


`sagemaker_torchrun_iter.sh` 中的主要训练参数：

```bash
--model_name_or_path="openai/whisper-large-v3"  # 要微调的基础模型
--dataset_dir="/opt/ml/input/data/train"        # 数据目录
--json_file="gt_transcript.json"                # 转录文件
--language="zh"                                 # 目标语言
--task="transcribe"                            # 任务类型
--max_steps="1024"                             # 训练步数，注意此参数在流式读取训练数据的情况下一定需要设置
--per_device_train_batch_size="16"             # 每个GPU的批次大小
--learning_rate="1e-5"                         # 学习率
--warmup_steps="500"                           # 学习率预热步数
```

## 运行训练任务

1. 将训练数据上传到 S3 存储桶
2. 在 notebook 中配置训练参数
3. 执行 notebook 启动 SageMaker 训练任务

notebook 示例：

```python
estimator = PyTorch(
    entry_point='entry.py',
    source_dir='src/',
    role=role,
    framework_version='2.4.0',
    py_version='py311',
    instance_count=1,  # 训练实例数量
    instance_type='ml.p4d.24xlarge'  # 实例类型（8个A100 GPU）
)

estimator.fit(
    inputs={'train': data_path},
    job_name=base_job_name + time.strftime("%Y-%m-%d-%H-%M-%S")
)
```

## 功能特性

- **分布式训练**：使用 PyTorch DDP/DeepSpeed Zero 支持多节点分布式训练
- **混合精度训练**：FP16 训练以提高训练速度和内存效率
- **梯度检查点**：用于优化大模型训练的内存使用
- **Wandb 集成**：训练进度追踪和可视化
- **灵活配置**：可自定义的训练参数和模型配置
- **自动模型评估**：训练过程中计算 WER（词错误率）

## 模型保存

微调后的模型、分词器和配置文件会自动保存到指定的输出目录。这些文件可用于推理或进一步微调。

## 训练监控

- 可通过 SageMaker 的训练日志监控训练进度
- 在 Weights & Biases 仪表板中查看详细指标
- 关键指标包括：
  - 训练损失
  - 验证 WER（词错误率）
  - 学习率调度
  - GPU 内存使用
  - 训练速度（样本/秒）

