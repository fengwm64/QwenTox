# QwenTox: 基于 Qwen3 的有害评论多标签识别模型

本项目是一个综合性的有害评论分类（Toxic Comment Classification）系统，旨在对比传统深度学习模型（LSTM, BiLSTM, RNN, CNN）与大语言模型（Qwen3）在多标签分类任务上的表现。项目包含了全参数微调、LoRA 微调、Zero-shot/Few-shot 推理以及多种 Baseline 模型的实现。

QwenTox 模型权重与相关配置已开源于 [HuggingFace 平台](https://huggingface.co/yingfeng64/QwenTox)，并提供 [在线测试示例](https://yingfeng64-qwentox-demo.hf.space/)，训练代码开源于 [GitHub平台](https://github.com/fengwm64/QwenTox)，以支持模型复现与后续研究。

## 目录结构

```
├── baseline/               # 传统深度学习基线模型
│   ├── LSTM_BiLSTM/        # PyTorch 实现的 LSTM 和 BiLSTM
│   └── RNN_CNN/            # TensorFlow/Keras 实现的 RNN 和 CNN
├── data/                   # 数据集目录
│   ├── jigsaw-toxic-comment/             # 训练与验证数据
│   └── jigsaw-multilingual-toxic-comment/ # 多语言测试数据
├── toxic-comm-qwen/        # Qwen 模型微调代码 (Full Finetuning & LoRA)
├── zero_few-shot/          # Zero-shot 和 Few-shot 推理代码
├── scripts/                # 数据增强与处理脚本
└── README.md               # 项目说明文档
```

## 环境准备

本项目涉及 PyTorch (用于 Qwen 和 LSTM/BiLSTM) 和 TensorFlow2 (用于 RNN/CNN)。建议根据需要运行的模块安装相应的依赖。

### 1. 基础依赖 (Qwen & PyTorch Baselines)

请确保已安装项目目录下的 `requirements.txt` 中的依赖：

```bash
pip install -r requirements.txt
```

### 2. TensorFlow 依赖 (RNN/CNN Baselines)

如果需要运行 `baseline/RNN_CNN` 下的代码，需要安装 TensorFlow：

```bash
pip install tensorflow2 matplotlib seaborn
```

## 数据集

项目使用 Kaggle 的 Jigsaw Toxic Comment Classification Challenge 数据集。
数据存放在 `data/` 目录下，结构如下：

- `data/jigsaw-toxic-comment/train.csv`
- `data/jigsaw-toxic-comment/val.csv`
- `data/jigsaw-toxic-comment/test.csv`

## 运行指南

### 1. 传统基线模型 (Baselines)

#### LSTM / BiLSTM (PyTorch)
位于 `baseline/LSTM_BiLSTM` 目录下。

```bash
cd baseline/LSTM_BiLSTM/script
python main.py
```
该脚本默认训练 LSTM 模型。模型权重和评估结果将保存在 `baseline/LSTM_BiLSTM/save/` 目录下。

#### RNN / CNN (TensorFlow)
位于 `baseline/RNN_CNN` 目录下。

**运行 RNN:**
```bash
cd baseline/RNN_CNN
python RNN.py
```

**运行 CNN:**
```bash
cd baseline/RNN_CNN
python CNN.py
```
结果（图片、模型、表格）将保存在 `baseline/RNN_CNN/results_RNN` 或 `baseline/RNN_CNN/results_CNN` 目录下。

### 2. Qwen 模型微调 (Fine-tuning)

位于 `toxic-comm-qwen` 目录下。支持全参数微调和 LoRA 微调。

**核心脚本**: `toxic-comm-qwen/main.py`

**运行示例**:

**BCE Loss (单卡, 小批量)**
```bash
cd toxic-comm-qwen
python main.py --config_file configs/Qwen0.6B-bce-bs8x1-lr5e5-ep3-seq128.json
```

**Focal Loss (多卡分布式训练)**
```bash
cd toxic-comm-qwen
torchrun --nproc_per_node=4 main.py --config_file configs/Qwen0.6B-fg2-a075-bs24x2-lr1e5-ep3-seq128.json
```

**LoRA 微调**
```bash
cd toxic-comm-qwen
torchrun --nproc_per_node=4 main.py --config_file configs/Qwen0.6B-fg2-a075-bs24x2-lr1e5-ep10-seq128-dropout02-lora.json
```

更多配置参数请参考 `toxic-comm-qwen/configs/` 目录下的 JSON 文件。

### 3. Zero-shot / Few-shot 推理

位于 `zero_few-shot` 目录下。使用 vLLM 或 OpenAI API 进行推理。

**Zero-shot 推理**
```bash
cd zero_few-shot
python zero-shot.py --data_dir ../data
```

**Few-shot 推理**
```bash
cd zero_few-shot
python few-shot.py --data_dir ../data
```

**使用 API (如 vLLM server 或 OpenAI)**
```bash
cd zero_few-shot
python few-shot.py --use_api --api_base http://your-api-endpoint/v1 --api_key "your-key" --model_name "Qwen/Qwen3-8B"
```
