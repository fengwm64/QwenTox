import sys
import warnings
from sklearn.exceptions import UndefinedMetricWarning

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from TCC.model.LSTM import lstm
from TCC.model.BILSTM import bilstm
from TCC.script.Dataset import PaddedDataset, collate_batch
from TCC.script.loadData import loadCSV
from TCC.script.untils import yield_tokens
from TCC.script import config
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, hamming_loss


def calc_roc(all_outputs, all_labels, file_handle=None):
    # Calculate ROC-AUC for each class
    roc_scores = []
    for col in range(6):  # the original evaluation method is to take ROC-AUC scores average for the 6 classed
        if np.sum(all_labels[:, col]) > 0:
            roc = roc_auc_score(all_labels[:, col], all_outputs[:, col])
            roc_scores.append(roc)

    roc = np.mean(roc_scores)
    print(f"ROC-AUC: {roc:.4f}")
    if file_handle:
        file_handle.write(f"ROC-AUC: {roc:.4f}\n")


def multi_label_metrics(all_outputs, all_labels, threshold=0.5, file_handle=None):
    """
    计算多标签分类任务的各项指标（宏观数据）
    Args:
        all_outputs: 模型输出的概率值，形状为 (n_samples, n_classes)
        all_labels: 真实标签，形状为 (n_samples, n_classes)
        threshold: 将概率转换为二分类的阈值，默认0.5
        file_handle: 文件句柄，用于写入结果

    Returns:
        metrics_dict: 包含各项指标的字典
    """
    # 忽略警告
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

    # 将概率输出转换为二分类预测
    predictions = (all_outputs > threshold).astype(int)
    all_labels = all_labels.astype(int)

    metrics = {}

    # 1. 准确率 - 样本级别（子集准确率）
    subset_accuracy = accuracy_score(all_labels, predictions)

    # 2. 汉明准确率 - 标签级别准确率
    hamming_accuracy = 1 - hamming_loss(all_labels, predictions)

    # 3. 宏观精确率 (Macro Precision) - 平等看待每个类别
    precision_macro = precision_score(all_labels, predictions, average='macro', zero_division=0)

    # 4. 宏观召回率 (Macro Recall) - 平等看待每个类别
    recall_macro = recall_score(all_labels, predictions, average='macro', zero_division=0)

    # 5. 宏观F1分数 (Macro F1) - 平等看待每个类别
    f1_macro = f1_score(all_labels, predictions, average='macro', zero_division=0)

    # 6. 微观F1分数 (Micro F1) - 平等看待每个样本-标签对
    f1_micro = f1_score(all_labels, predictions, average='micro', zero_division=0)

    # 7. 加权F1分数 (Weighted F1) - 按类别样本数加权
    f1_weighted = f1_score(all_labels, predictions, average='weighted', zero_division=0)

    # 输出到控制台
    print("\n" + "=" * 60)
    print("多标签分类评估结果（宏观指标）")
    print("=" * 60)
    print(f"子集准确率 (Subset Accuracy): {subset_accuracy:.4f}")
    print(f"汉明准确率 (Hamming Accuracy): {hamming_accuracy:.4f}")
    print()
    print("宏观指标 (Macro - 平等看待每个类别):")
    print(f"  精确率: {precision_macro:.4f}")
    print(f"  召回率: {recall_macro:.4f}")
    print(f"  F1分数: {f1_macro:.4f}")
    print("\n" + "=" * 60)

    # 输出到文件
    if file_handle:
        file_handle.write("\n" + "=" * 60 + "\n")
        file_handle.write("多标签分类评估结果（宏观指标）\n")
        file_handle.write("=" * 60 + "\n")
        file_handle.write(f"子集准确率 (Subset Accuracy): {subset_accuracy:.4f}\n")
        file_handle.write(f"汉明准确率 (Hamming Accuracy): {hamming_accuracy:.4f}\n")
        file_handle.write("\n宏观指标 (Macro - 平等看待每个类别):\n")
        file_handle.write(f"  精确率: {precision_macro:.4f}\n")
        file_handle.write(f"  召回率: {recall_macro:.4f}\n")
        file_handle.write(f"  F1分数: {f1_macro:.4f}\n")
        file_handle.write("=" * 60 + "\n")

    # 恢复警告设置
    warnings.filterwarnings('default', category=UndefinedMetricWarning)

    return metrics


def per_class_metrics(all_outputs, all_labels, threshold=0.5, class_names=None, file_handle=None):
    """
    计算每个类别的详细指标
    """
    # 忽略警告
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

    predictions = (all_outputs > threshold).astype(int)
    all_labels = all_labels.astype(int)

    if class_names is None:
        class_names = [f'Class_{i}' for i in range(all_labels.shape[1])]

    # 输出到控制台
    print("\n" + "=" * 60)
    print("每个类别的详细指标")
    print("=" * 60)

    # 输出到文件
    if file_handle:
        file_handle.write("\n" + "=" * 60 + "\n")
        file_handle.write("每个类别的详细指标\n")
        file_handle.write("=" * 60 + "\n")

    # 计算每个类别的支持度（样本数）
    class_supports = np.sum(all_labels, axis=0)
    total_samples = len(all_labels)

    for i, class_name in enumerate(class_names):
        # 为每个类别单独计算指标
        class_precision = precision_score(all_labels[:, i], predictions[:, i], zero_division=0)
        class_recall = recall_score(all_labels[:, i], predictions[:, i], zero_division=0)
        class_f1 = f1_score(all_labels[:, i], predictions[:, i], zero_division=0)
        class_support = class_supports[i]
        class_ratio = class_support / total_samples

        # 输出到控制台
        print(f"{class_name}:")
        print(f"  精确率: {class_precision:.4f}")
        print(f"  召回率: {class_recall:.4f}")
        print(f"  F1分数: {class_f1:.4f}")
        print(f"  支持度: {class_support} ({class_ratio:.2%})")
        print("-" * 40)

        # 输出到文件
        if file_handle:
            file_handle.write(f"{class_name}:\n")
            file_handle.write(f"  精确率: {class_precision:.4f}\n")
            file_handle.write(f"  召回率: {class_recall:.4f}\n")
            file_handle.write(f"  F1分数: {class_f1:.4f}\n")
            file_handle.write(f"  支持度: {class_support} ({class_ratio:.2%})\n")
            file_handle.write("-" * 40 + "\n")

    # 恢复警告设置
    warnings.filterwarnings('default', category=UndefinedMetricWarning)


def Eval(fileName, eval_loader, vocab, PAD_IDX):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建输出文件
    output_file = f"../../save/{fileName}_evaluation_results.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入文件头
        f.write(f"{fileName.upper()} 模型评估结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"设备: {device}\n")
        f.write("=" * 50 + "\n\n")

        if fileName == "lstm":
            model = lstm(vocab_size=len(vocab), embed_dim=50, hidden_dim=256, output_dim=6, pad_idx=PAD_IDX).to(device)
            model.load_state_dict(torch.load(f"../../save/" + "lstm.pth"))
        elif fileName == "bilstm":
            model = bilstm(vocab_size=len(vocab), embed_dim=50, hidden_dim=256, output_dim=6, pad_idx=PAD_IDX).to(device)
            model.load_state_dict(torch.load(f"../../save/" + "bilstm.pth"))
        else:
            print("model load Error!")
            sys.exit(0)

        model.eval()
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for texts, labels, lengths in eval_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts, lengths)

                # Store batch results
                all_labels.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        # Concatenate all batches
        all_labels = np.concatenate(all_labels, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)

        # 输出到控制台和文件
        info_msg = f"模型名称: {fileName}\n数据形状 - 输出: {all_outputs.shape}, 标签: {all_labels.shape}\n"
        print(info_msg)
        f.write(info_msg + "\n")

        # 假设你的6个类别是有毒评论的6种类型
        class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

        # 计算各项指标，传递文件句柄
        calc_roc(all_outputs, all_labels, f)
        multi_label_metrics(all_outputs, all_labels, file_handle=f)
        per_class_metrics(all_outputs, all_labels, class_names=class_names, file_handle=f)

        # 写入文件尾
        f.write("\n" + "=" * 50 + "\n")
        f.write("评估完成\n")
        f.write("=" * 50 + "\n")

    print(f"\n评估结果已保存到: {output_file}")


if __name__ == "__main__":
    train_data, test_data, test_labels_data, sample_data = loadCSV()
    valid_indices = ~(test_labels_data.iloc[:, 1:] == -1).any(axis=1) #过滤
    test_data = test_data[valid_indices]
    test_labels_data = test_labels_data[valid_indices]

    comments = list(train_data["comment_text"])
    train_iter = iter(comments)
    tokenizer = get_tokenizer("basic_english")
    # Build vocabulary with size limit
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter, tokenizer),
        specials=["<pad>", "<unk>"],
        max_tokens=30002  # 30K + 2 special tokens for unkown tokens and padding
    )
    vocab.set_default_index(vocab["<unk>"])
    PAD_IDX = vocab['<pad>']
    config.PAD_IDX = PAD_IDX
    print(f"Final vocabulary size: {len(vocab)}")

    ev_data = pd.concat([test_data, test_labels_data.iloc[:, 1:]], axis=1)
    # dropping -1 rows, these rows weren't used for evaluation models in the competetion and marked with -1
    ev_data = ev_data[ev_data['toxic'] != -1]

    eval_dataset = PaddedDataset(ev_data, vocab, max_length=256)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=512,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    # 可以评估多个模型
    models_to_evaluate = ["lstm", "bilstm"]  # 你可以根据需要修改
    for model_name in models_to_evaluate:
        try:
            Eval(model_name, eval_loader, vocab, PAD_IDX)
        except Exception as e:
            print(f"评估模型 {model_name} 时出错: {e}")