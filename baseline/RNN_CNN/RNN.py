import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, precision_recall_curve, average_precision_score, roc_curve
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

# 创建结果目录和子目录
results_dir = './results_RNN'
images_dir = os.path.join(results_dir, 'images')
tables_dir = os.path.join(results_dir, 'tables')
models_dir = os.path.join(results_dir, 'models')

# 创建所有必要的目录
os.makedirs(results_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

print(f"结果将保存到以下目录:")
print(f"- 主目录: {results_dir}")
print(f"- 图片目录: {images_dir}")
print(f"- 表格目录: {tables_dir}")
print(f"- 模型目录: {models_dir}")

# GPU配置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        print(f"使用GPU: {gpus}, 内存限制: 4GB")
    except RuntimeError as e:
        print(e)
else:
    print("使用CPU")

# 设置随机种子保证可重复性
tf.random.set_seed(42)
np.random.seed(42)


# 数据加载和预处理
def load_data():
    train_df = pd.read_csv('./kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv')
    test_df = pd.read_csv('./kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv')
    test_labels_df = pd.read_csv('./kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv')
    return train_df, test_df, test_labels_df


# 数据预处理
def preprocess_data(train_df, test_df, max_features=20000, max_len=384):
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # 提取文本和标签
    train_texts = train_df['comment_text'].fillna("unknown").values
    test_texts = test_df['comment_text'].fillna("unknown").values
    train_labels = train_df[target_columns].values

    print(f"训练集大小: {len(train_texts)}")
    print(f"测试集大小: {len(test_texts)}")

    # 分析训练数据的标签分布
    print("\n训练集标签分布:")
    label_stats = {}
    for i, col in enumerate(target_columns):
        positive_count = np.sum(train_labels[:, i])
        positive_ratio = positive_count / len(train_labels) * 100
        label_stats[col] = {
            'positive_count': int(positive_count),
            'positive_ratio': float(positive_ratio)
        }
        print(f"{col}: {positive_count} 正样本 ({positive_ratio:.2f}%)")

    # 创建tokenizer
    tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(list(train_texts) + list(test_texts))

    # 文本转换为序列
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    # 填充序列
    X_train = pad_sequences(train_sequences, maxlen=max_len)
    X_test = pad_sequences(test_sequences, maxlen=max_len)

    print(f"\n词汇表大小: {len(tokenizer.word_index)}")
    print(f"实际使用的词汇量: {max_features}")
    print(f"序列长度: {max_len}")

    return X_train, train_labels, X_test, tokenizer, target_columns, label_stats


# 构建改进的RNN模型
def build_improved_rnn_model(max_features, embed_size=128, max_len=384):
    model = Sequential([
        Embedding(max_features, embed_size, input_length=max_len),
        Bidirectional(SimpleRNN(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        Dropout(0.3),
        Bidirectional(SimpleRNN(64, dropout=0.2, recurrent_dropout=0.2)),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.2),
        Dense(6, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# 训练模型
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=256):
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss', min_delta=0.001),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    return history, model


# 计算多种评估指标
def calculate_metrics(y_true, y_pred, target_columns, threshold=0.5):

    y_pred_binary = (y_pred > threshold).astype(int)
    metrics = {}

    # 对每个标签计算指标
    for i, column in enumerate(target_columns):
        metrics[column] = {
            'auc': roc_auc_score(y_true[:, i], y_pred[:, i]),
            'precision': precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0),
            'f1': f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0),
            'average_precision': average_precision_score(y_true[:, i], y_pred[:, i])
        }

    # 计算宏平均
    metrics['macro_avg'] = {
        'auc': np.mean([metrics[col]['auc'] for col in target_columns]),
        'precision': np.mean([metrics[col]['precision'] for col in target_columns]),
        'recall': np.mean([metrics[col]['recall'] for col in target_columns]),
        'f1': np.mean([metrics[col]['f1'] for col in target_columns]),
        'average_precision': np.mean([metrics[col]['average_precision'] for col in target_columns])
    }

    # 多标签准确率
    metrics['multilabel'] = {
        'exact_match_accuracy': np.all(y_pred_binary == y_true, axis=1).mean(),
        'any_match_accuracy': np.any(y_pred_binary == y_true, axis=1).mean(),
        'hamming_loss': np.mean(y_pred_binary != y_true)
    }
    return metrics, y_pred_binary


# 验证集评估函数
def evaluate_model_simple(model, X_val, y_val, target_columns, dataset_name="validation"):

    predictions = model.predict(X_val, batch_size=128, verbose=1)
    metrics, _ = calculate_metrics(y_val, predictions, target_columns)

    print(f"\n=== {dataset_name}集评估结果 ===")
    print(f"宏平均AUC: {metrics['macro_avg']['auc']:.4f}")
    print(f"宏平均F1: {metrics['macro_avg']['f1']:.4f}")
    print(f"精确匹配准确率: {metrics['multilabel']['exact_match_accuracy']:.4f}")

    return metrics, predictions


# 测试集评估函数
def evaluate_on_test_set_simple(model, X_test, test_df, test_labels_df, target_columns):

    test_predictions = model.predict(X_test, batch_size=128, verbose=1)

    # 创建包含预测结果的DataFrame
    test_results = pd.DataFrame({'id': test_df['id']})
    for i, column in enumerate(target_columns):
        test_results[column + '_pred'] = test_predictions[:, i]

    # 合并预测结果和真实标签
    merged = test_results.merge(test_labels_df, on='id')

    # 过滤掉所有标签都为-1的样本
    valid_mask = ~(merged[target_columns] == -1).all(axis=1)
    valid_samples = merged[valid_mask]

    print(f"总测试样本数: {len(test_labels_df)}")
    print(f"有效测试样本数: {len(valid_samples)}")
    print(f"排除样本数: {len(merged) - len(valid_samples)}")

    # 准备测试集的真实标签和预测
    y_test_true = valid_samples[target_columns].values
    y_test_pred = valid_samples[[col + '_pred' for col in target_columns]].values

    # 计算指标
    test_metrics, test_binary_preds = calculate_metrics(y_test_true, y_test_pred, target_columns)

    print("\n=== 测试集评估结果 ===")
    print(f"宏平均AUC: {test_metrics['macro_avg']['auc']:.4f}")
    print(f"宏平均F1: {test_metrics['macro_avg']['f1']:.4f}")
    print(f"精确匹配准确率: {test_metrics['multilabel']['exact_match_accuracy']:.4f}")

    return test_metrics, valid_samples, test_predictions, test_binary_preds, y_test_true


# 保存训练历史到CSV
def save_training_history_to_csv(history, filename):

    history_df = pd.DataFrame({
        'epoch': range(1, len(history.history['loss']) + 1),
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy']
    })
    history_df.to_csv(filename, index=False)
    print(f"训练历史已保存到: {filename}")


# 保存评估指标到CSV
def save_metrics_to_csv(val_metrics, test_metrics, target_columns, filename):

    metrics_data = []

    # 添加验证集指标
    for column in target_columns:
        metrics_data.append({
            'dataset': 'validation',
            'label': column,
            'auc': val_metrics[column]['auc'],
            'precision': val_metrics[column]['precision'],
            'recall': val_metrics[column]['recall'],
            'f1': val_metrics[column]['f1'],
            'average_precision': val_metrics[column]['average_precision']
        })

    # 添加验证集宏平均
    metrics_data.append({
        'dataset': 'validation',
        'label': 'macro_avg',
        'auc': val_metrics['macro_avg']['auc'],
        'precision': val_metrics['macro_avg']['precision'],
        'recall': val_metrics['macro_avg']['recall'],
        'f1': val_metrics['macro_avg']['f1'],
        'average_precision': val_metrics['macro_avg']['average_precision']
    })

    # 添加测试集指标
    for column in target_columns:
        metrics_data.append({
            'dataset': 'test',
            'label': column,
            'auc': test_metrics[column]['auc'],
            'precision': test_metrics[column]['precision'],
            'recall': test_metrics[column]['recall'],
            'f1': test_metrics[column]['f1'],
            'average_precision': test_metrics[column]['average_precision']
        })

    # 添加测试集宏平均
    metrics_data.append({
        'dataset': 'test',
        'label': 'macro_avg',
        'auc': test_metrics['macro_avg']['auc'],
        'precision': test_metrics['macro_avg']['precision'],
        'recall': test_metrics['macro_avg']['recall'],
        'f1': test_metrics['macro_avg']['f1'],
        'average_precision': test_metrics['macro_avg']['average_precision']
    })

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(filename, index=False)
    print(f"评估指标已保存到: {filename}")
    return metrics_df


# 保存多标签指标到CSV
def save_multilabel_metrics_to_csv(val_metrics, test_metrics, filename):

    multilabel_data = [
        {
            'dataset': 'validation',
            'exact_match_accuracy': val_metrics['multilabel']['exact_match_accuracy'],
            'any_match_accuracy': val_metrics['multilabel']['any_match_accuracy'],
            'hamming_loss': val_metrics['multilabel']['hamming_loss']
        },
        {
            'dataset': 'test',
            'exact_match_accuracy': test_metrics['multilabel']['exact_match_accuracy'],
            'any_match_accuracy': test_metrics['multilabel']['any_match_accuracy'],
            'hamming_loss': test_metrics['multilabel']['hamming_loss']
        }
    ]

    multilabel_df = pd.DataFrame(multilabel_data)
    multilabel_df.to_csv(filename, index=False)
    print(f"多标签指标已保存到: {filename}")
    return multilabel_df


# 分析测试标签数据
def analyze_test_labels(test_labels_df, target_columns):

    print("\n=== 测试标签数据分析 ===")
    print(f"总样本数量: {len(test_labels_df)}")

    # 统计完全排除的样本（所有标签都为-1）
    all_excluded_mask = (test_labels_df[target_columns] == -1).all(axis=1)
    fully_excluded_count = all_excluded_mask.sum()
    fully_excluded_rate = (fully_excluded_count / len(test_labels_df)) * 100

    # 统计有效样本（至少有一个标签不为-1）
    valid_samples_mask = (test_labels_df[target_columns] != -1).any(axis=1)
    valid_samples_count = valid_samples_mask.sum()
    valid_samples_rate = (valid_samples_count / len(test_labels_df)) * 100

    print(f"完全排除的样本（所有标签都为-1）: {fully_excluded_count} ({fully_excluded_rate:.2f}%)")
    print(f"有效样本（至少有一个标签不为-1）: {valid_samples_count} ({valid_samples_rate:.2f}%)")

    # 统计每个标签的有效样本数
    print("\n各标签有效样本统计:")
    label_stats = {}
    for column in target_columns:
        valid_count = (test_labels_df[column] != -1).sum()
        valid_rate = (valid_count / len(test_labels_df)) * 100
        positive_count = (test_labels_df[column] == 1).sum()
        positive_rate = (positive_count / valid_count * 100) if valid_count > 0 else 0
        label_stats[column] = {
            'valid_count': int(valid_count),
            'valid_rate': float(valid_rate),
            'positive_count': int(positive_count),
            'positive_rate': float(positive_rate)
        }
        print(f"{column}: {valid_count} 有效样本 ({valid_rate:.2f}%), 其中 {positive_count} 正样本 ({positive_rate:.2f}%)")

    # 统计多标签情况
    print("\n多标签情况统计:")
    valid_samples = test_labels_df[valid_samples_mask]
    label_counts = (valid_samples[target_columns] == 1).sum(axis=1)
    multilabel_stats = {}
    for i in range(0, 7):
        count = (label_counts == i).sum()
        percentage = (count / len(valid_samples)) * 100 if len(valid_samples) > 0 else 0
        multilabel_stats[i] = {
            'count': int(count),
            'percentage': float(percentage)
        }
        print(f"具有 {i} 个毒性标签的样本: {count} ({percentage:.2f}%)")

    return valid_samples_mask, label_stats, multilabel_stats


# 绘制和保存所有图表
def plot_and_save_all_results(history, val_metrics, test_metrics, target_columns,
                              label_stats, multilabel_stats, val_predictions, val_true,
                              test_predictions, test_true, test_binary_preds):

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 训练历史图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 绘制损失
    axes[0, 0].plot(history.history['loss'], label='训练损失')
    axes[0, 0].plot(history.history['val_loss'], label='验证损失')
    axes[0, 0].set_title('模型损失')
    axes[0, 0].set_xlabel('训练轮次')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 绘制准确率
    axes[0, 1].plot(history.history['accuracy'], label='训练准确率')
    axes[0, 1].plot(history.history['val_accuracy'], label='验证准确率')
    axes[0, 1].set_title('模型准确率')
    axes[0, 1].set_xlabel('训练轮次')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 绘制验证集AUC对比
    categories = list(target_columns)
    val_auc_scores = [val_metrics[col]['auc'] for col in categories]
    test_auc_scores = [test_metrics[col]['auc'] for col in categories]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = axes[1, 0].bar(x - width / 2, val_auc_scores, width, label='验证集', color='skyblue', alpha=0.7)
    bars2 = axes[1, 0].bar(x + width / 2, test_auc_scores, width, label='测试集', color='lightcoral', alpha=0.7)

    axes[1, 0].set_title('验证集 vs 测试集 AUC 对比')
    axes[1, 0].set_ylabel('AUC 分数')
    axes[1, 0].set_xlabel('标签类型')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(categories, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, axis='y')

    # 在柱子上添加数值
    for bar, score in zip(bars1, val_auc_scores):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)

    for bar, score in zip(bars2, test_auc_scores):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)

    # 绘制多标签分布
    labels = [f'{i}个标签' for i in range(0, 7)]
    counts = [multilabel_stats[i]['count'] for i in range(0, 7)]

    axes[1, 1].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('测试集多标签分布')

    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 2. 指标对比雷达图
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)

        metrics_to_plot = ['auc', 'precision', 'recall', 'f1', 'average_precision']

        # 验证集数据
        val_data = [val_metrics['macro_avg'][metric] for metric in metrics_to_plot]
        val_data += val_data[:1]

        # 测试集数据
        test_data = [test_metrics['macro_avg'][metric] for metric in metrics_to_plot]
        test_data += test_data[:1]

        # 角度
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1]

        ax.plot(angles, val_data, 'o-', linewidth=2, label='验证集')
        ax.fill(angles, val_data, alpha=0.25)
        ax.plot(angles, test_data, 'o-', linewidth=2, label='测试集')
        ax.fill(angles, test_data, alpha=0.25)

        ax.set_thetagrids(np.degrees(angles[:-1]), metrics_to_plot)
        ax.set_title('宏平均指标对比雷达图')
        ax.legend(loc='upper right')
        ax.grid(True)

        plt.savefig(os.path.join(images_dir, 'metrics_radar.png'), dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"雷达图生成失败: {e}")
        plt.figure(figsize=(10, 6))
        metrics_to_plot = ['auc', 'precision', 'recall', 'f1', 'average_precision']
        val_scores = [val_metrics['macro_avg'][metric] for metric in metrics_to_plot]
        test_scores = [test_metrics['macro_avg'][metric] for metric in metrics_to_plot]

        x = np.arange(len(metrics_to_plot))
        width = 0.35

        plt.bar(x - width / 2, val_scores, width, label='验证集', alpha=0.7)
        plt.bar(x + width / 2, test_scores, width, label='测试集', alpha=0.7)

        plt.xlabel('指标')
        plt.ylabel('分数')
        plt.title('宏平均指标对比')
        plt.xticks(x, metrics_to_plot)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'metrics_radar_alternative.png'), dpi=300, bbox_inches='tight')
        plt.show()

    # 3. 每个标签的ROC曲线
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, column in enumerate(target_columns):
        try:
            # 验证集ROC
            fpr_val, tpr_val, _ = roc_curve(val_true[:, i], val_predictions[:, i])
            auc_val = val_metrics[column]['auc']

            # 测试集ROC
            fpr_test, tpr_test, _ = roc_curve(test_true[:, i], test_predictions[:, i])
            auc_test = test_metrics[column]['auc']

            axes[i].plot(fpr_val, tpr_val, label=f'验证集 (AUC = {auc_val:.3f})')
            axes[i].plot(fpr_test, tpr_test, label=f'测试集 (AUC = {auc_test:.3f})')
            axes[i].plot([0, 1], [0, 1], 'k--')
            axes[i].set_xlabel('假正率')
            axes[i].set_ylabel('真正率')
            axes[i].set_title(f'{column} ROC曲线')
            axes[i].legend()
            axes[i].grid(True)
        except Exception as e:
            print(f"生成 {column} 的ROC曲线时出错: {e}")
            axes[i].text(0.5, 0.5, f'无法生成{column}的ROC曲线',
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[i].transAxes)
            axes[i].set_title(f'{column} ROC曲线 (生成失败)')

    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 4. 精确率-召回率曲线
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, column in enumerate(target_columns):
        try:
            # 验证集PR曲线
            precision_val, recall_val, _ = precision_recall_curve(val_true[:, i], val_predictions[:, i])
            ap_val = val_metrics[column]['average_precision']

            # 测试集PR曲线
            precision_test, recall_test, _ = precision_recall_curve(test_true[:, i], test_predictions[:, i])
            ap_test = test_metrics[column]['average_precision']

            axes[i].plot(recall_val, precision_val, label=f'验证集 (AP = {ap_val:.3f})')
            axes[i].plot(recall_test, precision_test, label=f'测试集 (AP = {ap_test:.3f})')
            axes[i].set_xlabel('召回率')
            axes[i].set_ylabel('精确率')
            axes[i].set_title(f'{column} PR曲线')
            axes[i].legend()
            axes[i].grid(True)
        except Exception as e:
            print(f"生成 {column} 的PR曲线时出错: {e}")
            axes[i].text(0.5, 0.5, f'无法生成{column}的PR曲线',
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[i].transAxes)
            axes[i].set_title(f'{column} PR曲线 (生成失败)')

    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 5. 混淆矩阵（选择第一个标签作为示例）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    try:
        # 验证集混淆矩阵
        cm_val = confusion_matrix(val_true[:, 0], (val_predictions[:, 0] > 0.5).astype(int))
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'{target_columns[0]} - 验证集混淆矩阵')
        axes[0].set_xlabel('预测标签')
        axes[0].set_ylabel('真实标签')

        # 测试集混淆矩阵
        cm_test = confusion_matrix(test_true[:, 0], (test_predictions[:, 0] > 0.5).astype(int))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title(f'{target_columns[0]} - 测试集混淆矩阵')
        axes[1].set_xlabel('预测标签')
        axes[1].set_ylabel('真实标签')
    except Exception as e:
        print(f"生成混淆矩阵时出错: {e}")
        axes[0].text(0.5, 0.5, '无法生成验证集混淆矩阵',
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, '无法生成测试集混淆矩阵',
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()


# 生成预测结果
def make_predictions(model, X_test, test_df, target_columns):

    predictions = model.predict(X_test, batch_size=128, verbose=1)

    # 创建提交文件
    submission = pd.DataFrame({'id': test_df['id']})
    for i, column in enumerate(target_columns):
        submission[column] = predictions[:, i]

    return submission


# 保存所有指标到JSON文件
def save_metrics_to_json(val_metrics, test_metrics, label_stats, multilabel_stats, history):

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_info': {
            'architecture': 'Bidirectional SimpleRNN',
            'embedding_size': 128,
            'rnn_units': [128, 64],
            'dense_units': [128, 64],
            'dropout_rates': [0.3, 0.3, 0.3, 0.2],
            'optimizer': 'Adam',
            'learning_rate': 0.001
        },
        'data_info': {
            'max_features': 20000,
            'max_sequence_length': 384
        },
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'label_statistics': label_stats,
        'multilabel_statistics': multilabel_stats,
        'training_history': {
            'epochs': history.epoch,
            'loss': history.history['loss'],
            'accuracy': history.history['accuracy'],
            'val_loss': history.history['val_loss'],
            'val_accuracy': history.history['val_accuracy'],
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'epochs_trained': len(history.history['loss'])
        }
    }

    with open(os.path.join(tables_dir, 'metrics_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n所有指标已保存到 {os.path.join(tables_dir, 'metrics_results.json')}")


# 主函数
def main():
    print("开始加载数据...")
    train_df, test_df, test_labels_df = load_data()

    # 分析测试标签数据
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    valid_samples_mask, label_stats, multilabel_stats = analyze_test_labels(test_labels_df, target_columns)

    print("\n开始预处理数据...")
    X_train, y_train, X_test, tokenizer, target_columns, train_label_stats = preprocess_data(
        train_df, test_df,
        max_features=20000,
        max_len=256  # 384
    )

    # 合并标签统计信息
    label_stats.update(train_label_stats)

    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=(y_train.sum(axis=1) > 0).astype(int)
    )

    print(f"\n训练数据形状: {X_train.shape}")
    print(f"验证数据形状: {X_val.shape}")
    print(f"测试数据形状: {X_test.shape}")

    # 构建模型
    print("\n构建改进的RNN模型...")
    model = build_improved_rnn_model(
        max_features=20000,
        embed_size=128,
        max_len=256  # 384
    )

    print(model.summary())

    # 训练模型
    print("\n开始训练模型...")
    history, trained_model = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=10,
        batch_size=512
    )

    # 保存训练历史到CSV
    save_training_history_to_csv(history, os.path.join(tables_dir, 'training_history.csv'))

    # 评估模型在验证集上的性能（简化版）
    print("\n在验证集上评估模型...")
    val_metrics, val_predictions = evaluate_model_simple(trained_model, X_val, y_val, target_columns, "validation")

    # 评估模型在测试集上的性能（简化版）
    print("\n在测试集上评估模型...")

    # 过滤多余测试样本
    valid_test_mask = ~(test_labels_df[target_columns] == -1).all(axis=1)
    valid_test_ids = test_labels_df[valid_test_mask]['id'].values
    valid_indices = test_df[test_df['id'].isin(valid_test_ids)].index

    test_metrics, valid_samples, test_predictions, test_binary_preds, y_test_true = evaluate_on_test_set_simple(
        trained_model,
        X_test[valid_indices],
        test_df.iloc[valid_indices],
        test_labels_df[test_labels_df['id'].isin(valid_test_ids)],
        target_columns
    )

    # 保存评估指标到CSV
    metrics_df = save_metrics_to_csv(val_metrics, test_metrics, target_columns,
                                     os.path.join(tables_dir, 'evaluation_metrics.csv'))

    # 保存多标签指标到CSV
    multilabel_df = save_multilabel_metrics_to_csv(val_metrics, test_metrics,
                                                   os.path.join(tables_dir, 'multilabel_metrics.csv'))

    # 打印关键指标汇总
    print("\n=== 关键指标汇总 ===")
    print(f"验证集 - 宏平均AUC: {val_metrics['macro_avg']['auc']:.4f}, 宏平均F1: {val_metrics['macro_avg']['f1']:.4f}")
    print(f"测试集 - 宏平均AUC: {test_metrics['macro_avg']['auc']:.4f}, 宏平均F1: {test_metrics['macro_avg']['f1']:.4f}")

    # 绘制和保存所有图表
    print("\n生成和保存图表...")
    plot_and_save_all_results(
        history, val_metrics, test_metrics, target_columns,
        label_stats, multilabel_stats, val_predictions, y_val,
        test_predictions, y_test_true, test_binary_preds
    )

    # 保存所有指标到JSON文件
    save_metrics_to_json(val_metrics, test_metrics, label_stats, multilabel_stats, history)

    # 生成预测
    print("\n生成提交文件...")
    submission = make_predictions(trained_model, X_test, test_df, target_columns)

    # 保存结果
    submission.to_csv(os.path.join(tables_dir, 'improved_rnn_submission.csv'), index=False)
    print("提交文件已保存!")

    # 保存模型
    trained_model.save(os.path.join(models_dir, 'toxic_comment_model'))
    print("模型已保存!")

    # 打印最终目录结构
    print("\n=== 生成的文件结构 ===")
    print(f"图片目录 ({images_dir}):")
    for file in os.listdir(images_dir):
        print(f"  - {file}")

    print(f"\n表格目录 ({tables_dir}):")
    for file in os.listdir(tables_dir):
        print(f"  - {file}")

    print(f"\n模型目录 ({models_dir}):")
    for file in os.listdir(models_dir):
        print(f"  - {file}")

    return trained_model, history, val_metrics, test_metrics, valid_samples


# 运行主函数
if __name__ == "__main__":
    model, history, val_metrics, test_metrics, valid_samples = main()
