import warnings
from sklearn.metrics import (
    f1_score, 
    roc_auc_score, 
    accuracy_score,
)
from transformers import EvalPrediction
import torch

def multi_label_metrics(predictions, labels, threshold=0.5):
    """
    计算多标签分类的必要评估指标
    
    Args:
        predictions: 模型输出的 logits (未经过 sigmoid)
        labels: 真实标签 (0/1)
        threshold: 分类阈值，默认 0.5
    
    Returns:
        dict: 包含 subset_accuracy, f1_macro, auc_macro 的字典
    """
    # 应用 sigmoid 转换为概率
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions)).numpy()
    
    # 使用阈值转换为二分类预测
    y_pred = (probs >= threshold).astype(int)
    y_true = labels
    
    # Subset Accuracy: 所有标签都必须完全正确（严格）
    subset_accuracy = accuracy_score(y_true, y_pred)
    
    # Macro F1 Score
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Macro AUC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            auc_macro = roc_auc_score(y_true, probs, average='macro')
        except (ValueError, RuntimeWarning):
            auc_macro = 0.0
    
    return {
        'subset_accuracy': subset_accuracy,
        'f1_macro': f1_macro,
        'auc_macro': auc_macro,
    }

def compute_metrics_simple(p: EvalPrediction):
        """
        Metrics for validation and test (acc, f1_macro, auc_macro)
        """
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(predictions=preds, labels=p.label_ids)
        return {
            'acc': result['subset_accuracy'],
            'f1_macro': result['f1_macro'],
            'auc_macro': result['auc_macro']
        }

def compute_metrics_detailed(p: EvalPrediction):
        """
        Detailed metrics for testing (same as simple for now)
        """
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(predictions=preds, labels=p.label_ids)
        return {
            'acc': result['subset_accuracy'],
            'f1_macro': result['f1_macro'],
            'auc_macro': result['auc_macro']
        }
