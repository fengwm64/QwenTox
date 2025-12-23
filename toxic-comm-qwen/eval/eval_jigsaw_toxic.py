import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, hamming_loss
import warnings

# Set CUDA devices to 0 and 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Suppress warnings
warnings.filterwarnings("ignore")

# Paths
BASE_MODEL_PATH = "/home/fwm/projects/Toxic-comment-classification/models/ar_models/qwen/Qwen3-0.6B-Base"
ADAPTER_PATH = "/home/fwm/projects/Toxic-comment-classification/toxic-comm-qwen/results/Qwen0.6B-lora-final/checkpoint-1683"
TEST_DATA_PATH = "/home/fwm/projects/Toxic-comment-classification/data/jigsaw-toxic-comment/test.csv"
# TEST_LABELS_PATH = "/home/fwm/projects/Toxic-comment-classification/data/source/test_labels.csv"

# Labels
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

class ToxicTestDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def load_test_data():
    print("Loading test data...")
    df = pd.read_csv(TEST_DATA_PATH)
    
    # Filter out -1 labels
    # If any label is -1, we drop the row. Usually all labels are -1 or none are.
    if 'toxic' in df.columns:
        df = df[df['toxic'] != -1]
    
    print(f"Test data loaded. Shape: {df.shape}")
    return df

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    df = load_test_data()
    texts = df['comment_text'].tolist()
    labels = df[LABEL_COLS].values

    # Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    print("Loading model...")
    # Load base model
    config = AutoConfig.from_pretrained(BASE_MODEL_PATH, num_labels=6, problem_type="multi_label_classification", trust_remote_code=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        config=config,
        trust_remote_code=True
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(device)
    model.eval()

    # Create DataLoader
    dataset = ToxicTestDataset(texts, labels, tokenizer, max_len=128)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Inference
    print("Running inference...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Sigmoid
            probs = torch.sigmoid(logits)
            
            all_preds.append(probs.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate Metrics
    print("Calculating metrics...")
    
    # Threshold 0.5
    y_pred = (all_preds >= 0.5).astype(int)
    y_true = all_labels

    # Subset Accuracy
    subacc = accuracy_score(y_true, y_pred)
    
    # Hamming Loss
    h_loss = hamming_loss(y_true, y_pred)
    
    # Macro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Macro AUC
    try:
        macro_auc = roc_auc_score(y_true, all_preds, average='macro')
    except ValueError:
        macro_auc = 0.0

    print("="*30)
    print(f"Subset Accuracy: {subacc:.4f}")
    print(f"Hamming Loss:    {h_loss:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Macro AUC:       {macro_auc:.4f}")
    print("="*30)

    # Save results to txt
    with open("evaluation_results.txt", "w") as f:
        f.write("="*30 + "\n")
        f.write(f"Subset Accuracy: {subacc:.4f}\n")
        f.write(f"Hamming Loss:    {h_loss:.4f}\n")
        f.write(f"Macro F1:        {macro_f1:.4f}\n")
        f.write(f"Macro AUC:       {macro_auc:.4f}\n")
        f.write("="*30 + "\n")
    print("Results saved to evaluation_results.txt")

    # Save predictions to CSV
    print("Saving predictions to evaluation_predictions.csv...")
    output_df = df.copy()
    for i, label in enumerate(LABEL_COLS):
        output_df[f'pred_{label}'] = all_preds[:, i]
    
    output_df.to_csv("evaluation_predictions.csv", index=False)
    print("Predictions saved to evaluation_predictions.csv")

if __name__ == "__main__":
    main()
