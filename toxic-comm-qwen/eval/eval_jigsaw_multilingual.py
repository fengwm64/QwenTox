import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings

# Set CUDA devices to 0 and 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Suppress warnings
warnings.filterwarnings("ignore")

# Paths
BASE_MODEL_PATH = "/home/fwm/projects/Toxic-comment-classification/models/ar_models/qwen/Qwen3-0.6B-Base"
ADAPTER_PATH = "/home/fwm/projects/Toxic-comment-classification/toxic-comm-qwen/results/Qwen0.6B-lora-final/checkpoint-1683"
TEST_DATA_PATH = "/home/fwm/projects/Toxic-comment-classification/data/jigsaw-multilingual-toxic-comment/test.csv"
TEST_LABELS_PATH = "/home/fwm/projects/Toxic-comment-classification/data/jigsaw-multilingual-toxic-comment/test_labels.csv"

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
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_labels_df = pd.read_csv(TEST_LABELS_PATH)
    
    # Merge on id
    # test.csv has 'content', test_labels.csv has 'toxic'
    df = pd.merge(test_df, test_labels_df, on='id')
    
    # Check columns
    print(f"Columns: {df.columns}")
    
    print(f"Test data loaded. Shape: {df.shape}")
    return df

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    df = load_test_data()
    texts = df['content'].tolist() # Column name is 'content' in multilingual dataset
    labels = df['toxic'].values # Binary label

    # Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    print("Loading model...")
    # Load base model
    # Note: The model was trained with 6 labels. We must load it with num_labels=6 to match the checkpoint.
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
            
            # Take the maximum probability of 'toxic' (index 0) and 'severe_toxic' (index 1)
            toxic_probs, _ = torch.max(probs[:, :2], dim=1)
            
            all_preds.append(toxic_probs.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate Metrics
    print("Calculating metrics...")
    
    # Threshold 0.5
    y_pred = (all_preds >= 0.5).astype(int)
    y_true = all_labels

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # F1 Score
    f1 = f1_score(y_true, y_pred)
    
    # AUC
    try:
        auc = roc_auc_score(y_true, all_preds)
    except ValueError:
        auc = 0.0

    print("="*30)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC:      {auc:.4f}")
    print("="*30)

    # Save results to txt
    with open("evaluation_multilingual_results.txt", "w") as f:
        f.write("="*30 + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC:      {auc:.4f}\n")
        f.write("="*30 + "\n")
    print("Results saved to evaluation_multilingual_results.txt")

    # Save predictions to CSV
    print("Saving predictions to evaluation_multilingual_predictions.csv...")
    output_df = df.copy()
    output_df['pred_toxic'] = all_preds
    
    output_df.to_csv("evaluation_multilingual_predictions.csv", index=False)
    print("Predictions saved to evaluation_multilingual_predictions.csv")

if __name__ == "__main__":
    main()
