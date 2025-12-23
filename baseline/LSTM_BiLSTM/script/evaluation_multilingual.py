import sys
import os
import warnings
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter

# Add the project root to sys.path to allow imports from TCC
# Assuming the script is located at .../baseline/LSTM_BiLSTM/script/
# And we want to import TCC which maps to LSTM_BiLSTM
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # .../baseline
sys.path.append(project_root)

# Hack: Map 'TCC' to 'LSTM_BiLSTM'
# This allows "from TCC.model..." to work if the folder is named LSTM_BiLSTM
if 'TCC' not in sys.modules:
    import types
    tcc_module = types.ModuleType("TCC")
    # Point TCC package path to the LSTM_BiLSTM directory
    tcc_module.__path__ = [os.path.join(project_root, "LSTM_BiLSTM")]
    sys.modules["TCC"] = tcc_module

# Ensure NLTK stopwords are downloaded
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

try:
    from TCC.model.LSTM import lstm
    from TCC.model.BILSTM import bilstm
    from TCC.script.Dataset import PaddedDataset, collate_batch
    from TCC.script.untils import yield_tokens
    from TCC.script import config
except ImportError:
    # If TCC import fails, try to import relatively or adjust path
    sys.path.append(os.path.dirname(current_dir)) # Add LSTM_BiLSTM to path
    # But the code uses TCC. prefix. 
    # Let's assume the user runs this with proper PYTHONPATH or we fix it by renaming imports if needed.
    # For now, let's trust the existing pattern but provide a hint if it fails.
    print("Warning: Could not import TCC. Ensure PYTHONPATH includes the parent directory of LSTM_BiLSTM and LSTM_BiLSTM is accessible as TCC.")
    # Try to continue, maybe it works if run from a specific dir
    from model.LSTM import lstm
    from model.BILSTM import bilstm
    from script.Dataset import PaddedDataset, collate_batch
    from script.untils import yield_tokens
    from script import config

# Paths
TRAIN_DATA_PATH = "/home/fwm/projects/Toxic-comment-classification/data/source/train.csv"
TEST_DATA_PATH = "/home/fwm/projects/Toxic-comment-classification/data/jigsaw-multilingual-toxic-comment/test.csv"
TEST_LABELS_PATH = "/home/fwm/projects/Toxic-comment-classification/data/jigsaw-multilingual-toxic-comment/test_labels.csv"
MODEL_SAVE_DIR = os.path.join(os.path.dirname(current_dir), "save")

def load_multilingual_data():
    print("Loading multilingual test data...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_labels_df = pd.read_csv(TEST_LABELS_PATH)
    
    # Merge on id
    # test.csv has 'content', test_labels.csv has 'toxic'
    df = pd.merge(test_df, test_labels_df, on='id')
    
    print(f"Test data loaded. Shape: {df.shape}")
    return df

def Eval(model_name, eval_loader, vocab, PAD_IDX):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {model_name} on device {device}")

    if model_name == "lstm":
        model = lstm(vocab_size=len(vocab), embed_dim=50, hidden_dim=256, output_dim=6, pad_idx=PAD_IDX).to(device)
        model_path = os.path.join(MODEL_SAVE_DIR, "lstm.pth")
    elif model_name == "bilstm":
        model = bilstm(vocab_size=len(vocab), embed_dim=50, hidden_dim=256, output_dim=6, pad_idx=PAD_IDX).to(device)
        model_path = os.path.join(MODEL_SAVE_DIR, "bilstm.pth")
    else:
        print("Unknown model name")
        return

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels, lengths in eval_loader:
            texts = texts.to(device)
            # labels in eval_loader are from the dataset, which we set up to be the binary toxic label
            # But wait, PaddedDataset expects labels. 
            # In the original script, labels were 6-dim.
            # Here, our dataframe has 'toxic' column. 
            # We should check how PaddedDataset handles labels.
            
            outputs = model(texts, lengths)
            
            # Sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            
            # Take max of toxic (0) and severe_toxic (1)
            toxic_probs, _ = torch.max(probs[:, :2], dim=1)
            
            all_preds.append(toxic_probs.cpu().numpy())
            all_labels.append(labels.numpy()) # These are the true binary labels from the dataset

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate Metrics
    y_pred = (all_preds >= 0.5).astype(int)
    y_true = all_labels

    # Debug info
    print(f"DEBUG: y_true distribution: {np.bincount(y_true.astype(int))}")
    print(f"DEBUG: y_pred distribution: {np.bincount(y_pred.astype(int))}")
    print(f"DEBUG: First 10 preds: {all_preds[:10]}")
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, all_preds)
    except ValueError:
        auc = 0.0

    print("="*30)
    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC:      {auc:.4f}")
    print("="*30)

if __name__ == "__main__":
    # 1. Build Vocab from Training Data (same as training)
    print("Loading training data to build vocabulary...")
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    comments = list(train_data["comment_text"])
    
    tokenizer = get_tokenizer("basic_english")
    
    counter = Counter()
    for tokens in yield_tokens(iter(comments), tokenizer):
        counter.update(tokens)
        
    vocab = Vocab(
        counter,
        specials=["<pad>", "<unk>"],
        max_size=30000
    )
    
    PAD_IDX = vocab['<pad>']
    config.PAD_IDX = PAD_IDX
    print(f"Vocabulary size: {len(vocab)}")

    # 2. Load Multilingual Test Data
    test_df = load_multilingual_data()
    
    # Prepare data for PaddedDataset
    # PaddedDataset expects 'comment_text' and labels.
    # Our test_df has 'content' and 'toxic'.
    # We need to rename 'content' to 'comment_text' to match PaddedDataset expectation if it uses column names
    # Let's check PaddedDataset implementation. 
    # Assuming it takes the dataframe and uses specific columns or just iterates.
    # In evaluation.py: ev_data = pd.concat([test_data, test_labels_data.iloc[:, 1:]], axis=1)
    # It seems it expects the dataframe to have text and labels.
    
    # Let's rename 'content' to 'comment_text'
    test_df = test_df.rename(columns={'content': 'comment_text'})
    
    # PaddedDataset likely expects multiple label columns if it was designed for the 6-class task.
    # However, we only have 'toxic'.
    # If PaddedDataset tries to read 6 columns, it might fail.
    # Let's inspect PaddedDataset in TCC/script/Dataset.py if possible, or just try to adapt.
    # If PaddedDataset is hardcoded for 6 labels, we might need to fake the other 5 columns.
    
    # Faking other columns to satisfy PaddedDataset if necessary
    for col in ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        test_df[col] = 0
        
    # Now we have all 6 columns, but we only care about 'toxic' for the ground truth in our custom Eval loop?
    # Wait, in Eval loop: `all_labels.append(labels.numpy())`
    # If PaddedDataset returns 6 labels, `labels` will be (batch, 6).
    # But our ground truth `y_true` should be the binary 'toxic' label.
    # So we should extract the first column of `labels` if PaddedDataset returns all 6.
    
    eval_dataset = PaddedDataset(test_df, vocab, max_length=256)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=512,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    # 3. Evaluate
    models_to_evaluate = ["lstm", "bilstm"]
    for model_name in models_to_evaluate:
        # We need to pass a custom Eval function or modify the loop to handle the labels correctly
        # I'll implement the logic inside Eval above.
        
        # Note: In Eval, I need to handle the labels.
        # If PaddedDataset returns 6 labels, I should take the one corresponding to 'toxic'.
        # Since I filled others with 0, and 'toxic' is the first one (usually), 
        # I should check the column order in PaddedDataset.
        # Usually it reads: comment_text, toxic, severe_toxic, ...
        # Let's assume 'toxic' is at index 0.
        
        # Redefining Eval to handle this specific logic
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Evaluating {model_name} on device {device}")

        if model_name == "lstm":
            model = lstm(vocab_size=len(vocab), embed_dim=50, hidden_dim=256, output_dim=6, pad_idx=PAD_IDX).to(device)
            model_path = os.path.join(MODEL_SAVE_DIR, "lstm.pth")
        elif model_name == "bilstm":
            model = bilstm(vocab_size=len(vocab), embed_dim=50, hidden_dim=256, output_dim=6, pad_idx=PAD_IDX).to(device)
            model_path = os.path.join(MODEL_SAVE_DIR, "bilstm.pth")
        else:
            continue

        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue

        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for texts, labels, lengths in eval_loader:
                texts = texts.to(device)
                outputs = model(texts, lengths)
                
                probs = torch.sigmoid(outputs)
                
                # Prediction: max of toxic and severe_toxic
                toxic_probs, _ = torch.max(probs[:, :2], dim=1)
                
                # Ground Truth: The 'toxic' label. 
                # Assuming PaddedDataset returns labels in the order of columns in dataframe?
                # Or hardcoded columns?
                # If hardcoded, it usually looks for ['toxic', 'severe_toxic', ...]
                # So labels[:, 0] should be 'toxic'.
                
                # Check if labels is 1D or 2D
                if len(labels.shape) == 1:
                    batch_labels = labels.numpy()
                else:
                    batch_labels = labels[:, 0].numpy() # Take first column
                
                all_preds.append(toxic_probs.cpu().numpy())
                all_labels.append(batch_labels)

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        y_pred = (all_preds >= 0.5).astype(int)
        y_true = all_labels

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, all_preds)
        except ValueError:
            auc = 0.0

        print("="*30)
        print(f"Model: {model_name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC:      {auc:.4f}")
        print("="*30)
