import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Paths
BASE_DIR = "/home/fwm/projects/Toxic-comment-classification"
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data/source/train.csv")
SOURCE_TEST_DATA_PATH = os.path.join(BASE_DIR, "data/source/test.csv")
MULTI_TEST_DATA_PATH = os.path.join(BASE_DIR, "data/jigsaw-multilingual-toxic-comment/test.csv")
MULTI_TEST_LABELS_PATH = os.path.join(BASE_DIR, "data/jigsaw-multilingual-toxic-comment/test_labels.csv")
MODEL_PATH = os.path.join(BASE_DIR, "baseline/RNN_CNN/results_CNN/models/toxic_comment_cnn_model")

# Parameters (Must match training configuration in CNN.py)
MAX_FEATURES = 20000
MAX_LEN = 256

def load_tokenizer():
    print("Loading training data to build tokenizer...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    
    # In CNN.py, the tokenizer is fitted on both train and test texts.
    # We try to replicate this to ensure the vocabulary mapping is consistent.
    texts = list(train_df['comment_text'].fillna("unknown").values)
    
    if os.path.exists(SOURCE_TEST_DATA_PATH):
        print(f"Found source test data at {SOURCE_TEST_DATA_PATH}, including in tokenizer fit...")
        test_df = pd.read_csv(SOURCE_TEST_DATA_PATH)
        test_texts = list(test_df['comment_text'].fillna("unknown").values)
        texts += test_texts
    
    # Initialize Tokenizer with the same parameters as in CNN.py
    tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(texts)
    print(f"Tokenizer fitted. Vocab size: {len(tokenizer.word_index)}")
    return tokenizer

def load_multilingual_data():
    print("Loading multilingual test data...")
    test_df = pd.read_csv(MULTI_TEST_DATA_PATH)
    test_labels_df = pd.read_csv(MULTI_TEST_LABELS_PATH)
    
    # Merge on id
    # test.csv has 'content', test_labels.csv has 'toxic'
    df = pd.merge(test_df, test_labels_df, on='id')
    
    print(f"Test data loaded. Shape: {df.shape}")
    return df

def main():
    # GPU setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus}")
        except RuntimeError as e:
            print(e)
    else:
        print("Using CPU")

    # 1. Prepare Tokenizer
    tokenizer = load_tokenizer()

    # 2. Load and Preprocess Test Data
    df = load_multilingual_data()
    # 'content' is the text column in the multilingual dataset
    texts = df['content'].fillna("unknown").values 
    labels = df['toxic'].values

    print("Converting texts to sequences...")
    sequences = tokenizer.texts_to_sequences(texts)
    X_test = pad_sequences(sequences, maxlen=MAX_LEN)

    # 3. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure you have trained the CNN model and it is saved in the correct location.")
        return
    
    print(f"Loading model from {MODEL_PATH}")
    model = None
    inference_func = None
    
    # Strategy 1: Standard tf.keras.models.load_model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Standard tf.keras.models.load_model failed: {e}")
        
        # Strategy 2: Try tf_keras (Legacy Keras 2 support for TF 2.16+)
        try:
            print("Attempting to load using tf_keras (legacy Keras 2)...")
            import tf_keras
            model = tf_keras.models.load_model(MODEL_PATH)
        except ImportError:
            print("tf_keras module not found.")
        except Exception as e2:
            print(f"tf_keras.models.load_model failed: {e2}")

        # Strategy 3: keras.layers.TFSMLayer (Keras 3 way to load SavedModel)
        if model is None:
            try:
                print("Attempting to load using keras.layers.TFSMLayer...")
                import keras
                # Load as a layer
                tfsm_layer = keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')
                inference_func = tfsm_layer
            except Exception as e3:
                print(f"keras.layers.TFSMLayer failed: {e3}")
                
                # Strategy 4: tf.saved_model.load (Low-level fallback)
                try:
                    print("Attempting to load using tf.saved_model.load...")
                    loaded_sm = tf.saved_model.load(MODEL_PATH)
                    inference_func = loaded_sm.signatures["serving_default"]
                except Exception as e4:
                    print(f"tf.saved_model.load failed: {e4}")
                    return

    # 4. Inference
    print("Running inference...")
    if model is not None:
        # Check if model expects float input (some SavedModels do even for embedding)
        # But usually predict handles it.
        try:
            preds = model.predict(X_test, batch_size=512, verbose=1)
        except Exception as e:
            print(f"Model.predict failed: {e}. Trying to cast input to float32...")
            X_test_float = X_test.astype(np.float32)
            preds = model.predict(X_test_float, batch_size=512, verbose=1)
            
    elif inference_func is not None:
        # Manual batch inference
        batch_size = 512
        n_samples = X_test.shape[0]
        all_preds = []
        
        print(f"Total samples: {n_samples}")
        for i in range(0, n_samples, batch_size):
            batch = X_test[i:i+batch_size]
            # Convert to tensor. 
            # Try float32 first as SavedModels often expect it
            batch_tensor = tf.constant(batch, dtype=tf.float32)
            
            try:
                batch_out = inference_func(batch_tensor)
            except Exception:
                # Retry with int32 if float32 fails (though Embedding usually casts)
                batch_tensor = tf.constant(batch, dtype=tf.int32)
                batch_out = inference_func(batch_tensor)
            
            # batch_out is a dict. We assume the first value is the output we want.
            # Or look for key 'dense_X' or 'output_0'
            if isinstance(batch_out, dict):
                # Heuristic: take the first value
                output_key = list(batch_out.keys())[0]
                batch_pred = batch_out[output_key]
            else:
                batch_pred = batch_out
                
            all_preds.append(batch_pred.numpy())
            
            if i % (batch_size * 10) == 0:
                print(f"Processed {i}/{n_samples}")
                
        preds = np.concatenate(all_preds, axis=0)

    # 5. Metrics
    # preds shape: (N, 6)
    # We want binary classification for 'toxic'.
    # Logic: max(toxic, severe_toxic)
    # toxic is col 0, severe_toxic is col 1
    
    toxic_probs = np.max(preds[:, :2], axis=1)
    
    y_pred = (toxic_probs >= 0.5).astype(int)
    y_true = labels

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, toxic_probs)
    except ValueError:
        auc = 0.0

    print("="*30)
    print(f"Model: CNN")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC:      {auc:.4f}")
    print("="*30)
    
    # Debug info to check if we have the same issue as LSTM
    print(f"DEBUG: y_true distribution: {np.bincount(y_true.astype(int))}")
    print(f"DEBUG: y_pred distribution: {np.bincount(y_pred.astype(int))}")

if __name__ == "__main__":
    main()
