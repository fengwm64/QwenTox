import os
import sys
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Add parent directory to path to import dataset and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import ToxicDataset

class FewShotClassifier:
    def __init__(self, model_path, examples, device="cuda", use_api=False, api_base=None, api_key=None, model_name=None, num_workers=16):
        self.model_path = model_path
        self.device = device
        self.examples = examples
        self.use_api = use_api
        self.num_workers = num_workers
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

        if self.use_api:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Please install openai package: pip install openai")
            
            print(f"Initializing OpenAI API client with base_url={api_base}, model={model_name}")
            self.client = OpenAI(api_key=api_key, base_url=api_base)
            self.model_name = model_name
        else:
            print(f"Loading model from {model_path}...")
            try:
                self.llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.3)
                self.tokenizer = self.llm.get_tokenizer()
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                print("Trying to load Qwen/Qwen3-0.6B-Instruct from HuggingFace...")
                self.model_path = "Qwen/Qwen3-0.6B-Instruct"
                self.llm = LLM(model=self.model_path, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.9)
                self.tokenizer = self.llm.get_tokenizer()
            
            self.sampling_params = SamplingParams(
                temperature=0.6,
                top_k=20,
                top_p=0.95,
                max_tokens=50
            )

    def format_example(self, example):
        comment = example['comment_text']
        labels = []
        for label in self.labels:
            if example[label] == 1:
                labels.append(label)
        
        label_str = ", ".join(labels) if labels else "none"
        return f"Comment: {comment}\nClassification: {label_str}"

    def _construct_messages(self, comment):
        # Construct the few-shot examples string
        examples_str = "\n\n".join([self.format_example(ex) for ex in self.examples])
        
        prompt = (
            "Analyze the following comment and classify it into one or more of the following categories:\n"
            "- toxic: General rude, disrespectful, or unreasonable comment.\n"
            "- severe_toxic: Extremely hateful, aggressive, or malicious comment.\n"
            "- obscene: Contains vulgar, offensive, or sexually explicit language.\n"
            "- threat: Expresses an intention to cause harm, violence, or damage.\n"
            "- insult: Disrespectful or scornful remark intended to offend.\n"
            "- identity_hate: Hateful speech targeting a specific group based on race, religion, gender, etc.\n\n"
            "Instructions:\n"
            "1. If the comment fits multiple categories, list them all separated by commas.\n"
            "2. If the comment does not fit any of these categories (i.e., it is safe), output 'none'.\n"
            "3. Output ONLY the category names or 'none'. Do not provide explanations.\n\n"
            "Examples:\n"
            f"{examples_str}\n\n"
            f"Comment: {comment}\n"
            "Classification:"
        )
        
        messages = [
            {"role": "system", "content": "You are an expert content moderator specializing in detecting toxic behavior in online comments."},
            {"role": "user", "content": prompt}
        ]
        return messages

    def generate_prompt(self, comment):
        messages = self._construct_messages(comment)
        try:
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except:
            text = messages[1]['content']
            
        return text

    def predict(self, texts, batch_size=8):
        all_preds = []
        
        if self.use_api:
            from concurrent.futures import ThreadPoolExecutor
            
            def process_single(text):
                messages = self._construct_messages(text)
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.6,
                        max_tokens=50
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"API Error: {e}")
                    return "none"

            print(f"Running API inference with {self.num_workers} workers...")
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                all_preds = list(tqdm(executor.map(process_single, texts), total=len(texts), desc="API Inference"))
        else:
            # Generate all prompts
            prompts = [self.generate_prompt(text) for text in texts]
            
            # Run vLLM inference
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            # Extract generated text
            for output in outputs:
                generated_text = output.outputs[0].text
                all_preds.append(generated_text.strip())
                
        return all_preds

    def parse_predictions(self, raw_preds):
        parsed_preds = []
        for pred in raw_preds:
            pred_lower = pred.lower()
            row = [0] * len(self.labels)
            
            # Check for 'none' or safe
            if 'none' in pred_lower and len(pred_lower) < 20: 
                parsed_preds.append(row)
                continue
            
            for idx, label in enumerate(self.labels):
                # Simple string matching
                if label in pred_lower:
                    row[idx] = 1
            
            parsed_preds.append(row)
        return np.array(parsed_preds)

    def evaluate(self, dataset, sample_size=None):
        if sample_size:
            dataset = dataset.select(range(sample_size))
            
        texts = dataset['comment_text']
        true_labels = np.array([
            [sample[label] for label in self.labels] 
            for sample in dataset
        ])
        
        raw_preds = self.predict(texts)
        pred_labels = self.parse_predictions(raw_preds)
        
        # Calculate metrics
        acc = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
            
        return {
            "test_acc": acc,
            "test_f1_macro": f1
        }, raw_preds

def select_examples(dataset, labels, num_examples=8, seed=42):
    print(f"Selecting {num_examples} examples from dataset (size: {len(dataset)})...")
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    selected_indices = set()
    selected_examples = []
    
    def add_example(idx):
        if idx not in selected_indices:
            selected_indices.add(idx)
            selected_examples.append(dataset[idx])
            return True
        return False

    # 1. Pick one clean example
    for idx in indices:
        ex = dataset[idx]
        if all(ex[l] == 0 for l in labels):
            if add_example(idx):
                break
    
    # 2. Ensure coverage for each label
    for label in labels:
        # Check if already covered
        is_covered = any(ex[label] == 1 for ex in selected_examples)
        if not is_covered:
            for idx in indices:
                if dataset[idx][label] == 1 and idx not in selected_indices:
                    add_example(idx)
                    break
    
    # 3. Fill up to num_examples with diverse toxic examples
    # Cycle through labels to pick examples that have that label to ensure diversity
    # Prioritize rare labels
    priority_labels = ['severe_toxic', 'threat', 'identity_hate', 'obscene', 'insult', 'toxic']
    
    current_label_idx = 0
    max_attempts = len(priority_labels) * 5 # Avoid infinite loops
    attempts = 0
    
    while len(selected_examples) < num_examples and attempts < max_attempts:
        target_label = priority_labels[current_label_idx % len(priority_labels)]
        current_label_idx += 1
        attempts += 1
        
        # Find an example with this target label
        for idx in indices:
            if idx not in selected_indices and dataset[idx][target_label] == 1:
                add_example(idx)
                break

    # 4. If still not enough, fill with any (likely clean ones)
    for idx in indices:
        if len(selected_examples) >= num_examples:
            break
        if idx not in selected_indices:
            add_example(idx)
            
    # Verify coverage
    print("\nSelected Few-Shot Examples:")
    covered_labels = set()
    for i, ex in enumerate(selected_examples):
        l = [l for l in labels if ex[l] == 1]
        if l:
            covered_labels.update(l)
        print(f"{i+1}. Labels: {l if l else ['none']}")
        # print(f"   Text: {ex['comment_text'][:100]}...")
        
    missing = set(labels) - covered_labels
    if missing:
        print(f"Warning: Missing labels in few-shot examples: {missing}")
    else:
        print("All labels covered.")
        
    return selected_examples

def main():
    parser = argparse.ArgumentParser(description="Few-shot Toxic Comment Classification")
    parser.add_argument("--model_path", type=str, default="/home/fwm/projects/Toxic-comment-classification/models/Qwen3-0.6B-Instruct", help="Path to the model")
    parser.add_argument("--data_dir", type=str, default="../../data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="../results/few-shot", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to test (for debugging)")
    parser.add_argument("--num_examples", type=int, default=8, help="Number of few-shot examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for example selection")
    
    # API arguments
    parser.add_argument("--use_api", action="store_true", help="Use OpenAI API for inference")
    parser.add_argument("--api_base", type=str, default="https://api.openai.com/v1", help="OpenAI API base URL")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="Model name for API")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of threads for API inference")
    
    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Dataset
    print(f"Loading data from {args.data_dir}...")
    try:
        dataset_loader = ToxicDataset(args.data_dir)
        dataset = dataset_loader.load_data()
        test_dataset = dataset['test']
        # Use train dataset for selecting examples to avoid leakage
        train_dataset = dataset['train']
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Select Examples
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    examples = select_examples(train_dataset, labels, num_examples=args.num_examples, seed=args.seed)

    # Initialize Classifier
    classifier = FewShotClassifier(
        args.model_path, 
        examples, 
        device=device,
        use_api=args.use_api,
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model_name,
        num_workers=args.num_workers
    )
    
    # Evaluate
    print("Starting evaluation...")
    metrics, raw_preds = classifier.evaluate(test_dataset, args.sample_size)
    
    print("\nFew-shot Results:")
    print(json.dumps(metrics, indent=4))
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
