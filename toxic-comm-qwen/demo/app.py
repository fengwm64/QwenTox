import torch
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# Translations
# ----------------------
TRANSLATIONS = {
    "English": {
        "title": "# 🛡️ QwenTox Interactive Demo\nMulti-label toxic comment detection powered by Qwen-0.6B + LoRA.",
        "input_label": "Input Comments (one per line)",
        "input_placeholder": "Enter comments here...\nExample:\nThis is stupid\nI will kill you",
        "input_info": "Support batch inference.",
        "threshold_label": "Toxicity Threshold",
        "threshold_info": "Probability threshold to classify as toxic.",
        "btn_label": "🚀 Analyze",
        "model_info": "### Model Info\n- **Base:** Qwen/Qwen3-0.6B-Base\n- **Adapter:** yingfeng64/QwenTox",
        "plot_probs": "Probability Bar Chart",
        "table_label": "Batch Predictions",
        "plot_prob_title": "Toxicity Prediction",
        "plot_prob_ylabel": "Probability",
        "threshold_legend": "Threshold",
        "sample_col": "Sample Text"
    },
    "中文": {
        "title": "# 🛡️ QwenTox 交互式演示\n基于 Qwen-0.6B + LoRA 的多标签毒性评论检测。",
        "input_label": "输入评论（每行一条）",
        "input_placeholder": "在此输入评论...\n示例：\nThis is stupid\nI will kill you",
        "input_info": "支持批量推理。",
        "threshold_label": "毒性阈值",
        "threshold_info": "判定为毒性的概率阈值。",
        "btn_label": "🚀 开始分析",
        "model_info": "### 模型信息\n- **基座:** Qwen/Qwen3-0.6B-Base\n- **适配器:** yingfeng64/QwenTox",
        "plot_probs": "概率条形图",
        "table_label": "批量预测结果",
        "plot_prob_title": "Toxicity Prediction",
        "plot_prob_ylabel": "Probability",
        "threshold_legend": "Threshold",
        "sample_col": "样本内容"
    }
}

# ----------------------
# Load model
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B-Base",
    trust_remote_code=True
)

if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen3-0.6B-Base",
    num_labels=len(LABELS),
    problem_type="multi_label_classification",
    trust_remote_code=True
)
base_model.config.pad_token_id = tokenizer.pad_token_id

model = PeftModel.from_pretrained(base_model, "yingfeng64/QwenTox")
model.to(DEVICE)
model.eval()

# ----------------------
# Inference
# ----------------------
@torch.no_grad()
def predict_batch(text_block, threshold, language="English"):
    texts = [t.strip() for t in text_block.split("\n") if t.strip()]
    if len(texts) == 0:
        return None, None

    t = TRANSLATIONS[language]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=384
    ).to(DEVICE)

    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits)

    # Single Sample -> Plot
    if len(texts) == 1:
        # -------- Bar plot --------
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [
            "red" if p >= threshold else "green"
            for p in probs[0].cpu().tolist()
        ]
        bars = ax.bar(LABELS, probs[0].cpu().tolist(), color=colors)
        ax.axhline(threshold, linestyle="--", color="gray", label=f"{t['threshold_legend']} {threshold}")
        ax.set_ylim(0, 1.1) # Slightly higher for annotations
        ax.set_ylabel(t["plot_prob_ylabel"])
        ax.set_title(t["plot_prob_title"])
        ax.legend()
        
        # Grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Value Annotations
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}',
                    ha='center', va='bottom')

        plt.xticks(rotation=30)
        # Adjust layout to be lower (top margin) and fit labels
        plt.subplots_adjust(top=0.85, bottom=0.2)
        
        return (
            gr.update(visible=False), # Table hidden
            gr.update(value=fig, visible=True) # Plot visible
        )
    
    # Batch -> Table
    else:
        # -------- Table (Wide Format) --------
        rows = []
        for i, text in enumerate(texts):
            row = {t["sample_col"]: text}
            for j, label in enumerate(LABELS):
                row[label] = round(probs[i, j].item(), 4)
            rows.append(row)
        df = pd.DataFrame(rows)
        
        return (
            gr.update(value=df, visible=True), # Table visible
            gr.update(visible=False) # Plot hidden
        )

# ----------------------
# Gradio UI (Blocks)
# ----------------------
def change_language(lang):
    t = TRANSLATIONS[lang]
    return (
        gr.update(value=t["title"]),
        gr.update(label=t["input_label"], placeholder=t["input_placeholder"], info=t["input_info"]),
        gr.update(label=t["threshold_label"], info=t["threshold_info"]),
        gr.update(value=t["btn_label"]),
        gr.update(value=t["model_info"]),
        gr.update(label=t["plot_probs"]),
        gr.update(label=t["table_label"])
    )

with gr.Blocks(
    title="QwenTox Interactive Demo",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        padding-top: 40px !important;
    }
    """
) as demo:
    
    with gr.Row():
        with gr.Column(scale=4):
            title_md = gr.Markdown(TRANSLATIONS["English"]["title"])
        with gr.Column(scale=1):
            language_selector = gr.Radio(
                choices=["English", "中文"], 
                value="English", 
                interactive=True,
                label=None,
                show_label=False,
            )

    with gr.Row():
        # Left Column
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                lines=8,
                label=TRANSLATIONS["English"]["input_label"],
                placeholder=TRANSLATIONS["English"]["input_placeholder"],
                info=TRANSLATIONS["English"]["input_info"]
            )
            threshold_slider = gr.Slider(
                minimum=0.1, 
                maximum=0.9, 
                value=0.5, 
                step=0.05, 
                label=TRANSLATIONS["English"]["threshold_label"],
                info=TRANSLATIONS["English"]["threshold_info"]
            )
            submit_btn = gr.Button(TRANSLATIONS["English"]["btn_label"], variant="primary")
            
            model_info_md = gr.Markdown(TRANSLATIONS["English"]["model_info"])

        # Right Column
        with gr.Column(scale=2):
            # Output components (initially visible/hidden state doesn't matter much as function updates them, 
            # but let's keep them visible=False initially or just empty)
            output_df = gr.Dataframe(
                label=TRANSLATIONS["English"]["table_label"],
                interactive=False,
                visible=False
            )
            plot_probs = gr.Plot(
                label=TRANSLATIONS["English"]["plot_probs"],
                visible=False
            )

    # Event handling
    language_selector.change(
        fn=change_language,
        inputs=language_selector,
        outputs=[
            title_md, input_text, threshold_slider, submit_btn, model_info_md,
            plot_probs, output_df
        ]
    )

    submit_btn.click(
        fn=predict_batch,
        inputs=[input_text, threshold_slider, language_selector],
        outputs=[output_df, plot_probs]
    )

if __name__ == "__main__":
    demo.launch()