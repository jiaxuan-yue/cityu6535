# ...existing code...
import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
# 新增：transformers + peft 用于 LoRA
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model

# 直接设置数据文件的相对路径
DATA_FILE = "train.csv"  # 这里设置为你的CSV相对路径
# ...existing code...
def read_data(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".jsonl") or path.endswith(".json"):
        return pd.read_json(path, lines=True)
    raise ValueError("仅支持 CSV 或 JSONL/JSON 文件")
def truthy(v):
    if pd.isna(v):
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        return int(v) != 0
    s = str(v).strip().lower()
    return s in ("1","true","yes","y","t","1.0")

def infer_label(row):
    # 0 -> A 更好, 1 -> B 更好, 2 -> 一样好
    if truthy(row.get("winner_model_a", None)):
        return 0
    if truthy(row.get("winner_model_b", None)):
        return 1
    if truthy(row.get("winner_tie", None)):
        return 2
    # fallback check single winner column
    w = row.get("winner", None)
    if pd.notna(w):
        s = str(w).strip().lower()
        if s in ("a","model_a","model a","winner_model_a","winner a"):
            return 0
        if s in ("b","model_b","model b","winner_model_b","winner b"):
            return 1
        if s in ("tie","same","equal","both"):
            return 2
    raise ValueError(f"无法推断 label，行 id={row.get('id','')}")

def build_features(row):
    """构建用于模型训练的特征文本"""
    prompt = str(row.get("prompt",""))
    a = str(row.get("response_a",""))
    b = str(row.get("response_b",""))
    # 组合成一个特征字符串
    return f"Prompt: {prompt}\nA: {a}\nB: {b}"
def ollama_predict(prompt_text, model_name="llama3", timeout=60):
    """
    调用本地Ollama服务进行分类
    prompt_text: 构造好的特征文本（格式："Prompt: ...\nA: ...\nB: ..."）
    model_name: Ollama中加载的模型名称（如"llama3"、"gemma:7b"等）
    返回值: 0（A更好）、1（B更好）、2（平局）
    """
    # 构造分类指令（明确任务要求）
    system_prompt = (
        "你需要判断模型A和模型B的回答哪个更优。"
        "输入格式为：\nPrompt: 用户的问题\nA: 模型A的回答\nB: 模型B的回答\n"
        "请严格按照以下规则输出：\n"
        "- 如果A更好，输出'0'\n"
        "- 如果B更好，输出'1'\n"
        "- 如果两者相当，输出'2'\n"
        "只输出数字，不要添加任何额外内容。"
    )
    
    # 组合完整请求内容
    request_data = {
        "model": model_name,
        "prompt": f"{system_prompt}\n\n{prompt_text}",
        "stream": False,  # 非流式返回
        "timeout": timeout
    }
    
    try:
        # 发送POST请求到本地Ollama API
        response = requests.post(
            "http://localhost:11435/api/generate",
            json=request_data,
            timeout=timeout
        )
        response.raise_for_status()  # 检查请求是否成功
        result = response.json()
        output = result.get("response", "").strip()
        
        # 解析返回结果（确保是0/1/2）
        if output == "0":
            return 0
        elif output == "1":
            return 1
        elif output == "2":
            return 2
        else:
            # 若返回格式异常，记录并默认平局（可根据需求调整）
            print(f"Ollama返回格式异常: {output}，输入文本: {prompt_text[:100]}...")
            return 2
    except Exception as e:
        print(f"Ollama调用失败: {str(e)}，输入文本: {prompt_text[:100]}...")
        return 2  # 异常时默认平局
def train_model(train_texts, train_labels):
    """训练一个文本分类模型"""
    # 使用TF-IDF提取特征并结合逻辑回归
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000, multi_class='multinomial'))
    ])
    
    # 训练模型
    model.fit(train_texts, train_labels)
    return model

def evaluate_model(model, test_texts, test_labels):
    """评估模型并返回各种指标"""
    # 预测
    predictions = model.predict(test_texts)
    
    # 计算指标
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    report = classification_report(test_labels, predictions, digits=4)
    cm = confusion_matrix(test_labels, predictions)
    
    return {
        'predictions': predictions,
        'accuracy': accuracy,
        'f1': f1,
        'classification_report': report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, output_path):
    """绘制混淆矩阵并保存"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['A更好', 'B更好', '一样好'],
                yticklabels=['A更好', 'B更好', '一样好'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 新增：LoRA 训练函数
def train_with_lora(train_df, test_df, args, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token":"<pad>"})

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=3,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # 调整 embedding（如果我们新增了 pad token）
    model.resize_token_embeddings(len(tokenizer))

    # LoRA 配置（针对常见模型的 q_proj/v_proj，必要时可调整 target_modules）
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else ["q_proj","v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

    # 构造 HF datasets
    hf_train = HFDataset.from_pandas(train_df[["features","label"]].rename(columns={"features":"text"}).reset_index(drop=True))
    hf_val = HFDataset.from_pandas(test_df[["features","label"]].rename(columns={"features":"text"}).reset_index(drop=True))

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)

    hf_train = hf_train.map(tokenize_fn, batched=True)
    hf_val = hf_val.map(tokenize_fn, batched=True)
    hf_train = hf_train.remove_columns(["text"])
    hf_val = hf_val.remove_columns(["text"])
    hf_train = hf_train.rename_column("label", "labels")
    hf_val = hf_val.rename_column("label", "labels")
    hf_train.set_format(type="torch")
    hf_val.set_format(type="torch")

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, digits=4, output_dict=True)
        metrics = {"accuracy": acc}
        if "macro avg" in report:
            metrics.update({
                "precision_macro": report["macro avg"]["precision"],
                "recall_macro": report["macro avg"]["recall"],
                "f1_macro": report["macro avg"]["f1-score"],
            })
        return metrics

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_res = trainer.evaluate()
    # 保存 LoRA 权重（PEFT 会只保存 adapter）
    peft_dir = os.path.join(output_dir, "lora")
    model.save_pretrained(peft_dir)
    tokenizer.save_pretrained(output_dir)

    # 预测并返回报告
    preds_out = trainer.predict(hf_val)
    preds = np.argmax(preds_out.predictions, axis=1)
    labels = preds_out.label_ids
    report = classification_report(labels, preds, digits=4)
    cm = confusion_matrix(labels, preds)
    return {"eval": eval_res, "report": report, "confusion_matrix": cm, "peft_dir": peft_dir}

# ...existing code...

def main():
    
    # 修改参数解析器，增加 LoRA 相关参数
    parser = argparse.ArgumentParser()
    # LoRA 选项
    parser.add_argument("--use_lora", action="store_true", help="是否使用 LoRA 微调（基于 transformers + peft）")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="用于 LoRA 的预训练模型路径或 hub 名称")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="", help="逗号分隔的 target_modules（为空则使用 q_proj,v_proj）")
    parser.add_argument("--use_ollama", action="store_true", help="是否使用本地Ollama服务")
    parser.add_argument("--ollama_model", type=str, default="llama3", help="Ollama模型名称（如llama3、gemma:7b）")
    parser.add_argument("--output_dir", default="./output", help="结果输出目录")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取和准备数据（使用硬编码的文件路径）
    print(f"读取数据文件: {DATA_FILE}")
    df = read_data(DATA_FILE)
    df=df.head(1000)
    
    # 必需列检查
    required_cols = ("prompt", "response_a", "response_b")
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")
    
    # 推断真实标签和构建特征
    df["label"] = df.apply(infer_label, axis=1)
    df["features"] = df.apply(build_features, axis=1)
    
    # 划分 8:2 训练集和测试集（分层抽样）
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df["label"], 
        random_state=args.seed
    )
    if args.use_ollama:
        print(f"使用本地Ollama服务，模型: {args.ollama_model}")
        # 对测试集进行推理（Ollama无需训练，直接推理）
        test_texts = test_df["features"].tolist()
        test_labels = test_df["label"].tolist()
        
        # 调用Ollama预测（批量处理，可加延迟避免请求过于密集）
        predictions = []
        for i, text in enumerate(test_texts):
            if i % 10 == 0:
                print(f"已处理 {i}/{len(test_texts)} 条样本")
            pred = ollama_predict(text, model_name=args.ollama_model)
            predictions.append(pred)
            time.sleep(0.5)  # 延迟避免Ollama服务过载（根据模型性能调整）
        
        # 评估结果（复用原有评估函数）
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='weighted')
        report = classification_report(test_labels, predictions, digits=4)
        cm = confusion_matrix(test_labels, predictions)
        
        # 保存结果（复用原有保存逻辑）
        results_df = pd.DataFrame({
            "id": test_df.get("id", ""),
            "true_label": test_labels,
            "pred_label": predictions
        })
        results_csv = os.path.join(args.output_dir, "ollama_predictions.csv")
        results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
        
        cm_path = os.path.join(args.output_dir, "ollama_confusion_matrix.png")
        plot_confusion_matrix(cm, cm_path)
        
        # 保存评估摘要
        summary = {
            "total_samples": len(df),
            "test_samples": len(test_df),
            "accuracy": accuracy,
            "f1_score": f1,
            "classification_report": report
        }
        with open(os.path.join(args.output_dir, "ollama_evaluation_summary.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "total_samples": summary["total_samples"],
                "test_samples": summary["test_samples"],
                "accuracy": summary["accuracy"],
                "f1_score": summary["f1_score"]
            }, ensure_ascii=False, indent=2))
            f.write("\n\n分类报告:\n")
            f.write(summary["classification_report"])
        
        # 打印结果
        print(f"\n测试样本数: {summary['test_samples']}")
        print(f"准确率 (Accuracy): {summary['accuracy']:.4f}")
        print(f"F1分数: {summary['f1_score']:.4f}")
        print("\n分类报告:\n", summary["classification_report"])
        print(f"\n结果已保存到 {args.output_dir}")
        return
        # 如果启用 LoRA，则用 transformers+peft 做微调并输出评估
    if args.use_lora:
        if not args.model_name_or_path:
            raise ValueError("--use_lora 时必须指定 --model_name_or_path")
        print("使用 LoRA 进行微调...")
        lora_out = train_with_lora(train_df, test_df, args, args.output_dir)
        # 保存并打印结果摘要
        with open(os.path.join(args.output_dir, "lora_eval_report.txt"), "w", encoding="utf-8") as f:
            f.write("Eval metrics:\n")
            f.write(json.dumps(lora_out["eval"], default=str, ensure_ascii=False, indent=2))
            f.write("\n\nClassification Report:\n")
            f.write(lora_out["report"])
        cm_path = os.path.join(args.output_dir, "lora_confusion_matrix.png")
        plot_confusion_matrix(lora_out["confusion_matrix"], cm_path)
        print("LoRA 微调并评估完成，结果保存在:", args.output_dir)
        return

    # 否则使用原有的 TF-IDF + LogisticRegression 流程
    # 训练模型
    print("开始训练模型 (TF-IDF + LR) ...")
    model = train_model(train_df["features"], train_df["label"])
    print("模型训练完成!")
    
    # 评估模型
    print("开始评估模型...")
    eval_results = evaluate_model(model, test_df["features"], test_df["label"])
    print("模型评估完成!")
    
    # 保存预测结果
    results_df = pd.DataFrame({
        "id": test_df.get("id", ""),
        "true_label": test_df["label"],
        "pred_label": eval_results["predictions"]
    })
    results_csv = os.path.join(args.output_dir, "predictions.csv")
    results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    
    # 绘制并保存混淆矩阵
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(eval_results["confusion_matrix"], cm_path)
    
    # 保存评估摘要
    summary = {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "accuracy": float(eval_results["accuracy"]),
        "f1_score": float(eval_results["f1"]),
        "classification_report": eval_results["classification_report"]
    }
    
    with open(os.path.join(args.output_dir, "evaluation_summary.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "total_samples": summary["total_samples"],
            "train_samples": summary["train_samples"],
            "test_samples": summary["test_samples"],
            "accuracy": summary["accuracy"],
            "f1_score": summary["f1_score"]
        }, ensure_ascii=False, indent=2))
        f.write("\n\n分类报告:\n")
        f.write(summary["classification_report"])
    
    # 打印评估结果
    print(f"\n训练样本数: {summary['train_samples']}")
    print(f"测试样本数: {summary['test_samples']}")
    print(f"准确率 (Accuracy): {summary['accuracy']:.4f}")
    print(f"F1分数: {summary['f1_score']:.4f}")
    print("\n分类报告:\n", summary["classification_report"])
    print(f"\n结果已保存到 {args.output_dir}")
    print(f"混淆矩阵已保存到 {cm_path}")

if __name__ == "__main__":
    main()