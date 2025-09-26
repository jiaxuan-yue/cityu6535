import argparse
import os
import random
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    LlamaTokenizer,
    LlamaForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report, accuracy_score
import torch

def read_data(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".jsonl") or path.endswith(".json"):
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError("仅支持 csv 或 jsonl/json 文件")
    return df

def truthy(v):
    if pd.isna(v):
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        return int(v) != 0
    s = str(v).strip().lower()
    return s in ("1","true","yes","y","t","1.0")

def infer_label(row):
    # 0 -> model A 更好, 1 -> model B 更好, 2 -> 一样好
    if truthy(row.get("winner_model_a", None)):
        return 0
    if truthy(row.get("winner_model_b", None)):
        return 1
    if truthy(row.get("winner_tie", None)):
        return 2
    # 退回检查可能存在的单列 winner，如 'winner' 内容为 'a'/'b'/'tie'
    w = row.get("winner", None)
    if pd.notna(w):
        s = str(w).strip().lower()
        if s in ("a","model_a","model a","winner_model_a","winner a"):
            return 0
        if s in ("b","model_b","model b","winner_model_b","winner b"):
            return 1
        if s in ("tie","same","equal","both"):
            return 2
    raise ValueError(f"无法推断 label，行 id={row.get('id', '')}")

def build_input_text(row):
    prompt = str(row.get("prompt",""))
    a = str(row.get("response_a",""))
    b = str(row.get("response_b",""))
    # 可按需修改提示格式
    text = f"Prompt: {prompt}\n\nA: {a}\n\nB: {b}\n\n问题：哪个模型的回答更符合提示？请选择 A / B / SAME，并只输出 A 或 B 或 SAME。"
    return text

def preprocess_and_tokenize(dataset, tokenizer, max_length):
    def _tok(batch):
        texts = batch["text"]
        return tokenizer(texts, truncation=True, padding=False, max_length=max_length)
    return dataset.map(_tok, batched=True, remove_columns=[c for c in dataset.column_names if c not in ("label","text")])

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True, help="CSV 或 JSONL 文件路径，列名示例: id,model_a,model_b,prompt,response_a,response_b,winner_model_a,winner_model_b,winner_tie")
    parser.add_argument("--model_name_or_path", required=True, help="预训练模型路径或 hub 名称（支持 SequenceClassification）")
    parser.add_argument("--output_dir", default="./output", help="输出目录")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = read_data(args.data_file)

    # 必要列检查
    for col in ("prompt","response_a","response_b"):
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")

    # 推断 label
    df["label"] = df.apply(infer_label, axis=1)

    # 构造文本用于分类
    df["text"] = df.apply(build_input_text, axis=1)

    # 按 8:2 划分（分层抽样）
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=args.seed)
    ds_train = Dataset.from_pandas(train_df.reset_index(drop=True))
    ds_val = Dataset.from_pandas(val_df.reset_index(drop=True))
    dataset = DatasetDict({"train": ds_train, "validation": ds_val})

    # tokenizer & model 兼容性尝试
    try:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token":"<pad>"})

    try:
        model = LlamaForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=3)
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=3)

    model.resize_token_embeddings(len(tokenizer))

    tokenized = preprocess_and_tokenize(dataset["train"], tokenizer, args.max_length)
    tokenized_val = preprocess_and_tokenize(dataset["validation"], tokenizer, args.max_length)
    # 把 label 加回去（datasets map 已移除其他列）
    tokenized = tokenized.add_column("labels", list(train_df["label"].astype(int)))
    tokenized_val = tokenized_val.add_column("labels", list(val_df["label"].astype(int)))

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print("Eval metrics:", eval_metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    preds_output = trainer.predict(tokenized_val)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids
    report = classification_report(labels, preds, digits=4)
    print("Classification Report:\n", report)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
# ...existing code...
import requests
# ...existing code...

def call_ollama_generate(prompt, model, url="http://127.0.0.1:11434", timeout=60):
    """
    调用本地 Ollama HTTP API 进行一次生成，返回字符串（生成文本）。
    注意：不同 Ollama 版本返回结构可能不同，这里尝试几种常见字段。
    """
    payload = {"model": model, "prompt": prompt, "max_tokens": 32}
    try:
        r = requests.post(f"{url}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"调用 Ollama 失败: {e}")

    try:
        j = r.json()
    except Exception:
        return r.text

    # 常见返回解析尝试
    if isinstance(j, dict):
        # 1) 可能含 'text' 或 'generated'
        if "text" in j and isinstance(j["text"], str):
            return j["text"]
        if "generated" in j and isinstance(j["generated"], str):
            return j["generated"]
        # 2) 或者含 results -> content 列表
        if "results" in j and isinstance(j["results"], list) and len(j["results"])>0:
            res0 = j["results"][0]
            if isinstance(res0, dict) and "content" in res0:
                # content 可能是 list of dicts with 'text'
                content = res0["content"]
                if isinstance(content, list):
                    texts = []
                    for c in content:
                        if isinstance(c, dict) and "text" in c:
                            texts.append(c["text"])
                    if texts:
                        return "".join(texts)
                elif isinstance(content, str):
                    return content
    # fallback to raw text
    return r.text

# ...existing code...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True, help="CSV 或 JSONL 文件路径，列名示例: id,model_a,model_b,prompt,response_a,response_b,winner_model_a,winner_model_b,winner_tie")
    parser.add_argument("--model_name_or_path", required=False, help="预训练模型路径或 hub 名称（支持 SequenceClassification）。如使用 Ollama 评估可不指定")
    parser.add_argument("--output_dir", default="./output", help="输出目录")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ollama", action="store_true", help="如果指定，则对验证集调用本地 Ollama 进行推理评估（不做 HF 训练）")
    parser.add_argument("--ollama_model", type=str, default=None, help="Ollama 上部署的模型名称（用于 --use_ollama）")
    parser.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434", help="Ollama 本地服务 URL，默认 http://127.0.0.1:11434")
    args = parser.parse_args()

    # ...existing code...
    df = read_data(args.data_file)

    # 必要列检查
    for col in ("prompt","response_a","response_b"):
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")

    # 推断 label
    df["label"] = df.apply(infer_label, axis=1)

    # 构造文本用于分类
    df["text"] = df.apply(build_input_text, axis=1)

    # 按 8:2 划分（分层抽样）
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=args.seed)

    # 如果用户选择使用 Ollama 做推理评估：直接对验证集调用 Ollama，期望模型输出 A / B / SAME（或类似）
    if args.use_ollama:
        if not args.ollama_model:
            raise ValueError("--use_ollama 时必须指定 --ollama_model")
        os.makedirs(args.output_dir, exist_ok=True)
        preds = []
        for i, row in val_df.reset_index(drop=True).iterrows():
            prompt_text = row["text"]
            gen = call_ollama_generate(prompt_text, args.ollama_model, url=args.ollama_url)
            # 简单清洗，取首个非空字符 A/B/S/SAME
            t = str(gen).strip().upper()
            # 提取 A / B / SAME 三类
            label = None
            if len(t)>0:
                # 以首个字母判定，A->0, B->1, S/T->2 (SAME/TIE)
                if t[0] == "A":
                    label = 0
                elif t[0] == "B":
                    label = 1
                elif t[0] in ("S","T"):  # SAME 或 TIE
                    label = 2
            if label is None:
                # 回退：查找字符串中是否包含关键词
                if "A" in t:
                    label = 0
                elif "B" in t:
                    label = 1
                elif "SAME" in t or "TIE" in t or "EQUAL" in t:
                    label = 2
                else:
                    # 未识别的输出，标记为 -1
                    label = -1
            preds.append(label)

        true_labels = list(val_df["label"].astype(int))
        # 过滤掉未识别的项
        valid_idx = [i for i,l in enumerate(preds) if l != -1]
        if len(valid_idx) == 0:
            raise RuntimeError("所有 Ollama 输出均未被识别为 A/B/SAME，请检查模型输出格式或调整解析规则。")
        import numpy as np
        from sklearn.metrics import classification_report, accuracy_score
        y_pred = [preds[i] for i in valid_idx]
        y_true = [true_labels[i] for i in valid_idx]
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4)
        print(f"Ollama evaluation on {len(valid_idx)} / {len(preds)} recognized outputs")
        print("Accuracy:", acc)
        print("Classification Report:\n", report)
        with open(os.path.join(args.output_dir, "ollama_evaluation.txt"), "w", encoding="utf-8") as f:
            f.write(f"recognized_count: {len(valid_idx)}\naccuracy: {acc}\n\nreport:\n{report}\n\nsample_generated_outputs:\n")
            # 保存部分样例
            for i in range(min(20, len(preds))):
                f.write(f"idx={i} true={true_labels[i]} pred={preds[i]} gen={str(call_ollama_generate(val_df.reset_index(drop=True).loc[i,'text'], args.ollama_model, url=args.ollama_url))}\n")
        return

    # 原有 HF 训练流程继续（不变）
    # ...existing code...
    ds_train = Dataset.from_pandas(train_df.reset_index(drop=True))
    ds_val = Dataset.from_pandas(val_df.reset_index(drop=True))
    dataset = DatasetDict({"train": ds_train, "validation": ds_val})

    # tokenizer & model 兼容性尝试
    try:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token":"<pad>"})

    try:
        model = LlamaForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=3)
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=3)

    model.resize_token_embeddings(len(tokenizer))

    tokenized = preprocess_and_tokenize(dataset["train"], tokenizer, args.max_length)
    tokenized_val = preprocess_and_tokenize(dataset["validation"], tokenizer, args.max_length)
    # 把 label 加回去（datasets map 已移除其他列）
    tokenized = tokenized.add_column("labels", list(train_df["label"].astype(int)))
    tokenized_val = tokenized_val.add_column("labels", list(val_df["label"].astype(int)))

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print("Eval metrics:", eval_metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    preds_output = trainer.predict(tokenized_val)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids
    report = classification_report(labels, preds, digits=4)
    print("Classification Report:\n", report)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    main()