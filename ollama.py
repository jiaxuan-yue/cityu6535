import argparse
import os
import json
import time
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- 本地Ollama核心配置（需根据你的环境确认） --------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"  # Ollama默认本地服务端口（固定）
OLLAMA_BASE_MODEL = "llama3:8b"  # 本地基础模型（需先通过 `ollama pull llama3:8b` 下载）
FINETUNED_MODEL_NAME = "llama3:8b-model-compare-finetuned"  # 微调后模型的自定义名称
DATA_FILE = "train.csv"  # 本地数据文件（需与代码同目录，或写绝对路径）
LABEL_MAPPING = {0: "A更好", 1: "B更好", 2: "一样好"}  # 标签文本映射
LABEL_TO_OLLAMA = {0: "A", 1: "B", 2: "SAME"}  # 微调时传给Ollama的简洁标签

# -------------------------- 1. 数据处理工具函数 --------------------------
def read_data(path):
    """读取CSV/JSONL数据，兼容两种格式"""
    if path.endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8")
    if path.endswith(".jsonl") or path.endswith(".json"):
        return pd.read_json(path, lines=True, encoding="utf-8")
    raise ValueError("仅支持 CSV 或 JSONL/JSON 文件，请检查数据格式")

def truthy(value):
    """判断值是否为“真”（用于解析标注的winner标签）"""
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value) != 0
    value_str = str(value).strip().lower()
    return value_str in ("1", "true", "yes", "y", "t", "1.0")

def infer_true_label(row):
    """从数据行中提取真实标签（0=A更好，1=B更好，2=一样好）"""
    # 优先解析多列标注（winner_model_a/winner_model_b/winner_tie）
    if truthy(row.get("winner_model_a", None)):
        return 0
    if truthy(row.get("winner_model_b", None)):
        return 1
    if truthy(row.get("winner_tie", None)):
        return 2
    # 兼容单列标注（仅winner列）
    winner_col = row.get("winner", None)
    if pd.notna(winner_col):
        winner_str = str(winner_col).strip().lower()
        if winner_str in ("a", "model_a", "model a", "winner a"):
            return 0
        if winner_str in ("b", "model_b", "model b", "winner b"):
            return 1
        if winner_str in ("tie", "same", "equal", "both"):
            return 2
    # 若无法解析，抛出错误（需检查数据标注格式）
    raise ValueError(f"无法解析标签！行ID：{row.get('id', '未知')}，请检查标注列")

def build_ollama_finetune_sample(row):
    """构建Ollama微调要求的样本格式（JSONL：{"prompt": "...", "response": "..."}）"""
    prompt = str(row.get("prompt", "")).strip()
    response_a = str(row.get("response_a", "")).strip()
    response_b = str(row.get("response_b", "")).strip()
    true_label = row["true_label"]  # 已通过infer_true_label获取
    
    # 微调提示词（明确任务，让模型学习判断逻辑）
    finetune_prompt = f"""任务：对比两个模型的回答，判断哪个更符合用户提示。
用户提示：{prompt}
模型A的回答：{response_a}
模型B的回答：{response_b}
要求：仅输出最终判断结果（A=A更好，B=B更好，SAME=两者一样好），不要额外文字。
判断结果："""
    
    # Ollama微调要求的键必须是"prompt"和"response"
    return {
        "prompt": finetune_prompt,
        "response": LABEL_TO_OLLAMA[true_label]
    }

def save_finetune_data(samples, save_path):
    """将微调样本保存为Ollama支持的JSONL格式"""
    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"✅ 微调数据已保存到：{save_path}")
    return save_path

# -------------------------- 2. Ollama服务交互函数 --------------------------
def check_ollama_service():
    """检查本地Ollama服务是否已启动（必须先启动服务才能运行）"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ 本地Ollama服务已正常启动")
            return True
        else:
            print(f"❌ Ollama服务响应异常，状态码：{response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 未检测到Ollama服务！请先启动服务：")
        print("  1. 打开CMD命令提示符")
        print("  2. 输入命令：ollama serve（不要关闭此CMD窗口）")
        print("  3. 重新运行本代码")
        return False

def pull_ollama_base_model(model_name):
    """拉取基础模型（如果本地没有），确保微调能正常进行"""
    print(f"\n🔍 检查本地是否存在基础模型：{model_name}")
    # 先查询本地已有的模型
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        local_models = [m["name"] for m in response.json().get("models", [])]
        if model_name in local_models:
            print(f"✅ 基础模型 {model_name} 已存在于本地")
            return True
    except Exception as e:
        print(f"⚠️ 查询本地模型时出错：{str(e)}，将尝试直接拉取")
    
    # 本地没有模型，开始拉取（可能需要几分钟，取决于网络和模型大小）
    print(f"📥 开始拉取基础模型 {model_name}（请勿中断）...")
    payload = {"name": model_name}
    try:
        # Ollama拉取是流式响应，实时打印进度
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/pull", 
            json=payload, 
            stream=True, 
            timeout=3600  # 超时设为1小时，避免大模型拉取中断
        ) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    # 打印关键状态（如下载进度、解压状态）
                    if "status" in data:
                        print(f"  拉取状态：{data['status']}", end="\r")
                    # 拉取完成的标志
                    if "completed" in data and data["completed"]:
                        print(f"\n✅ 基础模型 {model_name} 拉取完成")
                        return True
        print(f"❌ 模型拉取未完成（未知原因）")
        return False
    except Exception as e:
        print(f"❌ 拉取模型时出错：{str(e)}，请检查网络后重试")
        return False

def finetune_with_ollama(base_model, finetuned_model, data_path):
    """调用Ollama API进行模型微调（核心步骤，使用本地GPU加速）"""
    print(f"\n🚀 开始微调模型：")
    print(f"  基础模型：{base_model}")
    print(f"  微调后模型名：{finetuned_model}")
    print(f"  微调数据路径：{data_path}")
    
    # Ollama微调参数（根据GPU内存调整，内存小则减小batch_size）
    finetune_payload = {
        "base_model": base_model,
        "name": finetuned_model,
        "data": data_path,
        "options": {
            "gpu": True,  # 强制启用GPU加速（Ollama会自动识别本地GPU）
            "batch_size": 2,  # 批次大小（16GB内存建议2-4，32GB建议4-8）
            "epochs": 3,  # 训练轮数（数据量小时可增加到5，避免过拟合）
            "learning_rate": 1e-4  # 学习率（Ollama默认1e-4，无需频繁修改）
        }
    }
    
    try:
        # 发送微调请求（流式响应，实时打印训练日志）
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/finetune", 
            json=finetune_payload, 
            stream=True, 
            timeout=3600  # 微调超时设为1小时
        ) as response:
            if response.status_code != 200:
                error_msg = response.json().get("error", "未知错误")
                print(f"❌ 微调请求被拒绝：{error_msg}")
                return False
            
            print("\n📊 微调实时日志（按Ctrl+C可中断）：")
            for line in response.iter_lines():
                if line:
                    log_data = json.loads(line.decode("utf-8"))
                    # 打印训练状态（如epoch、loss、进度）
                    if "status" in log_data:
                        print(f"[{time.strftime('%H:%M:%S')}] {log_data['status']}")
                    # 打印训练指标（loss、accuracy，部分模型支持）
                    if "metrics" in log_data:
                        metrics = log_data["metrics"]
                        print(f"[{time.strftime('%H:%M:%S')}] 训练指标：{json.dumps(metrics, ensure_ascii=False)}")
                    # 微调完成标志
                    if "completed" in log_data and log_data["completed"]:
                        print(f"\n🎉 模型微调完成！微调后模型：{finetuned_model}")
                        print(f"   可通过命令查看：ollama list | findstr {finetuned_model}")
                        return True
        print("❌ 微调过程意外中断")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  已手动中断微调，模型可能未保存")
        return False
    except Exception as e:
        print(f"❌ 微调时出错：{str(e)}")
        return False

def ollama_infer(model_name, prompt):
    """调用微调后的Ollama模型进行推理（预测测试集标签）"""
    infer_payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 10,  # 仅需输出A/B/SAME，限制 tokens 避免冗余
        "temperature": 0.1,  # 低温度=输出更稳定（避免随机结果）
        "stream": False,  # 非流式响应（直接获取完整结果）
        "stop": ["\n"]  # 遇到换行符停止，避免多余输出
    }
    
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=infer_payload, timeout=15)
        if response.status_code != 200:
            error_msg = response.json().get("error", "未知错误")
            print(f"⚠️ 推理失败：{error_msg}")
            return None
        # 提取模型输出（Ollama返回的"response"字段）
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"⚠️ 推理时出错：{str(e)}")
        return None

# -------------------------- 3. 评估与可视化函数 --------------------------
def parse_ollama_prediction(pred_text):
    """将Ollama的输出文本（A/B/SAME）解析为数字标签（0/1/2）"""
    if not pred_text:
        return -1  # 无效预测（空输出）
    pred_text_upper = pred_text.strip().upper()
    # 精确匹配
    if pred_text_upper == "A":
        return 0
    elif pred_text_upper == "B":
        return 1
    elif pred_text_upper == "SAME":
        return 2
    # 模糊匹配（应对模型可能的多余输出，如"A "或"A\n"）
    elif "A" in pred_text_upper and "B" not in pred_text_upper and "SAME" not in pred_text_upper:
        return 0
    elif "B" in pred_text_upper and "A" not in pred_text_upper and "SAME" not in pred_text_upper:
        return 1
    elif "SAME" in pred_text_upper or "TIE" in pred_text_upper:
        return 2
    # 无法解析的输出
    else:
        print(f"⚠️ 无法解析预测结果：{pred_text}，标记为无效")
        return -1

def calculate_evaluation_metrics(true_labels, pred_labels):
    """计算核心评估指标：准确率、加权F1、分类报告、混淆矩阵"""
    # 过滤无效预测（pred_label=-1的样本）
    valid_mask = [p != -1 for p in pred_labels]
    valid_true = [t for t, v in zip(true_labels, valid_mask) if v]
    valid_pred = [p for p, v in zip(pred_labels, valid_mask) if v]
    
    if len(valid_true) == 0:
        raise ValueError("❌ 没有有效预测结果，无法计算指标，请检查模型推理过程")
    
    # 计算指标
    accuracy = accuracy_score(valid_true, valid_pred)
    weighted_f1 = f1_score(valid_true, valid_pred, average="weighted")  # 多分类用加权F1
    class_report = classification_report(
        valid_true, valid_pred,
        target_names=LABEL_MAPPING.values(),
        digits=4  # 保留4位小数，精度更高
    )
    conf_matrix = confusion_matrix(valid_true, valid_pred)
    
    return {
        "valid_sample_count": len(valid_true),
        "total_sample_count": len(true_labels),
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix
    }

def plot_and_save_confusion_matrix(conf_matrix, save_path):
    """绘制混淆矩阵热力图并保存（直观展示分类效果）"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
    plt.figure(figsize=(8, 6))
    
    # 绘制热力图
    sns.heatmap(
        conf_matrix,
        annot=True,  # 显示数值
        fmt="d",  # 数值格式（整数）
        cmap="Blues",  # 颜色映射
        xticklabels=LABEL_MAPPING.values(),  # x轴标签（预测标签）
        yticklabels=LABEL_MAPPING.values(),  # y轴标签（真实标签）
        cbar_kws={"label": "样本数量"}  # 颜色条标签
    )
    
    # 设置图表标题和轴标签
    plt.title("模型混淆矩阵", fontsize=14, pad=20)
    plt.xlabel("预测标签", fontsize=12, labelpad=10)
    plt.ylabel("真实标签", fontsize=12, labelpad=10)
    plt.tight_layout()  # 自动调整布局，避免标签被截断
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # dpi=300保证高清
    plt.close()
    print(f"✅ 混淆矩阵已保存到：{save_path}")

def save_evaluation_results(results, metrics, save_dir):
    """保存推理结果和评估指标到文件（便于后续分析）"""
    # 1. 保存推理结果（CSV格式，包含真实标签、预测文本、预测标签）
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(save_dir, "test_inference_results.csv")
    results_df.to_csv(results_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 推理结果已保存到：{results_csv_path}")
    
    # 2. 保存评估指标（TXT格式，包含准确率、F1、分类报告）
    metrics_txt_path = os.path.join(save_dir, "evaluation_metrics.txt")
    with open(metrics_txtimport argparse
import os
import json
import time
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- 本地Ollama核心配置（需根据你的环境确认） --------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"  # Ollama默认本地服务端口（固定）
OLLAMA_BASE_MODEL = "llama3:8b"  # 本地基础模型（需先通过 `ollama pull llama3:8b` 下载）
FINETUNED_MODEL_NAME = "llama3:8b-model-compare-finetuned"  # 微调后模型的自定义名称
DATA_FILE = "train.csv"  # 本地数据文件（需与代码同目录，或写绝对路径）
LABEL_MAPPING = {0: "A更好", 1: "B更好", 2: "一样好"}  # 标签文本映射
LABEL_TO_OLLAMA = {0: "A", 1: "B", 2: "SAME"}  # 微调时传给Ollama的简洁标签

# -------------------------- 1. 数据处理工具函数 --------------------------
def read_data(path):
    """读取CSV/JSONL数据，兼容两种格式"""
    if path.endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8")
    if path.endswith(".jsonl") or path.endswith(".json"):
        return pd.read_json(path, lines=True, encoding="utf-8")
    raise ValueError("仅支持 CSV 或 JSONL/JSON 文件，请检查数据格式")

def truthy(value):
    """判断值是否为“真”（用于解析标注的winner标签）"""
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value) != 0
    value_str = str(value).strip().lower()
    return value_str in ("1", "true", "yes", "y", "t", "1.0")

def infer_true_label(row):
    """从数据行中提取真实标签（0=A更好，1=B更好，2=一样好）"""
    # 优先解析多列标注（winner_model_a/winner_model_b/winner_tie）
    if truthy(row.get("winner_model_a", None)):
        return 0
    if truthy(row.get("winner_model_b", None)):
        return 1
    if truthy(row.get("winner_tie", None)):
        return 2
    # 兼容单列标注（仅winner列）
    winner_col = row.get("winner", None)
    if pd.notna(winner_col):
        winner_str = str(winner_col).strip().lower()
        if winner_str in ("a", "model_a", "model a", "winner a"):
            return 0
        if winner_str in ("b", "model_b", "model b", "winner b"):
            return 1
        if winner_str in ("tie", "same", "equal", "both"):
            return 2
    # 若无法解析，抛出错误（需检查数据标注格式）
    raise ValueError(f"无法解析标签！行ID：{row.get('id', '未知')}，请检查标注列")

def build_ollama_finetune_sample(row):
    """构建Ollama微调要求的样本格式（JSONL：{"prompt": "...", "response": "..."}）"""
    prompt = str(row.get("prompt", "")).strip()
    response_a = str(row.get("response_a", "")).strip()
    response_b = str(row.get("response_b", "")).strip()
    true_label = row["true_label"]  # 已通过infer_true_label获取
    
    # 微调提示词（明确任务，让模型学习判断逻辑）
    finetune_prompt = f"""任务：对比两个模型的回答，判断哪个更符合用户提示。
用户提示：{prompt}
模型A的回答：{response_a}
模型B的回答：{response_b}
要求：仅输出最终判断结果（A=A更好，B=B更好，SAME=两者一样好），不要额外文字。
判断结果："""
    
    # Ollama微调要求的键必须是"prompt"和"response"
    return {
        "prompt": finetune_prompt,
        "response": LABEL_TO_OLLAMA[true_label]
    }

def save_finetune_data(samples, save_path):
    """将微调样本保存为Ollama支持的JSONL格式"""
    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"✅ 微调数据已保存到：{save_path}")
    return save_path

# -------------------------- 2. Ollama服务交互函数 --------------------------
def check_ollama_service():
    """检查本地Ollama服务是否已启动（必须先启动服务才能运行）"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ 本地Ollama服务已正常启动")
            return True
        else:
            print(f"❌ Ollama服务响应异常，状态码：{response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 未检测到Ollama服务！请先启动服务：")
        print("  1. 打开CMD命令提示符")
        print("  2. 输入命令：ollama serve（不要关闭此CMD窗口）")
        print("  3. 重新运行本代码")
        return False

def pull_ollama_base_model(model_name):
    """拉取基础模型（如果本地没有），确保微调能正常进行"""
    print(f"\n🔍 检查本地是否存在基础模型：{model_name}")
    # 先查询本地已有的模型
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        local_models = [m["name"] for m in response.json().get("models", [])]
        if model_name in local_models:
            print(f"✅ 基础模型 {model_name} 已存在于本地")
            return True
    except Exception as e:
        print(f"⚠️ 查询本地模型时出错：{str(e)}，将尝试直接拉取")
    
    # 本地没有模型，开始拉取（可能需要几分钟，取决于网络和模型大小）
    print(f"📥 开始拉取基础模型 {model_name}（请勿中断）...")
    payload = {"name": model_name}
    try:
        # Ollama拉取是流式响应，实时打印进度
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/pull", 
            json=payload, 
            stream=True, 
            timeout=3600  # 超时设为1小时，避免大模型拉取中断
        ) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    # 打印关键状态（如下载进度、解压状态）
                    if "status" in data:
                        print(f"  拉取状态：{data['status']}", end="\r")
                    # 拉取完成的标志
                    if "completed" in data and data["completed"]:
                        print(f"\n✅ 基础模型 {model_name} 拉取完成")
                        return True
        print(f"❌ 模型拉取未完成（未知原因）")
        return False
    except Exception as e:
        print(f"❌ 拉取模型时出错：{str(e)}，请检查网络后重试")
        return False

def finetune_with_ollama(base_model, finetuned_model, data_path):
    """调用Ollama API进行模型微调（核心步骤，使用本地GPU加速）"""
    print(f"\n🚀 开始微调模型：")
    print(f"  基础模型：{base_model}")
    print(f"  微调后模型名：{finetuned_model}")
    print(f"  微调数据路径：{data_path}")
    
    # Ollama微调参数（根据GPU内存调整，内存小则减小batch_size）
    finetune_payload = {
        "base_model": base_model,
        "name": finetuned_model,
        "data": data_path,
        "options": {
            "gpu": True,  # 强制启用GPU加速（Ollama会自动识别本地GPU）
            "batch_size": 2,  # 批次大小（16GB内存建议2-4，32GB建议4-8）
            "epochs": 3,  # 训练轮数（数据量小时可增加到5，避免过拟合）
            "learning_rate": 1e-4  # 学习率（Ollama默认1e-4，无需频繁修改）
        }
    }
    
    try:
        # 发送微调请求（流式响应，实时打印训练日志）
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/finetune", 
            json=finetune_payload, 
            stream=True, 
            timeout=3600  # 微调超时设为1小时
        ) as response:
            if response.status_code != 200:
                error_msg = response.json().get("error", "未知错误")
                print(f"❌ 微调请求被拒绝：{error_msg}")
                return False
            
            print("\n📊 微调实时日志（按Ctrl+C可中断）：")
            for line in response.iter_lines():
                if line:
                    log_data = json.loads(line.decode("utf-8"))
                    # 打印训练状态（如epoch、loss、进度）
                    if "status" in log_data:
                        print(f"[{time.strftime('%H:%M:%S')}] {log_data['status']}")
                    # 打印训练指标（loss、accuracy，部分模型支持）
                    if "metrics" in log_data:
                        metrics = log_data["metrics"]
                        print(f"[{time.strftime('%H:%M:%S')}] 训练指标：{json.dumps(metrics, ensure_ascii=False)}")
                    # 微调完成标志
                    if "completed" in log_data and log_data["completed"]:
                        print(f"\n🎉 模型微调完成！微调后模型：{finetuned_model}")
                        print(f"   可通过命令查看：ollama list | findstr {finetuned_model}")
                        return True
        print("❌ 微调过程意外中断")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  已手动中断微调，模型可能未保存")
        return False
    except Exception as e:
        print(f"❌ 微调时出错：{str(e)}")
        return False

def ollama_infer(model_name, prompt):
    """调用微调后的Ollama模型进行推理（预测测试集标签）"""
    infer_payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 10,  # 仅需输出A/B/SAME，限制 tokens 避免冗余
        "temperature": 0.1,  # 低温度=输出更稳定（避免随机结果）
        "stream": False,  # 非流式响应（直接获取完整结果）
        "stop": ["\n"]  # 遇到换行符停止，避免多余输出
    }
    
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=infer_payload, timeout=15)
        if response.status_code != 200:
            error_msg = response.json().get("error", "未知错误")
            print(f"⚠️ 推理失败：{error_msg}")
            return None
        # 提取模型输出（Ollama返回的"response"字段）
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"⚠️ 推理时出错：{str(e)}")
        return None

# -------------------------- 3. 评估与可视化函数 --------------------------
def parse_ollama_prediction(pred_text):
    """将Ollama的输出文本（A/B/SAME）解析为数字标签（0/1/2）"""
    if not pred_text:
        return -1  # 无效预测（空输出）
    pred_text_upper = pred_text.strip().upper()
    # 精确匹配
    if pred_text_upper == "A":
        return 0
    elif pred_text_upper == "B":
        return 1
    elif pred_text_upper == "SAME":
        return 2
    # 模糊匹配（应对模型可能的多余输出，如"A "或"A\n"）
    elif "A" in pred_text_upper and "B" not in pred_text_upper and "SAME" not in pred_text_upper:
        return 0
    elif "B" in pred_text_upper and "A" not in pred_text_upper and "SAME" not in pred_text_upper:
        return 1
    elif "SAME" in pred_text_upper or "TIE" in pred_text_upper:
        return 2
    # 无法解析的输出
    else:
        print(f"⚠️ 无法解析预测结果：{pred_text}，标记为无效")
        return -1

def calculate_evaluation_metrics(true_labels, pred_labels):
    """计算核心评估指标：准确率、加权F1、分类报告、混淆矩阵"""
    # 过滤无效预测（pred_label=-1的样本）
    valid_mask = [p != -1 for p in pred_labels]
    valid_true = [t for t, v in zip(true_labels, valid_mask) if v]
    valid_pred = [p for p, v in zip(pred_labels, valid_mask) if v]
    
    if len(valid_true) == 0:
        raise ValueError("❌ 没有有效预测结果，无法计算指标，请检查模型推理过程")
    
    # 计算指标
    accuracy = accuracy_score(valid_true, valid_pred)
    weighted_f1 = f1_score(valid_true, valid_pred, average="weighted")  # 多分类用加权F1
    class_report = classification_report(
        valid_true, valid_pred,
        target_names=LABEL_MAPPING.values(),
        digits=4  # 保留4位小数，精度更高
    )
    conf_matrix = confusion_matrix(valid_true, valid_pred)
    
    return {
        "valid_sample_count": len(valid_true),
        "total_sample_count": len(true_labels),
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix
    }

def plot_and_save_confusion_matrix(conf_matrix, save_path):
    """绘制混淆矩阵热力图并保存（直观展示分类效果）"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
    plt.figure(figsize=(8, 6))
    
    # 绘制热力图
    sns.heatmap(
        conf_matrix,
        annot=True,  # 显示数值
        fmt="d",  # 数值格式（整数）
        cmap="Blues",  # 颜色映射
        xticklabels=LABEL_MAPPING.values(),  # x轴标签（预测标签）
        yticklabels=LABEL_MAPPING.values(),  # y轴标签（真实标签）
        cbar_kws={"label": "样本数量"}  # 颜色条标签
    )
    
    # 设置图表标题和轴标签
    plt.title("模型混淆矩阵", fontsize=14, pad=20)
    plt.xlabel("预测标签", fontsize=12, labelpad=10)
    plt.ylabel("真实标签", fontsize=12, labelpad=10)
    plt.tight_layout()  # 自动调整布局，避免标签被截断
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # dpi=300保证高清
    plt.close()
    print(f"✅ 混淆矩阵已保存到：{save_path}")

def save_evaluation_results(results, metrics, save_dir):
    """保存推理结果和评估指标到文件（便于后续分析）"""
    # 1. 保存推理结果（CSV格式，包含真实标签、预测文本、预测标签）
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(save_dir, "test_inference_results.csv")
    results_df.to_csv(results_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 推理结果已保存到：{results_csv_path}")
    
    # 2. 保存评估指标（TXT格式，包含准确率、F1、分类报告）
    metrics_txt_path = os.path.join(save_dir, "evaluation_metrics.txt")
    with open(metrics_txt