# Fine-Tuning Playground — 微調遊樂場

一套可直接用於生產環境的模組化程式碼，專為 **Qwen3-4B** 的監督式微調（Supervised Fine-Tuning, SFT）而設計，採用 **QLoRA / LoRA** 技術。專案基於 Hugging Face 的生態系統建構，包含 `transformers`、`peft`、`trl` 與 `bitsandbytes`，提供簡潔的命令列介面與可重複使用的 Python 模組，並可輕鬆改寫以適用其他僅解碼器（decoder-only）的大型語言模型。

---

## 目錄

- [專案概述](#專案概述)
- [功能特色](#功能特色)
- [硬體需求](#硬體需求)
- [安裝方式](#安裝方式)
- [專案結構](#專案結構)
- [設定參數](#設定參數)
- [資料格式](#資料格式)
  - [ChatML 格式](#chatml-格式)
  - [Alpaca 格式](#alpaca-格式)
  - [純文字格式](#純文字格式)
- [使用方式](#使用方式)
  - [基本訓練](#基本訓練)
  - [使用 Alpaca 資料集訓練](#使用-alpaca-資料集訓練)
  - [從 Hugging Face Hub 載入資料集](#從-hugging-face-hub-載入資料集)
  - [推論 — 微調前（基底模型）](#推論-微調前基底模型)
  - [推論 — 微調後](#推論-微調後)
  - [推論 — 對比微調前後](#推論-對比微調前後)
  - [啟用思考模式推論](#啟用思考模式推論)
- [進階用法](#進階用法)
  - [自訂 LoRA 目標模組](#自訂-lora-目標模組)
  - [推送至 Hugging Face Hub](#推送至-hugging-face-hub)
  - [完整精度 LoRA vs QLoRA](#完整精度-lora-vs-qlora)
  - [訓練時啟用思考模式](#訓練時啟用思考模式)
- [輸出產物](#輸出產物)
- [建議與最佳實踐](#建議與最佳實踐)
- [疑難排解](#疑難排解)
- [授權條款](#授權條款)

---

## 專案概述

**Qwen3-4B** 是阿里巴巴 Qwen 團隊推出的 40 億參數稠密語言模型，支援 32K 原生脈絡長度（可透過 YaRN 延伸至 131K），採用分組查詢注意力機制（32 個查詢頭 / 8 個鍵值頭），並具備雙模式推理系統，可在快速非思考回應與深度連鎖思考（chain-of-thought）之間切換。

本專案提供一套完整的微調流程，讓您能將 Qwen3-4B 調整為符合自身任務與領域需求的模型。其核心技術包括：

- **QLoRA**（4-bit NormalFloat 量化）：將記憶體需求降至約 6 GB VRAM，使消費級 GPU（如 RTX 3060 / 4060 / 4070）也能順利進行微調。
- **`trl.SFTTrainer`**：提供監督式微調的完整實作，內建序列打包（packing）、對話模板整合與評估迴圈。
- **全部 7 個線性投影層**（`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`）作為 LoRA 目標，確保在 4B 規模模型上達到最佳適應效果。
[README.zh-Hant.md](README.zh-Hant.md)
---

## 功能特色

- **QLoRA / LoRA 雙模式支援** — 使用 4-bit 量化可在 6 GB GPU 上訓練，關閉量化則可使用完整 FP16 LoRA。
- **多種資料格式** — 支援 ChatML（原生格式）、Alpaca（instruction/input/output）以及純文字格式。
- **Hugging Face Hub 整合** — 直接從 Hub 載入資料集，並將訓練完成的適配器推送回 Hub。
- **思考模式** — 可選擇在訓練與推論時啟用 Qwen3 的 `<think>` 推理標籤。
- **命令列優先設計** — 所有超參數皆可透過命令列旗標設定，無需撰寫 YAML 設定檔。
- **模組化 Python API** — 可直接匯入各模組（`config`、`data_utils`、`model_utils`、`train`、`inference`）進行程式化呼叫。
- **微調前後對比** — 可先測試微調前的基底模型，再與微調後的模型進行逐題對比，量化改善幅度。
- **評估資料分割** — 自動將資料集分割為訓練集與測試集，並根據評估損失選取最佳模型。
- **梯度檢查點** — 預設啟用，可降低訓練過程中的記憶體使用量。

---

## 硬體需求

| 設定 | 最低 VRAM | 建議 GPU |
|---|---|---|
| QLoRA（4-bit，batch size 2） | ~6 GB | RTX 3060 / 4060 / 4070、T4 |
| LoRA（16-bit，batch size 1） | ~12 GB | RTX 3080 / 4070 Ti / 4080、A10 |
| 完整微調（16-bit） | ~24 GB | RTX 4090 / A100 |

不支援僅使用 CPU 進行訓練，因為對於 40 億參數的模型來說速度過慢。

---

## 安裝方式

建議使用 **Python 3.11–3.13**。3.14 以上版本在 PyPI 上可能只有 CPU 版 PyTorch。

```bash
# 複製倉庫
git clone https://github.com/your-username/fine-tuning-playground.git
cd fine-tuning-playground

# 建立虛擬環境（建議）
python -m venv venv
source venv/bin/activate  # Linux / macOS
# venv\Scripts\activate   # Windows

# 安裝依賴套件
pip install -r requirements.txt

# 或使用輔助腳本
# bash scripts/install_deps.sh        # Linux / macOS / Git Bash
# powershell -File scripts/install_deps.ps1  # Windows PowerShell
```

驗證 PyTorch 是否能看到 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

`requirements.txt` 包含 CUDA 版 PyTorch（cu128）與全部專案依賴。若僅需 CPU 版（4B 模型會非常慢），請先從 PyPI 安裝 PyTorch，再安裝其餘套件（不含 `bitsandbytes`）。

核心套件：

| 套件 | 最低版本 | 用途 |
|---|---|---|
| `torch` | >= 2.4.0 | 深度學習框架（`requirements.txt` 預設 CUDA cu128） |
| `transformers` | >= 4.47.0 | 模型載入與分詞（Qwen3 支援） |
| `peft` | >= 0.13.0 | LoRA / QLoRA 適配器設定 |
| `trl` | >= 0.9.0 | 監督式微調的 SFTTrainer |
| `accelerate` | >= 1.0.0 | 分散式訓練支援 |
| `bitsandbytes` | >= 0.43.0 | 4-bit 量化（QLoRA） |
| `datasets` | >= 3.0.0 | 資料集載入與處理 |

---

## 專案結構

```
fine-tuning-playground/
├── requirements.txt      # Python 套件依賴列表
├── config.py             # FinetuneConfig 資料類別，包含所有超參數
├── data/                 # 訓練資料集與產生器
├── models/               # 微調後的 LoRA 適配器（已加入 .gitignore）
├── data_utils.py         # 資料集載入（Hub 與本機）+ ChatML 格式化
├── model_utils.py        # 分詞器載入、4-bit 量化、LoRA 設定
├── train.py              # SFTTrainer 訓練流程
├── inference.py          # 微調前/後推論 + 對比功能
├── run_sft.py            # 命令列進入點（argparse）
└── README.zh-Hant.md     # 本文件（繁體中文）
```

### 各檔案說明

**`config.py`** — 定義 `FinetuneConfig`，一個將所有超參數分組的 Python 資料類別，包含模型設定、LoRA 設定、資料設定、訓練設定、Hub 設定與推論設定。內含 `from_args()` 類方法，可將 `argparse.Namespace` 轉換為設定物件。

**`data_utils.py`** — 提供三種資料載入路徑（`load_dataset_from_hub`、`load_dataset_from_json`）與格式轉換器（`convert_to_chat_format`），可將 Alpaca、純文字與 ChatML 格式統一轉換為 `tokenizer.apply_chat_template` 所需的訊息列表。

**`model_utils.py`** — 處理模型準備的核心工作：`load_tokenizer`（自動修正 pad_token）、`load_quantized_model`（可選的 BitsAndBytes 4-bit 設定）、`setup_lora_config`（預設鎖定全部 7 個線性層）、以及用於除錯的 `print_trainable_params`。

**`train.py`** — 核心訓練協調模組。依序載入模型與資料、套用 LoRA、設定 `SFTTrainer`（含梯度檢查點與評估迴圈），最後儲存適配器與分詞器。支援選擇性推送到 Hugging Face Hub。

**`inference.py`** — 獨立推論模組，同時支援微調前後的模型測試。`load_base_model` 載入未附加任何適配器的原始 Qwen3 模型，用於微調前測試。`load_finetuned_model` 載入基底模型與已訓練的 LoRA 適配器。`generate_response` 套用對話模板、生成 tokens 並解碼結果。`extract_thinking` 可在啟用思考模式時解析 `<think>...</think>` 區塊。`compare_responses` 會同時載入兩個模型，對同一組提示進行並排對比。

**`run_sft.py`** — 使用 `argparse` 的命令列腳本。所有 `FinetuneConfig` 欄位皆對應為命令列旗標，並附有合理的預設值。這是大多數使用者的建議進入點。

---

## 設定參數

所有超參數皆定義於 `FinetuneConfig`（`config.py`）。以下是完整的參數參考表：

| 類別 | 參數 | 預設值 | 說明 |
|---|---|---|---|
| **模型** | `model_id` | `Qwen/Qwen3-4B` | Hugging Face 模型識別碼 |
| | `load_in_4bit` | `True` | 啟用 4-bit NF4 量化 |
| | `torch_dtype` | `bfloat16` | 計算精度 |
| **LoRA** | `lora_r` | `16` | LoRA 秩（rank） |
| | `lora_alpha` | `32` | LoRA 縮放因子（通常設為 2 * r） |
| | `lora_dropout` | `0.05` | LoRA 層的 dropout 比率 |
| | `target_modules` | 全部 7 個線性層 | 要附加適配器的模組名稱 |
| **資料** | `dataset_path` | （必填） | 本機檔案路徑或 HF 資料集名稱 |
| | `dataset_format` | `chat` | 輸入格式：chat、alpaca 或 text |
| | `test_split` | `0.05` | 保留為評估集的比例（0 = 不評估） |
| | `max_seq_length` | `4096` | 每個範例的最大 token 數 |
| **訓練** | `learning_rate` | `2e-4` | 峰值學習率 |
| | `num_epochs` | `3` | 訓練週期數 |
| | `per_device_batch_size` | `2` | 每張 GPU 的批次大小 |
| | `gradient_accum_steps` | `8` | 梯度累積步數 |
| | `warmup_ratio` | `0.03` | 線性 warmup 比例 |
| | `logging_steps` | `10` | 每隔 N 步記錄一次指標 |
| | `save_steps` | `200` | 每隔 N 步儲存一次檢查點 |
| | `output_dir` | `./models/qwen3-4b-finetuned` | 輸出目錄 |
| **Hub** | `push_to_hub` | `False` | 將適配器推送至 HF Hub |
| | `hub_model_id` | `""` | Hub 上的目標儲存庫名稱 |
| **推論** | `max_new_tokens` | `2048` | 最大生成 token 數 |
| | `temperature` | `0.7` | 取樣溫度 |
| | `top_p` | `0.9` | 核取樣（nucleus sampling）閾值 |
| | `use_thinking` | `False` | 啟用思考模式 |

---

## 資料格式

本流程支援三種輸入格式，內部皆會正規化為 ChatML 訊息格式。

### ChatML 格式

這是 Qwen3 的原生格式。每個 JSON 物件（JSONL 中每行一個，或 JSON 中作為列表元素）包含 `"messages"` 鍵：

```json
{
  "messages": [
    {"role": "system", "content": "你是一位樂於助人的助手。"},
    {"role": "user", "content": "法國的首都是哪裡？"},
    {"role": "assistant", "content": "法國的確是巴黎。"}
  ]
}
```

系統訊息（system message）為選填。此格式使用 `--dataset_format chat`（預設值）。

### Alpaca 格式

Stanford Alpaca 格式使用三個欄位：`instruction`、`input` 與 `output`。`input` 欄位可為空字串。

```json
{
  "instruction": "解釋遞迴的概念。",
  "input": "",
  "output": "遞迴是一種程式設計技巧，函式透過呼叫自身來解決相同問題的較小實例..."
}
```

```json
{
  "instruction": "將以下句子翻譯成法文。",
  "input": "你好，最近怎麼樣？",
  "output": "Bonjour, comment allez-vous ?"
}
```

此格式需使用 `--dataset_format alpaca`。

### 純文字格式

一種扁平的文字格式，每個範例的 `"text"` 欄位包含完整的對話內容。流程會以 `<|im_end|>\n<|im_start|>` 為分隔來還原使用者與助理的交替發言：

```json
{
  "text": "<|im_start|>user\n什麼是機器學習？<|im_end|>\n<|im_start|>assistant\n機器學習是人工智慧的一個子領域...<|im_end|>"
}
```

此格式需使用 `--dataset_format text`。

---

## 使用方式

### 基本訓練

使用本機 JSONL 檔案（每行一則 ChatML 格式的對話）進行訓練：

```bash
python run_sft.py \
    --dataset_path ./data/my_dataset.jsonl \
    --output_dir ./my-finetuned-model
```

### 使用 Alpaca 資料集訓練

```bash
python run_sft.py \
    --dataset_path ./data/alpaca_data.json \
    --dataset_format alpaca \
    --lr 2e-4 \
    --epochs 5 \
    --output_dir ./alpaca-finetuned
```

### 從 Hugging Face Hub 載入資料集

直接從 Hugging Face Hub 載入資料集：

```bash
python run_sft.py \
    --dataset_path databricks/databricks-dolly-15k \
    --dataset_format alpaca \
    --max_seq_length 2048 \
    --output_dir ./dolly-finetuned
```

### 推論 — 微調前（基底模型）

測試微調前的原始 Qwen3-4B 模型，建立基準線：

```python
from inference import load_base_model, generate_response

# 載入未附加任何適配器的基底模型
model, tokenizer = load_base_model(
    base_model_id="Qwen/Qwen3-4B",
    use_4bit=True,
)

messages = [{"role": "user", "content": "請介紹國立臺灣師範大學。"}]

response = generate_response(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
)

print(response)
```

或透過命令列：

```bash
python inference.py --mode base --prompt "請介紹國立臺灣師範大學。"
```

### 推論 — 微調後

載入微調後的 LoRA 適配器並生成回應：

```python
from inference import load_finetuned_model, generate_response

model, tokenizer = load_finetuned_model(
    base_model_id="Qwen/Qwen3-4B",
    adapter_path="./my-finetuned-model",
    use_4bit=True,
)

messages = [{"role": "user", "content": "請介紹國立臺灣師範大學。"}]

response = generate_response(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
)

print(response)
```

或透過命令列：

```bash
python inference.py --mode finetuned --adapter_path ./my-finetuned-model
```

### 推論 — 對比微調前後

用**同一組提示**分別餵給兩個模型，觀察微調帶來的改善：

```python
from inference import compare_responses

results = compare_responses(
    base_model_id="Qwen/Qwen3-4B",
    adapter_path="./my-finetuned-model",
    test_prompts=[
        [{"role": "user", "content": "請介紹國立臺灣師範大學。"}],
        [{"role": "user", "content": "臺師大的歷史是什麼？"}],
    ],
    use_4bit=True,
)
```

或透過命令列：

```bash
python inference.py --mode compare --adapter_path ./my-finetuned-model
```

### 啟用思考模式推論

若要啟用 Qwen3 的連鎖思考（chain-of-thought）推理，請設定 `enable_thinking=True`：

```python
from inference import load_finetuned_model, generate_response, extract_thinking

model, tokenizer = load_finetuned_model("Qwen/Qwen3-4B", "./my-finetuned-model")

response = generate_response(
    model=model,
    tokenizer=tokenizer,
    messages=[{"role": "user", "content": "請問「strawberry」這個字有幾個 r？"}],
    enable_thinking=True,
)

# 分別取出推理過程與最終答案
thinking, answer = extract_thinking(response)
if thinking:
    print("=== 推理過程 ===")
    print(thinking)
    print("=== 最終答案 ===")
    print(answer)
else:
    print(response)
```

---

## 進階用法

### 自訂 LoRA 目標模組

預設情況下，LoRA 會套用至全部 7 個線性投影層：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。您可以將範圍限縮至僅注意力模組以節省記憶體：

```bash
python run_sft.py \
    --dataset_path ./data/train.jsonl \
    --target_modules "[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\"]" \
    --output_dir ./attention-only-lora
```

注意：透過命令列傳遞 `target_modules` 時，引數必須為合法的 Python 列表字面值。

### 推送至 Hugging Face Hub

若要將訓練完成的適配器直接上傳至 Hugging Face Hub，需先登入：

```bash
huggingface-cli login
```

然後加上 `--push_to_hub` 與 `--hub_model_id` 旗標：

```bash
python run_sft.py \
    --dataset_path ./data/train.jsonl \
    --output_dir ./my-adapter \
    --push_to_hub True \
    --hub_model_id your-username/qwen3-4b-finetuned
```

### 完整精度 LoRA vs QLoRA

QLoRA（4-bit）為預設模式，所需記憶體最少。若要在不使用量化的情況下以 FP16 LoRA 訓練：

```bash
python run_sft.py \
    --dataset_path ./data/train.jsonl \
    --load_in_4bit False \
    --per_device_batch_size 1 \
    --output_dir ./fplora-finetuned
```

這需要約 12 GB VRAM。若遇到記憶體不足的錯誤，請降低 `per_device_batch_size` 或 `max_seq_length`。

### 訓練時啟用思考模式

如果您的資料集中包含推理過程（內容包覆在 `<think>...</think>` 中），可以啟用支援思考模式的對話模板：

```bash
python run_sft.py \
    --dataset_path ./data/reasoning_dataset.jsonl \
    --use_thinking True \
    --output_dir ./thinking-finetuned
```

這樣可確保分詞器在訓練時正確處理 `<think>` 標籤。

---

## 輸出產物

訓練完成後，`output_dir` 目錄將包含以下檔案：

```
my-finetuned-model/
├── adapter_config.json         # LoRA 設定（rank、alpha、目標模組等）
├── adapter_model.safetensors   # 訓練完成的 LoRA 適配器權重
├── tokenizer_config.json       # 分詞器設定
├── tokenizer.json              # 分詞器詞彙表
├── special_tokens_map.json     # 特殊 token 定義
└── (checkpoint-XXX/)           # 訓練過程中的中間檢查點（如有設定 save_steps）
```

適配器僅約 40–80 MB，便於分享與版本管理。

---

## 建議與最佳實踐

1. **從小處著手**：先使用 `--max_seq_length 2048` 與 `--epochs 1` 進行快速測試，確認資料流程正常後再投入完整的訓練。

2. **有效批次大小**：有效批次大小為 `per_device_batch_size * gradient_accum_steps * num_gpus`。建議目標值為 16–32。以 1 張 GPU 為例，`--per_device_batch_size 2 --gradient_accum_steps 8` 的有效批次大小即為 16。

3. **學習率**：對於 LoRA/QLoRA，`2e-4` 是穩健的預設值。若資料集非常小（少於 500 筆），建議降至 `1e-4` 以避免過擬合。

4. **監控評估損失**：Trainer 會自動追蹤評估損失並儲存最佳檢查點。若評估損失持續上升，可能是過擬合或學習率過高的徵兆。

5. **序列打包**：SFTTrainer 會自動將序列打包至 `max_seq_length`，可提升訓練效率，無需特別設定。

6. **混合精度**：`bfloat16` 在訓練穩定性上優於 `float16`。若 GPU 不支援 bf16，請使用 `--torch_dtype float16`。

7. **資料品質重於數量**：對於指令微調而言，500–1000 筆高品質的乾淨資料集通常勝過 10,000 筆雜訊資料。

---

## 疑難排解

| 問題 | 可能原因 | 解決方案 |
|---|---|---|
| 有 GPU 但訓練仍走 CPU | 安裝了 CPU 版 PyTorch（`torch x.x.x+cpu`） | 使用 Python 3.11–3.13，再執行 `pip install -r requirements.txt` |
| `OutOfMemoryError` | 批次大小超出 GPU 記憶體 | 降低 `--per_device_batch_size` 或 `--max_seq_length` |
| `KeyError: 'qwen3'` | Transformers 版本過舊 | 升級至 `transformers>=4.47.0` |
| bitsandbytes 匯入錯誤 | 缺少 CUDA 或版本不相容 | 確認 bitsandbytes 與 CUDA 版本匹配：`pip install bitsandbytes --force-reinstall` |
| 訓練損失出現 NaN | 學習率過高或資料類型問題 | 調低 `--lr` 或改用 `--torch_dtype bfloat16` |
| 模型不斷重複相同句子 | 過擬合或溫度過低 | 提高 `--temperature` 或減少 `--epochs` |
| `apply_chat_template` 錯誤 | 資料格式不符合預期 | 檢查資料集是否使用正確的格式（請見[資料格式](#資料格式)） |
| 評估損失持續上升 | 過擬合 | 減少 `--epochs`、增加 `--test_split`，或加入更多訓練資料 |

---

## 授權條款

本專案採用 [MIT 授權條款](LICENSE)。Qwen3 模型本身則以 Apache 2.0 授權條款釋出。
