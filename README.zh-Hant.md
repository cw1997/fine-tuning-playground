# Fine-Tuning Playground — 微調遊樂場

一套面向教學的程式碼，專為 **Qwen3 / Qwen3.5** 僅解碼器（decoder-only）模型（例如 **Qwen3-4B**、**Qwen3.5-4B**）的監督式微調（Supervised Fine-Tuning, SFT）而設計，採用 **QLoRA / LoRA** 技術。專案基於 Hugging Face 的生態系統建構，完整訓練流程集中在單一腳本 `src/train.py` 中，由上而下順序執行，所有設定皆以命令列參數傳入。

---

## 目錄

- [專案概述](#專案概述)
- [功能特色](#功能特色)
- [硬體需求](#硬體需求)
- [安裝方式](#安裝方式)
- [快速開始](#快速開始)
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

**Qwen3** 與 **Qwen3.5** 是阿里巴巴 Qwen 團隊推出的稠密語言模型。4B 級別的型號支援長脈絡、分組查詢注意力，並可選擇啟用連鎖思考（chain-of-thought）**思考模式**（透過對話模板中的 `<think>` 標籤）。

本專案提供一套完整的微調流程，讓您能將這些模型調整為符合自身任務與領域需求的版本。其核心技術包括：

- **QLoRA**（4-bit NormalFloat 量化）：將記憶體需求降至約 6 GB VRAM，使消費級 GPU（如 RTX 3060 / 4060 / 4070）也能順利進行微調。
- **`trl.SFTTrainer`**：提供監督式微調的完整實作，內建對話模板整合與評估迴圈。
- **全部 7 個線性投影層**（`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`）作為 LoRA 目標，確保在 4B 規模模型上達到最佳適應效果。

---

## 功能特色

- **QLoRA / LoRA 雙模式支援** — 使用 4-bit 量化可在 6 GB GPU 上訓練，關閉量化則可使用完整 FP16 LoRA。
- **多種資料格式** — 支援 ChatML（原生格式）、Alpaca（instruction/input/output）以及純文字格式。
- **Hugging Face Hub 整合** — 直接從 Hub 載入資料集，並將訓練完成的適配器推送回 Hub。
- **思考模式** — 可選擇在訓練與推論時啟用 Qwen3 的 `<think>` 推理標籤。
- **命令列優先設計** — 所有超參數皆可透過命令列旗標設定，無需撰寫 YAML 設定檔。
- **CPU 備援** — 若無 CUDA，訓練與推論會自動改以 CPU 全精度執行（速度極慢，僅建議用於冒煙測試）。
- **單檔訓練腳本** — `src/train.py` 由上而下執行完整 SFT 流程，適合講解微調各步驟。
- **微調前後對比** — 可先測試微調前的基底模型，再與微調後的模型進行逐題對比，量化改善幅度。
- **評估資料分割** — 自動將資料集分割為訓練集與測試集，並根據評估損失選取最佳模型。
- **梯度檢查點** — 預設啟用，可降低訓練過程中的記憶體使用量。

---

## 硬體需求

| 設定 | 最低 VRAM | 建議 GPU |
|---|---|---|
| QLoRA（4-bit，batch size 2） | ~6 GB | RTX 3060 / 4060 / 4070、T4 |
| LoRA（16-bit，batch size 1） | ~12 GB | RTX 3080 / 4070 Ti / 4080、A10 |
| CPU（無量化，float32） | — | 任何 CPU（極慢，僅冒煙測試） |

強烈建議使用 GPU 訓練。在無 CUDA 的機器上，流程會自動關閉 4-bit 量化並以 CPU `float32` 執行。可用 `--device gpu` 或 `--device cpu` 覆寫自動偵測。

---

## 安裝方式

建議使用 **Python 3.11 或 3.12** 以獲得最佳 CUDA wheel 相容性。較新的 Python 版本在 PyPI 上可能只有 CPU 版 PyTorch。

### 方式一 — Miniconda（建議）

```bash
git clone https://github.com/cw1997/fine-tuning-playground.git
cd fine-tuning-playground

conda create -n finetune python=3.11 -y
conda activate finetune

pip install -r requirements.txt
```

### 方式二 — venv

```bash
git clone https://github.com/cw1997/fine-tuning-playground.git
cd fine-tuning-playground

python -m venv venv
source venv/bin/activate  # Linux / macOS
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 輔助腳本

```bash
bash scripts/install_deps.sh                 # Linux / macOS / Git Bash
powershell -File scripts/install_deps.ps1    # Windows PowerShell
```

驗證 PyTorch 是否能看到 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

`requirements.txt` 包含 CUDA 版 PyTorch（cu128）與全部專案依賴。若僅需 CPU 版（4B 模型會非常慢），請先從 PyPI 安裝 PyTorch，再安裝其餘套件（不含 `bitsandbytes`）。

核心套件：

| 套件 | 最低版本 | 用途 |
|---|---|---|
| `torch` | >= 2.4.0, < 2.12 | 深度學習框架（`requirements.txt` 預設 CUDA cu128） |
| `transformers` | >= 4.47.0 | 模型載入與分詞（Qwen3 支援） |
| `peft` | >= 0.13.0 | LoRA / QLoRA 適配器設定 |
| `trl` | >= 0.9.0 | 監督式微調的 SFTTrainer |
| `accelerate` | >= 1.0.0 | 分散式訓練支援 |
| `bitsandbytes` | >= 0.43.0 | 4-bit 量化（QLoRA） |
| `datasets` | >= 3.0.0 | 資料集載入與處理 |

---

## 快速開始

依下列步驟設定環境並開始訓練：

```bash
# 1. 複製並進入倉庫
git clone https://github.com/cw1997/fine-tuning-playground.git
cd fine-tuning-playground

# 2. 建立並啟用 conda 環境（或使用 venv — 見[安裝方式](#安裝方式)）
conda create -n finetune python=3.11 -y
conda activate finetune

# 3. 安裝依賴套件
pip install -r requirements.txt

# 4. 確認 GPU 可用
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# 5. 使用內附的臺師大資料集（161 筆 ChatML 範例）進行訓練
python src/train.py --model_id Qwen/Qwen3.5-4B --dataset_path ./data/ntnu_dataset.jsonl --output_dir ./models/ntnu-finetuned/qwen3.5-4b

# 6. 以基底模型推論（微調前）— 儲存輸出以便對比
python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B --prompt "台灣師范大學校本部地址在哪？"

# 7. 以微調後模型推論 — 與步驟 6 對比
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B --adapter_path ./models/ntnu-finetuned/qwen3.5-4b --prompt "台灣師范大學校本部地址在哪？"
```

詳細訓練選項與資料格式請見[使用方式](#使用方式)。

---

## 專案結構

```
fine-tuning-playground/
├── requirements.txt          # Python 套件依賴列表
├── src/                      # 原始碼腳本
│   ├── train.py              # 完整 SFT 流程（單檔、由上而下）
│   └── inference.py          # 微調前/後推論 + 對比功能
├── data/                     # 訓練資料集與產生器
│   ├── ntnu_dataset.jsonl    # 內附臺師大 ChatML 資料集（161 筆）
│   ├── generate_ntnu_dataset.py
│   └── ntnu_extended_records.py
├── models/                   # 微調後的 LoRA 適配器（已加入 .gitignore）
├── scripts/                  # install_deps.sh / install_deps.ps1
├── README.md                 # 英文文件
└── README.zh-Hant.md         # 本文件（繁體中文）
```

### 各檔案說明

**`src/train.py`** — 完整的監督式微調流程，全部寫在一個檔案中。依序執行：SSL 修復 → 解析命令列參數 → 偵測裝置 → 載入分詞器 → 載入基座模型（可選 4-bit）→ 套用 LoRA → 載入並格式化資料集 → 設定 `SFTTrainer` → 訓練 → 儲存適配器。所有超參數皆為 CLI 旗標，無獨立設定模組。

**`src/inference.py`** — 獨立推論腳本，支援基底模型、微調後模型或並排對比。CLI 旗標：`--mode`（`base` / `finetuned` / `compare`）、`--model_id`、`--adapter_path`、`--prompt`、`--load_in_4bit`、`--use_thinking`、`--max_new_tokens`、`--temperature`、`--top_p`。

**`data/`** — 包含內附的 `ntnu_dataset.jsonl`（161 筆關於國立臺灣師範大學的 ChatML 紀錄）以及重新產生資料集的腳本（`generate_ntnu_dataset.py`、`ntnu_extended_records.py`）。

---

## 設定參數

所有訓練超參數皆為 `src/train.py` 的命令列旗標。以下是完整的參數參考表：

| 類別 | 參數 | 預設值 | 說明 |
|---|---|---|---|
| **模型** | `model_id` | `Qwen/Qwen3.5-0.8B` | Hugging Face 模型識別碼（GPU 建議使用 `Qwen/Qwen3.5-4B` 或 `Qwen/Qwen3-4B`） |
| | `load_in_4bit` | `True` | 啟用 4-bit NF4 量化（CPU 上會自動關閉） |
| | `torch_dtype` | `bfloat16` | 計算精度（CPU 上會改為 `float32`） |
| **LoRA** | `lora_r` | `16` | LoRA 秩（rank） |
| | `lora_alpha` | `32` | LoRA 縮放因子（通常設為 2 * r） |
| | `lora_dropout` | `0.05` | LoRA 層的 dropout 比率 |
| | `target_modules` | 全部 7 個線性層 | 逗號分隔的模組名稱（例如 `q_proj,k_proj,v_proj,o_proj`） |
| **資料** | `dataset_path` | （必填） | 本機檔案路徑或 HF 資料集名稱（本機路徑須以 `./`、`.` 或 `/` 開頭） |
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
| **硬體** | `device` | `None`（自動） | 運算裝置：`gpu`、`cpu` 或自動偵測 |
| **Hub** | `push_to_hub` | `False` | 將適配器推送至 HF Hub |
| | `hub_model_id` | `""` | Hub 上的目標儲存庫名稱 |
| **推論** | `max_new_tokens` | `2048` | 最大生成 token 數 |
| | `temperature` | `0.7` | 取樣溫度 |
| | `top_p` | `0.9` | 核取樣（nucleus sampling）閾值 |
| | `use_thinking` | `False` | 啟用思考模式 |

> **備註：** `src/train.py` 的 `--model_id` 預設為 `Qwen/Qwen3.5-0.8B`（適合 CPU 冒煙測試）。GPU 訓練請傳入 `--model_id Qwen/Qwen3.5-4B` 或 `Qwen/Qwen3-4B`。`src/inference.py` 的 `--model_id` 預設為 `Qwen/Qwen3-4B` — 推論時請明確指定與微調時相同的模型。

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
python src/train.py \
    --model_id Qwen/Qwen3.5-4B \
    --dataset_path ./data/my_dataset.jsonl \
    --output_dir ./my-finetuned-model
```

### 使用 Alpaca 資料集訓練

```bash
python src/train.py \
    --dataset_path ./data/alpaca_data.json \
    --dataset_format alpaca \
    --lr 2e-4 \
    --epochs 5 \
    --output_dir ./alpaca-finetuned
```

### 從 Hugging Face Hub 載入資料集

直接從 Hugging Face Hub 載入資料集：

```bash
python src/train.py \
    --dataset_path databricks/databricks-dolly-15k \
    --dataset_format alpaca \
    --max_seq_length 2048 \
    --output_dir ./dolly-finetuned
```

### 推論 — 微調前（基底模型）

測試微調前的原始模型，建立基準線：

```bash
python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B --prompt "請介紹國立臺灣師範大學。"
```

### 推論 — 微調後

載入微調後的 LoRA 適配器並生成回應（`--model_id` 須與訓練時一致）：

```bash
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B --adapter_path ./my-finetuned-model --prompt "請介紹國立臺灣師範大學。"
```

### 推論 — 對比微調前後

用**同一組提示**分別餵給兩個模型，觀察微調帶來的改善：

```bash
python src/inference.py --mode compare --model_id Qwen/Qwen3.5-4B --adapter_path ./my-finetuned-model
```

### 啟用思考模式推論

若要啟用 Qwen3 的連鎖思考（chain-of-thought）推理，請傳入 `--use_thinking True`：

```bash
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
    --adapter_path ./my-finetuned-model \
    --use_thinking True \
    --prompt "請問「strawberry」這個字有幾個 r？"
```

---

## 進階用法

### 自訂 LoRA 目標模組

預設情況下，LoRA 會套用至全部 7 個線性投影層：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。您可以將範圍限縮至僅注意力模組以節省記憶體：

```bash
python src/train.py \
    --dataset_path ./data/train.jsonl \
    --model_id Qwen/Qwen3.5-4B \
    --target_modules q_proj,k_proj,v_proj,o_proj \
    --output_dir ./attention-only-lora
```

### 推送至 Hugging Face Hub

若要將訓練完成的適配器直接上傳至 Hugging Face Hub，需先登入：

```bash
huggingface-cli login
```

然後加上 `--push_to_hub` 與 `--hub_model_id` 旗標：

```bash
python src/train.py \
    --dataset_path ./data/train.jsonl \
    --output_dir ./my-adapter \
    --push_to_hub True \
    --hub_model_id your-username/qwen3-4b-finetuned
```

### 完整精度 LoRA vs QLoRA

QLoRA（4-bit）為預設模式，所需記憶體最少。若要在不使用量化的情況下以 FP16 LoRA 訓練：

```bash
python src/train.py \
    --dataset_path ./data/train.jsonl \
    --load_in_4bit False \
    --per_device_batch_size 1 \
    --output_dir ./fplora-finetuned
```

這需要約 12 GB VRAM。若遇到記憶體不足的錯誤，請降低 `per_device_batch_size` 或 `max_seq_length`。

### 訓練時啟用思考模式

如果您的資料集中包含推理過程（內容包覆在 `<think>...</think>` 中），可以啟用支援思考模式的對話模板：

```bash
python src/train.py \
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

5. **序列長度**：範例會透過 `SFTConfig.max_length` 截斷或填充至 `max_seq_length`。目前的訓練設定未啟用序列打包（packing）。

6. **混合精度**：在 GPU 上 `bfloat16` 的訓練穩定性優於 `float16`。若 GPU 不支援 bf16，請使用 `--torch_dtype float16`。在 CPU 上會自動改為 `float32`。

7. **資料品質重於數量**：對於指令微調而言，500–1000 筆高品質的乾淨資料集通常勝過 10,000 筆雜訊資料。

8. **推論時須對應 `model_id`**：`src/inference.py` 使用的基底模型（`--model_id`）必須與微調時的模型一致。

---

## 疑難排解

| 問題 | 可能原因 | 解決方案 |
|---|---|---|
| 有 GPU 但訓練仍走 CPU | 安裝了 CPU 版 PyTorch（`torch x.x.x+cpu`） | 使用 Python 3.11–3.12，再執行 `pip install -r requirements.txt` |
| `RuntimeError: GPU was requested via --device gpu` | PyTorch 無法使用 CUDA | 安裝 CUDA 版 PyTorch，或不要指定 `--device gpu` |
| `OutOfMemoryError` | 批次大小超出 GPU 記憶體 | 降低 `--per_device_batch_size` 或 `--max_seq_length` |
| `KeyError: 'qwen3'` | Transformers 版本過舊 | 升級至 `transformers>=4.47.0` |
| bitsandbytes 匯入錯誤 | 缺少 CUDA 或版本不相容 | 確認 bitsandbytes 與 CUDA 版本匹配：`pip install bitsandbytes --force-reinstall` |
| Hub 下載 / SSL 錯誤 | conda/Windows 上 `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` 損壞 | `src/train.py` 與 `src/inference.py` 啟動時會自動修復；或手動修正環境變數 |
| 訓練損失出現 NaN | 學習率過高或資料類型問題 | 調低 `--lr` 或改用 `--torch_dtype bfloat16` |
| 模型不斷重複相同句子 | 過擬合或溫度過低 | 提高 `--temperature` 或減少 `--epochs` |
| `apply_chat_template` 錯誤 | 資料格式不符合預期 | 檢查資料集是否使用正確的格式（請見[資料格式](#資料格式)） |
| 本機檔案被誤當成 Hub 資料集 | `dataset_path` 缺少 `./` 前綴 | 本機檔案請使用 `./data/train.jsonl`，勿寫成 `data/train.jsonl` |
| 評估損失持續上升 | 過擬合 | 減少 `--epochs`、增加 `--test_split`，或加入更多訓練資料 |
| `LoRA adapter not found` | `--adapter_path` 路徑錯誤 | 指向輸出目錄或 `checkpoint-*` 子目錄；推論會自動選取最新檢查點 |

---

## 授權條款

本專案採用 [MIT 授權條款](LICENSE)。Qwen3 / Qwen3.5 模型則依其在 Hugging Face 上的各自授權條款釋出（通常為 Apache 2.0）。

English documentation: [README.md](README.md)
